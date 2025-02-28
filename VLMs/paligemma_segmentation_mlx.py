"""Part of the code below is adapted from https://github.com/google-research/big_vision/blob/main/big_vision/evaluators/proj/paligemma/transfers/segmentation.py.
This version uses MLX instead of JAX/Flax.
"""

import functools
import logging
import re

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm import apply_chat_template, generate, load
from mlx_vlm.utils import load_image
from tensorflow.io import gfile

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

MODEL_PATH = "mlx-community/paligemma2-3b-mix-224-bf16"
IMAGE_PATH = "./images/android.png"
_KNOWN_MODELS = {
    # Trained on open images.
    "oi": "gs://big_vision/paligemma/vae-oid.npz",
}


class ResBlock(nn.Module):
    """Residual block for MLX."""

    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=features, out_channels=features, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=features, out_channels=features, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=features, out_channels=features, kernel_size=1, padding=0
        )

    def __call__(self, x):
        original_x = x
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        x = self.conv3(x)
        return x + original_x


class Decoder(nn.Module):
    """Upscales quantized vectors to mask in MLX."""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels=512, out_channels=128, kernel_size=1, padding=0
        )

        self.res_blocks = [ResBlock(features=128), ResBlock(features=128)]

        self.upsample1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1
        )
        self.upsample2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.upsample3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.upsample4 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1
        )

        self.conv_out = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, padding=0
        )

    def __call__(self, x):
        x = self.conv_in(x)
        x = nn.relu(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.upsample1(x)
        x = nn.relu(x)
        x = self.upsample2(x)
        x = nn.relu(x)
        x = self.upsample3(x)
        x = nn.relu(x)
        x = self.upsample4(x)
        x = nn.relu(x)

        x = self.conv_out(x)

        return x


def load_model_weights(decoder, params):
    """Load the pretrained weights into the decoder model."""
    decoder.conv_in.weight = params["conv_in"]["weight"]
    decoder.conv_in.bias = params["conv_in"]["bias"]

    for i, res_block in enumerate(decoder.res_blocks):
        res_block_params = params["res_blocks"][i]

        res_block.conv1.weight = res_block_params["conv1"]["weight"]
        res_block.conv1.bias = res_block_params["conv1"]["bias"]

        res_block.conv2.weight = res_block_params["conv2"]["weight"]
        res_block.conv2.bias = res_block_params["conv2"]["bias"]

        res_block.conv3.weight = res_block_params["conv3"]["weight"]
        res_block.conv3.bias = res_block_params["conv3"]["bias"]

    upsample_layers = [
        decoder.upsample1,
        decoder.upsample2,
        decoder.upsample3,
        decoder.upsample4,
    ]

    for i, upsample in enumerate(upsample_layers):
        upsample.weight = params["upsample_layers"][i]["weight"]
        upsample.bias = params["upsample_layers"][i]["bias"]

    decoder.conv_out.weight = params["conv_out"]["weight"]
    decoder.conv_out.bias = params["conv_out"]["bias"]

    return decoder


def _get_params(checkpoint):
    """Converts PyTorch checkpoint to MLX params."""

    # For Conv2d: PyTorch uses (out_channels, in_channels, kernel_h, kernel_w)
    # MLX expects (out_channels, kernel_h, kernel_w, in_channels)
    def transp(kernel):
        kernel = mx.array(kernel)
        return mx.transpose(kernel, (0, 2, 3, 1))

    # For ConvTranspose2d, the input/output channels are swapped in PyTorch
    def transp_transpose(kernel):
        kernel = mx.array(kernel)
        kernel = mx.transpose(kernel, (1, 0, 2, 3))  # Swap in/out channels
        return mx.transpose(kernel, (0, 2, 3, 1))  # Convert to MLX format

    def conv(name):
        return {
            "bias": checkpoint[name + ".bias"],
            "weight": transp(checkpoint[name + ".weight"]),
        }

    def conv_transpose(name):
        return {
            "bias": mx.array(checkpoint[name + ".bias"]),
            "weight": mx.array(transp_transpose(checkpoint[name + ".weight"])),
        }

    def resblock(name):
        return {
            "conv1": conv(name + ".0"),
            "conv2": conv(name + ".2"),
            "conv3": conv(name + ".4"),
        }

    params = {
        "_embeddings": mx.array(checkpoint["_vq_vae._embedding"]),
        "conv_in": conv("decoder.0"),
        "res_blocks": [
            resblock("decoder.2.net"),
            resblock("decoder.3.net"),
        ],
        "upsample_layers": [
            conv_transpose("decoder.4"),
            conv_transpose("decoder.6"),
            conv_transpose("decoder.8"),
            conv_transpose("decoder.10"),
        ],
        "conv_out": conv("decoder.12"),
    }

    return params


def _quantized_values_from_codebook_indices(codebook_indices, embeddings):
    batch_size, num_tokens = codebook_indices.shape
    assert num_tokens == 16, codebook_indices.shape
    unused_num_embeddings, embedding_dim = embeddings.shape

    encodings = mx.take(embeddings, codebook_indices.reshape((-1,)), axis=0)
    encodings = encodings.reshape((batch_size, 4, 4, embedding_dim))
    return encodings


@functools.cache
def get_reconstruct_masks(model):
    """Reconstructs masks from codebook indices using MLX."""

    def reconstruct_masks(codebook_indices):
        quantized = _quantized_values_from_codebook_indices(
            codebook_indices, params["_embeddings"]
        )

        decoder = Decoder()
        decoder = load_model_weights(decoder, params)

        return decoder(quantized)

    with gfile.GFile(_KNOWN_MODELS.get(model, model), "rb") as f:
        checkpoint_data = dict(np.load(f))
        params = _get_params(checkpoint_data)

    return reconstruct_masks


def extract_and_create_array(pattern: str):
    seg_tokens = re.findall(r"<seg(\d{3})>", pattern)
    seg_numbers = [int(match) for match in seg_tokens]
    return mx.array(seg_numbers)


if __name__ == "__main__":
    model, processor = load(MODEL_PATH)
    config = model.config

    image = load_image(IMAGE_PATH)
    log.info(f"Image size: {image.size}")

    prompt = "segment android\n"
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
    output = generate(model, processor, formatted_prompt, image, verbose=False)
    log.info(output)

    reconstruct_fn = get_reconstruct_masks("oi")

    parts = output.split(" ")
    segs = list(filter(lambda x: "seg" in x, parts))
    codes = extract_and_create_array(segs[0])[None]
    log.info(f"Codes shape: {codes.shape}")

    masks = reconstruct_fn(codes)
    mask = masks[0]
    log.info(f"Masks shape: {masks.shape}")
    plt.imshow(mask)
    plt.show()
