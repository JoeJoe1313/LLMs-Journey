"""Part of the code below is adapted from 
https://github.com/google-research/big_vision/blob/main/big_vision/evaluators/proj/paligemma/transfers/segmentation.py.
This version uses MLX instead of JAX/Flax.
"""

import functools
import logging
import re
from typing import Any, Callable, Tuple

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

    def __init__(self, features: int):
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

    def __call__(self, x: Any) -> Any:
        original_x = x
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = self.conv3(x)
        return x + original_x


class Decoder(nn.Module):
    """
    Decoder that upscales quantized vectors to produce a mask.
    The architecture is parameterized to avoid hardcoded layer definitions.
    """

    def __init__(
        self,
        in_channels: int = 512,
        res_channels: int = 128,
        out_channels: int = 1,
        num_res_blocks: int = 2,
        upsample_channels: Tuple[int, ...] = (128, 64, 32, 16),
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels=in_channels, out_channels=res_channels, kernel_size=1, padding=0
        )
        self.res_blocks = [
            ResBlock(features=res_channels) for _ in range(num_res_blocks)
        ]
        self.upsample_layers = []
        prev_channels = res_channels
        for ch in upsample_channels:
            self.upsample_layers.append(
                nn.ConvTranspose2d(
                    in_channels=prev_channels,
                    out_channels=ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            prev_channels = ch
        self.conv_out = nn.Conv2d(
            in_channels=upsample_channels[-1],
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )

    def __call__(self, x: Any) -> Any:
        x = nn.relu(self.conv_in(x))
        for block in self.res_blocks:
            x = block(x)
        for layer in self.upsample_layers:
            x = nn.relu(layer(x))
        return self.conv_out(x)


def load_model_weights(decoder: Decoder, params: dict) -> Decoder:
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

    for i, upsample in enumerate(decoder.upsample_layers):
        upsample.weight = params["upsample_layers"][i]["weight"]
        upsample.bias = params["upsample_layers"][i]["bias"]

    decoder.conv_out.weight = params["conv_out"]["weight"]
    decoder.conv_out.bias = params["conv_out"]["bias"]

    return decoder


def _get_params(checkpoint: dict) -> dict:
    """Converts PyTorch checkpoint to MLX params."""

    def transp(kernel: Any) -> Any:
        kernel = mx.array(kernel)
        return mx.transpose(kernel, (0, 2, 3, 1))

    def transp_transpose(kernel: Any) -> Any:
        kernel = mx.array(kernel)
        kernel = mx.transpose(kernel, (1, 0, 2, 3))  # Swap in/out channels
        return mx.transpose(kernel, (0, 2, 3, 1))  # Convert to MLX format

    def conv(name: str) -> dict:
        return {
            "bias": checkpoint[f"{name}.bias"],
            "weight": transp(checkpoint[f"{name}.weight"]),
        }

    def conv_transpose(name: str) -> dict:
        return {
            "bias": mx.array(checkpoint[f"{name}.bias"]),
            "weight": mx.array(transp_transpose(checkpoint[f"{name}.weight"])),
        }

    def resblock(name: str) -> dict:
        return {
            "conv1": conv(f"{name}.0"),
            "conv2": conv(f"{name}.2"),
            "conv3": conv(f"{name}.4"),
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


def _quantized_values_from_codebook_indices(
    codebook_indices: mx.array, embeddings: mx.array
) -> mx.array:
    batch_size, num_tokens = codebook_indices.shape
    expected_tokens = 16
    assert (
        num_tokens == expected_tokens
    ), f"Expected {expected_tokens} tokens, got {codebook_indices.shape}"
    encodings = mx.take(embeddings, codebook_indices.reshape((-1,)), axis=0)

    return encodings.reshape((batch_size, 4, 4, embeddings.shape[1]))


@functools.cache
def get_reconstruct_masks(model: str) -> Callable[[mx.array], Any]:
    """Loads the checkpoint and returns a function that reconstructs masks
    from codebook indices using a preloaded MLX decoder.
    """
    checkpoint_path = _KNOWN_MODELS.get(model, model)
    with gfile.GFile(checkpoint_path, "rb") as f:
        checkpoint_data = dict(np.load(f))
    params = _get_params(checkpoint_data)

    decoder = Decoder()
    decoder = load_model_weights(decoder, params)

    def reconstruct_masks(codebook_indices: mx.array) -> Any:
        quantized = _quantized_values_from_codebook_indices(
            codebook_indices, params["_embeddings"]
        )
        return decoder(quantized)

    return reconstruct_masks


def extract_and_create_array(pattern: str) -> mx.array:
    """Extracts segmentation tokens from the pattern and returns them as an MLX array."""
    seg_tokens = re.findall(r"<seg(\d{3})>", pattern)
    seg_numbers = [int(token) for token in seg_tokens]
    return mx.array(seg_numbers)


def main() -> None:
    model, processor = load(MODEL_PATH)
    config = model.config

    image = load_image(IMAGE_PATH)
    log.info(f"Image size: {image.size}")

    prompt = "segment android\n"
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
    output = generate(model, processor, formatted_prompt, image, verbose=False)
    log.info(f"Output: {output}")

    reconstruct_fn = get_reconstruct_masks("oi")

    # Extract segmentation tokens using a list comprehension.
    segs = [part for part in output.split(" ") if "seg" in part]
    if not segs:
        log.error("No segmentation tokens found in output.")
        return

    codes = extract_and_create_array(segs[0])[None]
    log.info(f"Codes shape: {codes.shape}")

    masks = reconstruct_fn(codes)
    log.info(f"Masks shape: {masks.shape}")

    plt.imshow(masks[0])
    plt.show()


if __name__ == "__main__":
    main()
