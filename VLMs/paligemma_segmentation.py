"""Part of the code below is taken from https://github.com/google-research/big_vision/blob/main/big_vision/evaluators/proj/paligemma/transfers/segmentation.py.

pip install jax==0.4.26 jaxlib==0.4.26
pip install jax-metal
pip install flax
"""

import functools
import logging
import re

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
    features: int

    @nn.compact
    def __call__(self, x):
        original_x = x
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(1, 1), padding=0)(x)
        return x + original_x


class Decoder(nn.Module):
    """Upscales quantized vectors to mask."""

    @nn.compact
    def __call__(self, x):
        num_res_blocks = 2
        dim = 128
        num_upsample_layers = 4

        x = nn.Conv(features=dim, kernel_size=(1, 1), padding=0)(x)
        x = nn.relu(x)

        for _ in range(num_res_blocks):
            x = ResBlock(features=dim)(x)

        for _ in range(num_upsample_layers):
            x = nn.ConvTranspose(
                features=dim,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding=2,
                transpose_kernel=True,
            )(x)
            x = nn.relu(x)
            dim //= 2

        x = nn.Conv(features=1, kernel_size=(1, 1), padding=0)(x)

        return x


def _get_params(checkpoint):
    """Converts PyTorch checkpoint to Flax params."""

    def transp(kernel):
        return np.transpose(kernel, (2, 3, 1, 0))

    def conv(name):
        return {
            "bias": checkpoint[name + ".bias"],
            "kernel": transp(checkpoint[name + ".weight"]),
        }

    def resblock(name):
        return {
            "Conv_0": conv(name + ".0"),
            "Conv_1": conv(name + ".2"),
            "Conv_2": conv(name + ".4"),
        }

    return {
        "_embeddings": checkpoint["_vq_vae._embedding"],
        "Conv_0": conv("decoder.0"),
        "ResBlock_0": resblock("decoder.2.net"),
        "ResBlock_1": resblock("decoder.3.net"),
        "ConvTranspose_0": conv("decoder.4"),
        "ConvTranspose_1": conv("decoder.6"),
        "ConvTranspose_2": conv("decoder.8"),
        "ConvTranspose_3": conv("decoder.10"),
        "Conv_1": conv("decoder.12"),
    }


def _quantized_values_from_codebook_indices(codebook_indices, embeddings):
    batch_size, num_tokens = codebook_indices.shape
    assert num_tokens == 16, codebook_indices.shape
    unused_num_embeddings, embedding_dim = embeddings.shape

    encodings = jnp.take(embeddings, codebook_indices.reshape((-1)), axis=0)
    encodings = encodings.reshape((batch_size, 4, 4, embedding_dim))
    return encodings


@functools.cache
def get_reconstruct_masks(model):
    """Reconstructs masks from codebook indices.

    Based on code from https://arxiv.org/abs/2301.02229

    Verified in
    https://colab.research.google.com/drive/1AOr0cokOpM6-N9Z5HmxoeGxGj6jS37Vl

    Args:
        model: Model to use for conversion.

    Returns:
        A function that expects indices shaped `[B, 16]` of dtype int32, each
        ranging from 0 to 127 (inclusive), and that returns a decoded masks sized
        `[B, 64, 64, 1]`, of dtype float32, in range [-1, 1].
    """

    def reconstruct_masks(codebook_indices):
        quantized = _quantized_values_from_codebook_indices(
            codebook_indices, params["_embeddings"]
        )
        return Decoder().apply({"params": params}, quantized)

    with gfile.GFile(_KNOWN_MODELS.get(model, model), "rb") as f:
        params = _get_params(dict(np.load(f)))

    return jax.jit(reconstruct_masks, backend="cpu")


def extract_and_create_array(pattern: str):
    seg_tokens = re.findall(r"<seg(\d{3})>", pattern)
    seg_numbers = [int(match) for match in seg_tokens]

    return np.array(seg_numbers)


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
    plt.imshow(masks[0])
    plt.show()
