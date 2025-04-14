"""Part of the code below is adapted from
https://github.com/google-research/big_vision/blob/main/big_vision/evaluators/proj/paligemma/transfers/segmentation.py.
This version uses MLX instead of JAX/Flax.
"""

import argparse
import functools
import logging
import re
from typing import Callable, List, Tuple

import cv2
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

MODEL_PATH = "mlx-community/paligemma2-10b-mix-448-8bit"
IMAGE_PATH = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
_KNOWN_MODELS = {"oi": "gs://big_vision/paligemma/vae-oid.npz"}


class ResBlock(nn.Module):
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

    def __call__(self, x: mx.array) -> mx.array:
        original_x = x
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = self.conv3(x)
        return x + original_x


class Decoder(nn.Module):
    """
    Decoder that upscales quantized vectors to produce a mask.
    The architecture is parameterized to avoid hardcoded layer definitions.
    Takes channels-last input data (B, H, W, C).
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
        out_up_ch = res_channels
        for ch in upsample_channels:
            self.upsample_layers.append(
                nn.ConvTranspose2d(
                    in_channels=out_up_ch,
                    out_channels=ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            out_up_ch = ch
        self.conv_out = nn.Conv2d(
            in_channels=upsample_channels[-1],
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.conv_in(x))
        for block in self.res_blocks:
            x = block(x)
        for layer in self.upsample_layers:
            x = nn.relu(layer(x))

        return self.conv_out(x)


def _get_params(checkpoint: dict) -> dict:
    """Converts PyTorch checkpoint to MLX params (nested dict).
    Uses transpositions yielding (Out, H, W, In) format weights."""

    def transp(kernel: np.ndarray) -> mx.array:
        return mx.transpose(mx.array(kernel), (0, 2, 3, 1))

    def transp_transpose(kernel: np.ndarray) -> mx.array:
        intermediate = mx.transpose(mx.array(kernel), (1, 0, 2, 3))

        return mx.transpose(intermediate, (0, 2, 3, 1))

    def conv(name: str) -> dict:
        return {
            "bias": mx.array(checkpoint[f"{name}.bias"]),
            "weight": transp(checkpoint[f"{name}.weight"]),
        }

    def conv_transpose(name: str) -> dict:
        return {
            "bias": mx.array(checkpoint[f"{name}.bias"]),
            "weight": transp_transpose(checkpoint[f"{name}.weight"]),
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
    if num_tokens != expected_tokens:
        log.error(f"Expected {expected_tokens} tokens, got {codebook_indices.shape}")

    encodings = mx.take(embeddings, codebook_indices.reshape((-1,)), axis=0)

    return encodings.reshape((batch_size, 4, 4, embeddings.shape[1]))


@functools.cache
def get_reconstruct_masks(model: str) -> Callable[[mx.array], mx.array]:
    """Loads the checkpoint and returns a function that reconstructs masks
    from codebook indices using a preloaded MLX decoder.
    """
    checkpoint_path = _KNOWN_MODELS.get(model, model)
    with gfile.GFile(checkpoint_path, "rb") as f:
        checkpoint_data = dict(np.load(f))

    params = _get_params(checkpoint_data)
    embeddings = params.pop("_embeddings")
    log.info(f"VAE embedding dimension: {embeddings.shape[1]}")

    decoder = Decoder()
    decoder.update(params)

    def reconstruct_masks(codebook_indices: mx.array) -> mx.array:
        quantized = _quantized_values_from_codebook_indices(
            codebook_indices, embeddings
        )
        return decoder(quantized)

    return reconstruct_masks


def extract_and_create_arrays(pattern: str) -> List[mx.array]:
    """Extracts segmentation tokens from each object in the pattern and returns a list of MLX arrays."""
    object_strings = [obj.strip() for obj in pattern.split(";") if obj.strip()]

    seg_tokens_arrays = []
    for obj in object_strings:
        seg_tokens = re.findall(r"<seg(\d{3})>", obj)
        if seg_tokens:
            seg_numbers = [int(token) for token in seg_tokens]
            seg_tokens_arrays.append(mx.array(seg_numbers))

    return seg_tokens_arrays


def parse_bbox(model_output: str):
    entries = model_output.split(";")

    results = []
    for entry in entries:
        entry = entry.strip()
        numbers = re.findall(r"<loc(\d+)>", entry)
        if len(numbers) == 4:
            bbox = [int(num) for num in numbers]
            results.append(bbox)

    return results


def gather_masks(output, codes_list, reconstruct_fn):
    masks_list = []

    target_width, target_height = 448, 448
    for i, codes in enumerate(codes_list):
        codes_batch = codes[None, :]
        masks = reconstruct_fn(codes_batch)
        mask_np = np.array(masks[0, :, :, 0], copy=False)

        y_min, x_min, y_max, x_max = parse_bbox(output)[i]
        x_min_norm = int(x_min / 1024 * target_width)
        y_min_norm = int(y_min / 1024 * target_height)
        x_max_norm = int(x_max / 1024 * target_width)
        y_max_norm = int(y_max / 1024 * target_height)

        masks_list.append(
            {
                "mask": mask_np,
                "coordinates": (x_min_norm, y_min_norm, x_max_norm, y_max_norm),
            }
        )

    return masks_list


def plot_masks(args, processor, masks_list):

    image = load_image(args.image_path)
    img_array = processor.image_processor(image)["pixel_values"][0].transpose(1, 2, 0)
    img_array = (img_array * 0.5 + 0.5).clip(0, 1)

    full = np.ones((448, 448, 1)) * (-1)
    for mask_info in masks_list:
        mask_np = mask_info["mask"]
        x_min_norm, y_min_norm, x_max_norm, y_max_norm = mask_info["coordinates"]

        width = x_max_norm - x_min_norm
        height = y_max_norm - y_min_norm

        resized_mask = cv2.resize(
            mask_np, (width, height), interpolation=cv2.INTER_NEAREST
        )
        resized_mask = resized_mask.reshape((height, width, 1))

        full[y_min_norm:y_max_norm, x_min_norm:x_max_norm] = resized_mask

    n_masks = len(masks_list)
    _, axs = plt.subplots(1, n_masks + 1, figsize=(5 * (n_masks + 1), 6))

    axs[0].imshow(img_array)
    axs[0].imshow(full, alpha=0.5)
    axs[0].set_title("Mask Overlay")
    axs[0].axis("on")

    for i, mask_info in enumerate(masks_list, start=1):
        mask_np = mask_info["mask"]
        axs[i].imshow(mask_np)
        axs[i].set_title(f"Reconstructed Mask {i}")
        axs[i].axis("on")

    plt.tight_layout()
    plt.show()


def main(args) -> None:
    log.info(f"Loading PaliGemma model: {args.model_path}")
    model, processor = load(args.model_path)
    config = model.config

    image = load_image(args.image_path)
    log.info(f"Image size: {image.size}")

    vae_path = _KNOWN_MODELS.get(args.vae_checkpoint_path, args.vae_checkpoint_path)
    reconstruct_fn = get_reconstruct_masks(vae_path)

    prompt = args.prompt.strip() + "\n"
    log.info(f"Using prompt: '{prompt.strip()}'")
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)

    log.info("Generating segmentation output...")
    output = generate(model, processor, formatted_prompt, image, verbose=False)
    log.info(f"Model output: {output}")

    codes_list = extract_and_create_arrays(output)
    log.info(f"Extracted codes: {codes_list}")

    log.info("Reconstructing mask from codes...")
    masks_list = gather_masks(output, codes_list, reconstruct_fn)

    log.info("Plotting masks...")
    plot_masks(args, processor, masks_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision tasks using PaliGemma 2 mix.")
    parser.add_argument(
        "--image_path", type=str, default=IMAGE_PATH, help="Path to the input image."
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Prompt for the model."
    )
    parser.add_argument(
        "--model_path", type=str, default=MODEL_PATH, help="Path to the mlx model."
    )
    parser.add_argument(
        "--vae_checkpoint_path", type=str, default="oi", help="Path to the .npz file."
    )

    cli_args = parser.parse_args()
    main(cli_args)
