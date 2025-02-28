# Copyright © 2023-2024 Apple Inc.

import argparse
from functools import partial

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load, stream_generate

DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MODEL = "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit"


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="PRNG seed")
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load(
        args.model,
        adapter_path=args.adapter_path,
        tokenizer_config={"trust_remote_code": True},
    )

    wait_token = "Wait"
    wait_token_id = tokenizer.convert_tokens_to_ids(wait_token)
    end_think_token = "</think>"
    end_think_token_id = tokenizer.convert_tokens_to_ids(end_think_token)
    think_more_prompt = mx.array([wait_token_id], mx.uint32)
    end_think_prompt = mx.array(
        tokenizer.encode(end_think_token + "\n", add_special_tokens=False), mx.uint32
    )
    generator = partial(
        stream_generate,
        model=model,
        tokenizer=tokenizer,
        sampler=make_sampler(args.temp, args.top_p),
    )
    print(f"[INFO] Starting reasoning session with {args.model}. To exit, enter 'q'.")
    while True:
        prompt_cache = make_prompt_cache(model)
        query = input(">> ")
        if query == "q":
            break
        messages = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        while True:
            max_tokens = args.max_tokens
            end_think_idx = None
            for response in generator(
                prompt=prompt,
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
            ):
                if response.token == wait_token_id:
                    break
                elif response.token == end_think_token_id:
                    end_think_idx = prompt_cache[0].offset

                print(response.text, flush=True, end="")

            max_tokens -= response.generation_tokens

            # If we got a wait token insert </think> and generate the response
            if end_think_idx is None:
                print(end_think_token, flush=True)
                end_think_idx = prompt_cache[0].offset
                prompt = end_think_prompt

                # Trim the wait token from the cache
                trim_prompt_cache(prompt_cache, 1)

                # Generate answer
                for response in generator(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    prompt_cache=prompt_cache,
                ):
                    print(response.text, flush=True, end="")

                max_tokens -= response.generation_tokens
            think_more = input(
                "\n\n\033[31mWould you like me to think more? (y/n):\033[0m "
            )
            if think_more == "y":
                # Trim the prompt cache to just before the end of think token
                print("<think>")
                print(wait_token, flush=True, end="")
                num_to_trim = prompt_cache[0].offset - end_think_idx + 1
                max_tokens += num_to_trim
                trim_prompt_cache(prompt_cache, num_to_trim)
                prompt = think_more_prompt
            else:
                break

        print()


if __name__ == "__main__":
    main()
