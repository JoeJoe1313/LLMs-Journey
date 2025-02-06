"""Taken from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/tuner/datasets.py,
and slighlty modified. At the time of writing this these functions are still not part of the
released stable versions of the package.
"""

from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer


class Dataset:
    """
    Light-weight wrapper to hold a dataset.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        text_key: str = "text",
    ):
        self._data = [tokenizer.encode(d[text_key]) for d in data]
        for d in self._data:
            if d[-1] != tokenizer.eos_token_id:
                d.append(tokenizer.eos_token_id)

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class ChatDataset:
    """
    A dataset for chat data in the format of {"messages": [...]}
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(self, data: List[Dict[str, str]], tokenizer: PreTrainedTokenizer):
        self._data = [
            tokenizer.apply_chat_template(
                d["messages"],
                tools=d.get("tools", None),
            )
            for d in data
        ]

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class CompletionsDataset:
    """
    A dataset for prompt-completion data in the format of {"prompt": ..., "completion": ...}
    or using user-provided keys for prompt and completion values
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str,
        completion_key: str,
    ):
        self._data = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": d[prompt_key]},
                    {"role": "assistant", "content": d[completion_key]},
                ],
            )
            for d in data
        ]

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def create_dataset(
    data,
    tokenizer: PreTrainedTokenizer,
    prompt_feature: Optional[str] = None,
    completion_feature: Optional[str] = None,
):
    prompt_feature = prompt_feature or "prompt"
    completion_feature = completion_feature or "completion"
    sample = data[0]
    if "messages" in sample:
        return ChatDataset(data, tokenizer)
    elif prompt_feature in sample and completion_feature in sample:
        return CompletionsDataset(data, tokenizer, prompt_feature, completion_feature)
    elif "text" in sample:
        return Dataset(data, tokenizer)
    else:
        raise ValueError(
            "Unsupported data format, check the supported formats here:\n"
            "https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#data."
        )


def load_hf_dataset(
    data_id: str,
    tokenizer: PreTrainedTokenizer,
    prompt_feature: Optional[str] = None,
    completion_feature: Optional[str] = None,
):
    from datasets import exceptions, load_dataset

    try:
        dataset = load_dataset(data_id)

        names = ("train", "validation", "test")

        train, valid, test = [
            (
                create_dataset(
                    dataset[n], tokenizer, prompt_feature, completion_feature
                )
                if n in dataset.keys()
                else []
            )
            for n in names
        ]

    except exceptions.DatasetNotFoundError:
        raise ValueError(f"Not found Hugging Face dataset: {data_id} .")

    return train, valid, test
