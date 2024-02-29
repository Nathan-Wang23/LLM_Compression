from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sentencepiece import SentencePieceProcessor
import os
from logging import getLogger
from typing import List
from transformers import LlamaTokenizer

logger = getLogger()


class Tokenizer:
    """tokenizing and encoding/decoding text using SentencePiece."""

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a SentencePiece model.

        Args:
            model_path (str): The path to the SentencePiece model file.
        """
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.sp_model.decode(t)


tokenizer = LlamaTokenizer("tokenizer.model")


# TESTING FOR Eval dataset loading. Not an official file.
class EvaluationDataset(Dataset):
    def __init__(self, data, max_seq_len):
        self.input = data
        self.max_seq_len = max_seq_len
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for val in self.input:
            text = val["text"]
            text_split = 0.3
            words = tokenizer(text)["input_ids"]
            words = words[0 : self.max_seq_len]
            len_text = len(words)
            beginning = tokenizer.decode(words[: int(text_split * len_text)])
            ref = tokenizer.decode(words[int(text_split * len_text) :])
            comp = {"text": beginning, "reference": ref}
            data.append(comp)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


start = 1_500_000
test_split = 100
data = load_dataset("DKYoon/SlimPajama-6B", split="train")
data_split = data.select(range(start, start + test_split))

evaluation_dataset = EvaluationDataset(data_split, 512)

print("TEXT:\n")
print(evaluation_dataset[3]["text"])
print("\nREFERENCE:\n")
print(evaluation_dataset[3]["reference"])

batch_size = 32

dataloader = DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=False)
