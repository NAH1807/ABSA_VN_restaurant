import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

LABEL2ID = {
    "O": 0,
    "B-ASP": 1,
    "I-ASP": 2
}
ID2LABEL = {
    0: "O",
    1: "B-ASP",
    2: "I-ASP"
}


class ABSADataset(Dataset):

    def __init__(
        self,
        file_path,
        max_len=128
    ):

        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(
            "vinai/phobert-base"
        )

        self.samples = self.load_data(file_path)

    def load_data(self, file_path):

        samples = []

        tokens = []
        labels = []

        with open(file_path, "r", encoding="utf-8") as f:

            lines = f.readlines()

            for line in lines:

                line = line.strip()

                if line == "":

                    if len(tokens) > 0:

                        samples.append(
                            (tokens, labels)
                        )

                    tokens = []
                    labels = []

                    continue

                token, label = line.split("\t")

                tokens.append(token)

                labels.append(label)

        return samples

    def align_labels_manually(
        self,
        tokens,
        labels,
        input_ids
    ):

        aligned_labels = []

        token_idx = 0

        for input_id in input_ids:

            if input_id == self.tokenizer.cls_token_id:
                aligned_labels.append(-100)
                continue

            if input_id == self.tokenizer.sep_token_id:
                aligned_labels.append(-100)
                break

            if input_id == self.tokenizer.pad_token_id:
                aligned_labels.append(-100)
                continue

            if token_idx < len(labels):
                aligned_labels.append(
                    LABEL2ID[labels[token_idx]]
                )
                token_idx += 1

            else:
                aligned_labels.append(-100)

        while len(aligned_labels) < self.max_len:
            aligned_labels.append(-100)

        return aligned_labels[:self.max_len]

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        tokens, labels = self.samples[idx]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()

        if len(input_ids.shape) == 0:
            input_ids_list = [input_ids.item()]
        else:
            input_ids_list = input_ids.tolist()

        aligned_labels = self.align_labels_manually(
            tokens,
            labels,
            input_ids_list
        )

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(aligned_labels)
        }