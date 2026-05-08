import os
import sys
import torch
import pandas as pd

from tqdm import tqdm

from torch.utils.data import (
    Dataset,
    DataLoader
)

from transformers import AutoTokenizer
from torch.optim import AdamW

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sentiment_model import (
    SentimentClassifier
)


# =========================
# DEVICE
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================
# CONFIG
# =========================

BATCH_SIZE = 8
EPOCHS = 3
MAX_LEN = 128
LEARNING_RATE = 2e-5


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "sentiment_classification",
    "train.csv"
)

DEV_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "sentiment_classification",
    "dev.csv"
)


SAVE_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "sentiment_model"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# =========================
# LABELS
# =========================

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

ID2LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# =========================
# DATASET
# =========================

class SentimentDataset(Dataset):

    def __init__(
        self,
        file_path,
        max_len=128
    ):

        self.df = pd.read_csv(
            file_path
        )

        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(
            "vinai/phobert-base"
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        sentence = str(
            row["sentence"]
        )

        aspect = str(
            row["aspect"]
        )

        sentiment = str(
            row["sentiment"]
        ).strip().lower()

        # aspect-aware input
        text = (
            f"{aspect} [SEP] {sentence}"
        )

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {

            "input_ids":
                encoding["input_ids"].squeeze(),

            "attention_mask":
                encoding[
                    "attention_mask"
                ].squeeze(),

            "labels":
                torch.tensor(
                    LABEL2ID[sentiment]
                )
        }


# =========================
# TRAIN
# =========================

def train_epoch(
    model,
    dataloader,
    optimizer
):

    model.train()

    total_loss = 0

    progress_bar = tqdm(dataloader)

    for batch in progress_bar:

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch[
            "labels"
        ].to(DEVICE)

        optimizer.zero_grad()

        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_description(
            f"Loss: {loss.item():.4f}"
        )

    return total_loss / len(dataloader)


# =========================
# EVALUATE
# =========================

def evaluate(
    model,
    dataloader
):

    model.eval()

    total_loss = 0

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch[
                "labels"
            ].to(DEVICE)

            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    return (
        total_loss / len(dataloader),
        accuracy
    )


# =========================
# MAIN
# =========================

def main():

    # Dataset
    train_dataset = SentimentDataset(
        TRAIN_PATH,
        max_len=MAX_LEN
    )

    dev_dataset = SentimentDataset(
        DEV_PATH,
        max_len=MAX_LEN
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE
    )

    # Model
    model = SentimentClassifier()

    model.to(DEVICE)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_acc = 0

    # Training loop
    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch+1}/{EPOCHS}"
        )

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer
        )

        dev_loss, dev_acc = evaluate(
            model,
            dev_loader
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Dev Loss: {dev_loss:.4f}"
        )

        print(
            f"Dev Accuracy: {dev_acc:.4f}"
        )

        # Save best model
        if dev_acc > best_acc:

            best_acc = dev_acc

            save_path = os.path.join(
                SAVE_DIR,
                "best_model.pt"
            )

            torch.save(
                model.state_dict(),
                save_path
            )

            print(
                f"Saved best model to {save_path}"
            )


if __name__ == "__main__":
    main()