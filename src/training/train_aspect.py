import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataset import ABSADataset
from models.aspect_model import AspectExtractor

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_LEN = 128

# Resolve paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "aspect_extraction", "train.txt")
DEV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "aspect_extraction", "dev.txt")

SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "aspect_model")


def train_epoch(
    model,
    dataloader,
    optimizer
):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader)
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        loss = model(
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


def evaluate(
    model,
    dataloader
):
    model.eval()
    total_loss = 0
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
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    os.makedirs(
        SAVE_DIR,
        exist_ok=True
    )

    # Dataset
    train_dataset = ABSADataset(
        TRAIN_PATH,
        max_len=MAX_LEN
    )

    dev_dataset = ABSADataset(
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
    model = AspectExtractor()

    model.to(DEVICE)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_dev_loss = float("inf")

    # Training loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer
        )
        dev_loss = evaluate(
            model,
            dev_loader
        )
        print(
            f"Train Loss: {train_loss:.4f}"
        )
        print(
            f"Dev Loss: {dev_loss:.4f}"
        )

        # Save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
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