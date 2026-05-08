import torch
import torch.nn as nn

from transformers import AutoModel


class SentimentClassifier(nn.Module):

    def __init__(
        self,
        model_name="vinai/phobert-base",
        num_labels=3,
        dropout=0.3
    ):

        super().__init__()

        # PhoBERT
        self.phobert = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = (
            self.phobert.config.hidden_size
        )

        # Dropout
        self.dropout = nn.Dropout(
            dropout
        )

        # Classifier
        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):

        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS embedding
        cls_output = outputs.last_hidden_state[
            :,
            0,
            :
        ]

        cls_output = self.dropout(
            cls_output
        )

        logits = self.classifier(
            cls_output
        )

        # Training
        if labels is not None:

            loss_fn = nn.CrossEntropyLoss()

            loss = loss_fn(
                logits,
                labels
            )

            return loss, logits

        # Inference
        return logits
    
if __name__ == "__main__":

    model = SentimentClassifier()

    batch_size = 2
    seq_len = 32

    input_ids = torch.randint(
        0,
        1000,
        (batch_size, seq_len)
    )

    attention_mask = torch.ones(
        batch_size,
        seq_len
    ).long()

    labels = torch.randint(
        0,
        3,
        (batch_size,)
    )

    loss, logits = model(
        input_ids,
        attention_mask,
        labels
    )

    print("Loss:", loss)

    print("Logits shape:", logits.shape)