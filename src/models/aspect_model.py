import torch
import torch.nn as nn

from transformers import AutoModel

from TorchCRF import CRF


class AspectExtractor(nn.Module):

    def __init__(
        self,
        model_name="vinai/phobert-base",
        hidden_dim=256,
        num_labels=3,
        dropout=0.3
    ):

        super().__init__()

        # PhoBERT
        self.phobert = AutoModel.from_pretrained(
            model_name
        )

        phobert_hidden_size = (
            self.phobert.config.hidden_size
        )

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=phobert_hidden_size,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classifier
        self.classifier = nn.Linear(
            hidden_dim * 2,
            num_labels
        )

        # CRF
        self.crf = CRF(
            num_labels,
            batch_first=True
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):

        # PhoBERT embeddings
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state

        # BiLSTM
        lstm_output, _ = self.bilstm(
            sequence_output
        )

        lstm_output = self.dropout(
            lstm_output
        )

        # Emission scores
        emissions = self.classifier(
            lstm_output
        )

        # Training
        if labels is not None:

            # Replace -100 labels with 0 for masked positions
            # CRF will use the mask to ignore these positions
            active_labels = labels.clone()
            active_labels[labels == -100] = 0

            loss = -self.crf(
                emissions,
                active_labels,
                mask=attention_mask.bool(),
                reduction="mean"
            )

            return loss

        # Inference
        predictions = self.crf.decode(
            emissions,
            mask=attention_mask.bool()
        )

        return predictions

if __name__ == "__main__":

    model = AspectExtractor()

    batch_size = 2
    seq_len = 16

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
        (batch_size, seq_len)
    )

    loss = model(
        input_ids,
        attention_mask,
        labels
    )

    print("Loss:", loss)