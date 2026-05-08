import os
import sys
import torch
from transformers import AutoTokenizer
sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.aspect_model import AspectExtractor


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

ID2LABEL = {
    0: "O",
    1: "B-ASP",
    2: "I-ASP"
}

class ABSAPredictor:
    def __init__(
        self,
        checkpoint_path=None
    ):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_checkpoint = os.path.join(
            script_dir,
            "..",
            "..",
            "checkpoints",
            "aspect_model",
            "best_model.pt"
        )

        checkpoint_path = checkpoint_path or default_checkpoint
        checkpoint_path = os.path.abspath(checkpoint_path)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Please train the model or provide a valid checkpoint path."
            )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "vinai/phobert-base"
        )

        # Model
        self.model = AspectExtractor()

        self.model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=DEVICE
            )
        )

        self.model.to(DEVICE)

        self.model.eval()

    def decode_aspects(
        self,
        tokens,
        labels
    ):

        aspects = []
        current_aspect = []
        for token, label_id in zip(tokens, labels):
            label = ID2LABEL[label_id]
            if label == "B-ASP":
                if len(current_aspect) > 0:
                    aspects.append(
                        " ".join(current_aspect)
                    )
                current_aspect = [token]
            elif label == "I-ASP":
                current_aspect.append(token)
            else:
                if len(current_aspect) > 0:
                    aspects.append(
                        " ".join(current_aspect)
                    )
                    current_aspect = []
        if len(current_aspect) > 0:
            aspects.append(
                " ".join(current_aspect)
            )

        return aspects

    def predict(self, text, max_len=128):

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_len
        )

        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            predictions = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        pred_labels = predictions[0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        aspects = []
        current = []

        for token, label_id in zip(tokens, pred_labels):

            label = ID2LABEL[label_id]

            # bỏ special tokens
            if token in ["<s>", "</s>", "<pad>"]:
                continue

            if label == "B-ASP":
                if current:
                    aspects.append(" ".join(current))
                current = [token]

            elif label == "I-ASP":
                current.append(token)

            else:
                if current:
                    aspects.append(" ".join(current))
                    current = []

        if current:
            aspects.append(" ".join(current))

        return aspects
    
# Test
if __name__ == "__main__":
    predictor = ABSAPredictor()
    text = "Đồ ăn ngon nhưng phục vụ chậm"
    aspects = predictor.predict(text)
    print("Aspects:", aspects)