"""Custom MLflow pyfunc wrapper around a Hugging Face sentiment pipeline (as in Lab 9).

The Hugging Face model ID is stored in the logged model's model_config, so a
model loaded from MLflow knows which checkpoint it is without any outside code.
"""
import mlflow
import pandas as pd


def normalise_label(label: str) -> str:
    """Map every label style (POSITIVE, LABEL_1, positive) to 'positive'/'negative'."""
    l = str(label).strip().lower()
    if l in ("positive", "pos", "label_1", "1"):
        return "positive"
    if l in ("negative", "neg", "label_0", "0"):
        return "negative"
    return l


class HFSentimentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import pipeline

        cfg = context.model_config or {}
        self.hf_model_id = cfg["hf_model_id"]
        self.max_length = int(cfg.get("max_length", 256))
        self.batch_size = int(cfg.get("batch_size", 16))
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.hf_model_id,
            truncation=True,
            max_length=self.max_length,
            device=-1,
        )

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]
        outs = self.pipe(texts, batch_size=self.batch_size)
        return pd.DataFrame(
            {
                "text": texts,
                "prediction": [normalise_label(o["label"]) for o in outs],
                "score": [round(float(o["score"]), 4) for o in outs],
            }
        )
