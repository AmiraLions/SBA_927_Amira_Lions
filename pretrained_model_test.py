
# Requirement 7: Pre-trained Model Experiment

from pathlib import Path
from transformers import pipeline

# Load the first entry from our existing dataset.
file_path = Path(__file__).parent / "SBA927.txt"
dataset = file_path.read_text(encoding="utf-8").splitlines()

# Load a pre-trained sentiment analysis model.
classifier = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,
)


# Analyze all 12 dataset entries.
print("\nRequirement 7 - DistilBERT Sentiment Analysis")

for i, text in enumerate(dataset, start=1):
    result = classifier(text)[0]

    sentiment = result["label"]
    confidence = result["score"]

    print(
        f"Text {i}: {sentiment} "
        f"(confidence: {confidence:.4f})"
    )
