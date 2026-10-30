
# SBA 927 — Requirement 7: Pre-trained Language Model Experiment

## Objective
Experiment with a pre-trained language model to analyze
sentiment in the 12 entries from SBA927.txt.

## Model and Environment
- Model: DistilBERT, fine-tuned on SST-2 English sentiment data
- Model identifier:
  distilbert/distilbert-base-uncased-finetuned-sst-2-english
- Library: Hugging Face Transformers 5.22.1
- Backend: PyTorch 2.15.0+cpu
- Python: 3.14.7
- Execution: Local Windows computer using uv

## Methodology
The pre-trained model was downloaded from Hugging Face
and executed locally using Python.

The original text entries were supplied to the model
without stemming, lemmatization, or stop-word removal.

The model generated a sentiment label and confidence
score for each entry.

The executable code is saved in pretrained_model_test.py.
The results are saved in pretrained_model_results.csv.

## Results
- Total entries analyzed: 12
- Positive predictions: 9
- Negative predictions: 3

The model supports Positive and Negative classifications,
but does not provide a Neutral category.

## Observations
1. Text 1: DistilBERT and ChatGPT classified the sarcastic
   statement as Negative, while VADER classified it as Neutral.

2. Text 2: DistilBERT classified the passage as Positive
   with 0.9687 confidence, despite references to a
   potential threat to humanity.

3. Text 8: DistilBERT classified the passage as Positive,
   while VADER and ChatGPT classified it as Negative.

4. Text 12: DistilBERT classified the passage as Negative,
   while VADER classified it as Positive and ChatGPT
   classified it as Neutral.

## Limitations
- DistilBERT SST-2 performs binary sentiment classification.
- It cannot directly identify Neutral sentiment.
- Confidence scores do not establish accuracy.
- The dataset contains movie dialogue and narrative descriptions,
  rather than business customer reviews.
- No independently verified reference labels were available.
- The model was used for inference, not fine-tuning or training.

## Conclusion
The experiment demonstrated successful execution of a
pre-trained neural-network model on a local computer.

Comparing DistilBERT, VADER, and ChatGPT revealed differences
in sentiment classification, particularly for sarcasm,
narrative descriptions, and context-dependent language.

The results illustrate why human evaluation and appropriate
model selection are important when applying NLP to
business decision-making.
