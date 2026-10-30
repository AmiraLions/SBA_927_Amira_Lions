
# SBA 927 — Requirement 6: Sentiment Analysis Evaluation

## Objective
Evaluate sentiment classifications generated using NLTK VADER
and OpenAI ChatGPT (GPT-6) on the same 12 text entries.

## Methods
- Dataset: SBA927.txt (12 entries from I, Robot).
- VADER: Executed locally using Python and NLTK.
- ChatGPT: Used the Requirement 5 sentiment prompt to
  classify the entries as Positive, Negative, or Neutral.
- Comparison: Saved in sentiment_comparison.csv.

ChatGPT analysis was performed interactively in ChatGPT,
not through a Python API.

## Results
- Total entries: 12
- Agreements: 5
- Disagreements: 7
- Agreement rate: 41.7%

## Observations
1. Sarcasm: Text 1 was Neutral in VADER and Negative
   in ChatGPT. Sarcasm may require contextual interpretation.

2. Mixed sentiment: Text 5 was Positive in VADER and
   Negative in ChatGPT. The passage contains both
   positive and threatening language.

3. Negation: Text 10 was Positive in VADER and Negative
   in ChatGPT. A denial involving a serious accusation
   can be difficult to classify.

## Limitations
- Agreement is not the same as accuracy.
- No independently verified reference labels were used.
- The dataset contains movie dialogue and descriptions,
  rather than actual customer reviews.
- ChatGPT classifications are interpretations and may
  vary across runs.
- Neither model should be treated as automatically correct.

## Conclusion
VADER provides repeatable numerical sentiment scores.
ChatGPT can interpret broader context and explain its
classifications. Comparing their outputs revealed
differences that require human review.

These findings illustrate the importance of evaluating
NLP outputs before using them for business decisions.
