
from pathlib import Path

# Locate the dataset in the same folder as this Python file.
file_path = Path(__file__).parent / "SBA927.txt"

# Read the dataset, keeping each line as a separate text entry.
with open(file_path, "r", encoding="utf-8") as file:
    dataset = file.read().splitlines()

# Confirm the dataset loaded successfully.
print("SBA 927 - Dataset Verification")
print("Number of text entries:", len(dataset))
print("First entry:", dataset[0])

# Requirement 1: Tokenization and Preprocessing

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load English stop words.
# Preserve words that can reverse the meaning of a review.
negation_words = {"no", "not", "nor", "never", "n't"}

stop_words = set(stopwords.words("english")) - negation_words

def tokenize_and_preprocess(text):
    # Break text into individual tokens.
    tokens = word_tokenize(text)

    # Normalize text and remove punctuation, numbers, and stop words.
    filtered_tokens = [
        word.lower()
        for word in tokens
        if (word.isalpha() or word == "n't") and word.lower() not in stop_words
    ]

    return filtered_tokens


# Process all 12 entries in the dataset.
processed_dataset = [
    tokenize_and_preprocess(text) for text in dataset
]

# Display the first entry before and after preprocessing.
print("\nRequirement 1 - Tokenization and Preprocessing")
print("Original:", dataset[0])
print("Processed:", processed_dataset[0])

# Verify that preprocessing preserves important negation.
test_review = "I don't like this product. It is not good."

print("\nNegation Preservation Test")
print("Original:", test_review)
print("Processed:", tokenize_and_preprocess(test_review))


# Requirement 2: Stemming and Lemmatization

from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def stem_and_lemmatize(tokens):
    stemmed_tokens = [
        stemmer.stem(token) for token in tokens
    ]

    lemmatized_tokens = [
        lemmatizer.lemmatize(token) for token in tokens
    ]

    return stemmed_tokens, lemmatized_tokens


# Apply both techniques to all 12 entries.
stemmed_and_lemmatized_dataset = [
    stem_and_lemmatize(tokens)
    for tokens in processed_dataset
]

# Display results for the first entry.
stemmed_tokens, lemmatized_tokens = (
    stemmed_and_lemmatized_dataset[0]
)

print("\nRequirement 2 - Stemming and Lemmatization")
print("Original Tokens:", processed_dataset[0])
print("Stemmed Tokens:", stemmed_tokens)
print("Lemmatized Tokens:", lemmatized_tokens)


# Requirement 3: Named Entity Recognition (NER)

from nltk import pos_tag, ne_chunk
from nltk.tokenize import sent_tokenize


def named_entity_recognition(text):
    named_entities = []

    # Split text into sentences.
    sentences = sent_tokenize(text)

    for sentence in sentences:
        # Tokenize and assign grammatical tags.
        tokens = word_tokenize(sentence)
        tagged_words = pos_tag(tokens)

        # Identify named entities.
        entity_tree = ne_chunk(tagged_words)

        for subtree in entity_tree:
            if hasattr(subtree, "label"):
                entity_name = " ".join(
                    word for word, tag in subtree.leaves()
                )

                named_entities.append(
                    (entity_name, subtree.label())
                )

    return named_entities


# Apply NER to all 12 original dataset entries.
ner_results_dataset = [
    named_entity_recognition(text)
    for text in dataset
]

print("\nRequirement 3 - Named Entity Recognition")

for i, entities in enumerate(ner_results_dataset, start=1):
    print(f"Text {i}: {entities}")


# Requirement 4: Part-of-Speech (POS) Tagging

def part_of_speech_tagging(text):
    tokens = word_tokenize(text)
    tagged_words = pos_tag(tokens)
    return tagged_words

pos_results_dataset = [
    part_of_speech_tagging(text)
    for text in dataset
]

print("\nRequirement 4 - Part-of-Speech Tagging")
print("Original:", dataset[0])
print("POS Tags:", pos_results_dataset[0])

# Requirement 5: Sentiment Analysis Prompt Creation

def create_sentiment_prompt(text):
    prompt = f"""
Analyze the sentiment of the following text.

Classify the sentiment as:
- Positive
- Negative
- Neutral

Consider the full meaning of the statement, including
negation and context.

Return:
Sentiment: [Positive, Negative, or Neutral]
Explanation: [One brief sentence explaining the classification]

Text: {text}
"""
    return prompt


sentiment_prompts = [
    create_sentiment_prompt(text)
    for text in dataset
]

print("\nRequirement 5 - Sentiment Prompt Creation")
print(sentiment_prompts[0])


# Requirement 6: Sentiment Analysis Evaluation

from nltk.sentiment import SentimentIntensityAnalyzer

sentiment_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, compound


sentiment_results_dataset = [
    analyze_sentiment(text)
    for text in dataset
]

print("\nRequirement 6 - VADER Sentiment Analysis")

for i, (sentiment, score) in enumerate(
    sentiment_results_dataset, start=1
):
    print(f"Text {i}: {sentiment} (compound score: {score:.4f})")
