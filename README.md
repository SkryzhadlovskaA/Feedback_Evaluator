# Feedback_Evaluator

#Overview

This project evaluates participant feedback from educational and Erasmus+ projects using Natural Language Processing (NLP).

The pipeline automatically:

splits feedback into sentences,
detects sentiment,
identifies learning-related statements,
classifies learning outcomes,
clusters positive and negative themes,
and generates structured CSV outputs.



#Features
1. Sentence Splitting
Breaks long responses into individual sentences for analysis.

Output:
[project_name]_sentences.csv

2. Sentiment Analysis

Classifies sentences as:
positive
negative
neutral

Uses:

VADER sentiment analysis
custom rule-based improvements
negation handling (e.g. “no negative aspects”)

Output:
[project_name]_sentiment.csv

3. Learning Detection

Detects whether a sentence contains educational or learning-related content.

Output:
[project_name]_learning_flags.csv

4. Learning Outcome Classification

Maps learning-related sentences to predefined educational outcome categories using semantic similarity.

Output:
[project_name]_learning_outcomes.csv

5. Clustering

Groups positive and negative feedback into thematic clusters.

Outputs:
[project_name]_positive_clusters.csv
[project_name]_negative_clusters.csv

Cluster labels are loaded from:

clustering_labels.json which can be edited manually, allowing user to select the most relevant to the project labels



#Requirements

Install dependencies:

pip install pandas nltk scikit-learn sentence-transformers

Download NLTK resources automatically:

nltk.download("vader_lexicon")

Running the Pipeline

Run the complete evaluator:

python run_evaluator_all_steps.py



#Notes

Negation Handling

The sentiment module includes rule-based handling for phrases such as:

“no negative aspects”
“nothing bad”
“not bad”

to avoid false negative classifications.
