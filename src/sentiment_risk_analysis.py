#!/usr/bin/env python3

"""
Script Name: sentiment_risk_classification.py

This script applies sentiment analysis and crisis risk classification to a dataset of social media posts.
It uses VADER or TextBlob for sentiment classification (Positive, Neutral, Negative) and TF-IDF to detect
high-risk crisis terms to categorize each post into High-Risk, Moderate Concern, or Low Concern.
The script outputs a table and exactly three plots:
  1) Sentiment distribution
  2) Risk level distribution
  3) Stacked bar chart for sentiment vs. risk level

Usage:
  python sentiment_risk_classification.py \
    --input_file INPUT_FILE_PATH \
    --output_dir OUTPUT_DIRECTORY \
    --plot_dir PLOT_DIRECTORY \
    --high_risk_terms HIGH_RISK_TERMS_FILE \
    --moderate_risk_terms MODERATE_RISK_TERMS_FILE \
    --use_vader True/False \
    --text_column COLUMN_NAME

Example:
  python sentiment_risk_classification.py \
    --input_file data/cleaned_data.csv \
    --output_dir results \
    --plot_dir results/plots \
    --high_risk_terms data/high_risk.txt \
    --moderate_risk_terms data/moderate_risk.txt \
    --use_vader True \
    --text_column content
"""

import argparse
import os
import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm.auto import tqdm  # Import tqdm for progress bars

# Configure tqdm to work with pandas operations
tqdm.pandas()

# Global variables to store high-risk and moderate-risk terms
high_risk_terms = []
moderate_risk_terms = []

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Classify social media posts by sentiment and crisis risk level.")
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True,
        default="data/cleaned_data.csv", 
        help="Path to the input CSV file containing posts."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        default="results", 
        help="Directory to store the output CSV files."
    )
    parser.add_argument(
        "--plot_dir", 
        type=str, 
        default="results/plots", 
        help="Directory to store the generated plots."
    )
    parser.add_argument(
        "--high_risk_terms", 
        type=str, 
        default="data/high_risk.txt", 
        help="Path to text file with high-risk terms, one per line."
    )
    parser.add_argument(
        "--moderate_risk_terms", 
        type=str, 
        default="data/moderate_risk.txt", 
        help="Path to text file with moderate-risk terms, one per line."
    )
    parser.add_argument(
        "--use_vader", 
        type=lambda x: (str(x).lower() == 'true'), 
        default=True, 
        help="Use VADER for sentiment classification if True, otherwise use TextBlob."
    )
    parser.add_argument(
        "--text_column", 
        type=str, 
        default="content", 
        help="Name of the column containing the text data."
    )
    parser.add_argument(
        "--output_csv_name", 
        type=str, 
        default="posts_with_sentiment_and_risk.csv",
        help="Name of the output CSV file containing sentiment and risk columns."
    )
    return parser.parse_args()


def read_terms_from_file(file_path):
    """
    Reads a list of terms from a text file, one term per line.
    If the file is not found, returns a default list of terms.
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            terms = [line.strip().lower() for line in f if line.strip()]
        return terms
    
    # Return a default list if no file provided or file not found
    if file_path and "high_risk" in file_path:
        return ["i don't want to be here anymore", "kill myself", "no reason to live", 
                "suicide", "end my life", "better off dead"]
    else:
        return ["feel lost", "seeking help", "struggling", "depressed", 
                "anxiety", "need support", "therapy"]


def preprocess_text(text):
    """
    Preprocesses text by lowercasing, removing punctuation, tokenizing, 
    and removing stopwords.
    """
    if not isinstance(text, str) or not text:
        return ""
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)


def classify_sentiment_vader(text, vader_analyzer):
    """
    Classifies text sentiment using VADER.
    Returns "Positive", "Negative", or "Neutral".
    """
    if not isinstance(text, str) or not text:
        return "Neutral"
    
    score = vader_analyzer.polarity_scores(text)
    compound = score['compound']
    
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def classify_sentiment_textblob(text):
    """
    Classifies text sentiment using TextBlob.
    Returns "Positive", "Negative", or "Neutral".
    """
    if not isinstance(text, str) or not text:
        return "Neutral"
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"


def build_tfidf_model(corpus, ngram_range=(1, 3)):
    """
    Builds a TF-IDF model from a corpus of texts.
    Returns the TF-IDF vectorizer and the transformed matrix.
    """
    if not corpus or all(not text for text in corpus):
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        return vectorizer, np.zeros((len(corpus), 1))
    
    corpus = [text if text else " " for text in corpus]
    
    # Use tqdm for transforming the corpus to show progress
    print("Building TF-IDF model...")
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=2, max_df=0.95)
    
    with tqdm(total=1, desc="Vectorizing text") as pbar:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        pbar.update(1)
        
    return vectorizer, tfidf_matrix


def detect_risk_terms_tfidf(text, risk_terms_high, risk_terms_moderate, vectorizer, sentiment):
    """
    Detects high and moderate risk terms in text using TF-IDF.
    Takes sentiment into account for more logical risk assignment.
    Returns "High-Risk", "Moderate Concern", or "Low Concern".
    """
    if not isinstance(text, str) or not text:
        return "Low Concern"
    
    text_lower = text.lower()
    
    # Apply higher threshold for risk detection in positive sentiment
    sentiment_weights = {
        "Negative": 1.0,  # Full sensitivity for negative sentiment
        "Neutral": 0.7,   # Reduced sensitivity for neutral sentiment
        "Positive": 0.4   # Further reduced sensitivity for positive sentiment
    }
    
    weight = sentiment_weights.get(sentiment, 1.0)
    
    # Direct pattern matching with sentiment weighting
    # For high risk terms in clearly positive text, require multiple signals
    high_risk_signal = False
    for term in risk_terms_high:
        if term in text_lower:
            if sentiment == "Positive":
                # For positive sentiment, look for exact matches of high-risk terms
                # This ensures we're only using terms from the provided file
                if term in risk_terms_high:
                    high_risk_signal = True
            else:
                return "High-Risk"
    
    # Return high risk if we found explicit signals even in positive text
    if high_risk_signal:
        return "High-Risk"
    
    # For moderate risk terms, apply sentiment weighting
    moderate_risk_count = 0
    for term in risk_terms_moderate:
        if term in text_lower:
            moderate_risk_count += 1
            # If we have enough moderate signals or not positive sentiment
            if moderate_risk_count >= 2 or sentiment != "Positive":
                return "Moderate Concern"
    
    # TF-IDF based detection with sentiment weighting
    if hasattr(vectorizer, 'get_feature_names_out'):
        features = vectorizer.get_feature_names_out()
    else:
        features = vectorizer.get_feature_names()
    
    tfidf_vector = vectorizer.transform([text])
    non_zero = tfidf_vector.nonzero()[1]
    scores = zip(non_zero, [tfidf_vector[0, i] for i in non_zero])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    high_risk_score = 0
    moderate_risk_score = 0
    
    for idx, score in sorted_scores[:10]:
        term = features[idx]
        
        # Calculate risk scores with weighting
        for risk_term in risk_terms_high:
            if risk_term in term or term in risk_term:
                high_risk_score += score * weight
                if high_risk_score > 0.15:  # Threshold with weighting applied
                    return "High-Risk"
        
        for risk_term in risk_terms_moderate:
            if risk_term in term or term in risk_term:
                moderate_risk_score += score * weight
                if moderate_risk_score > 0.2:  # Threshold with weighting applied
                    return "Moderate Concern"
    
    return "Low Concern"


def adjust_risk_level(row, text_column):
    """
    Applies logical rules to check and adjust for anomalies in risk level classification.
    Uses the global high_risk_terms and moderate_risk_terms variables.
    """
    global high_risk_terms, moderate_risk_terms
    
    # If positive sentiment but high risk, look more closely
    if row['sentiment'] == 'Positive' and row['risk_level'] == 'High-Risk':
        # Look for explicit high-risk terms in positive posts
        text_lower = row[text_column].lower() if isinstance(row[text_column], str) else ""
        # Only use the terms from the provided high-risk terms file
        if not any(term in text_lower for term in high_risk_terms):
            return "Moderate Concern"  # Downgrade unless very explicit
    
    # If negative sentiment but low concern, check for potential missed signals
    if row['sentiment'] == 'Negative' and row['risk_level'] == 'Low Concern':
        # Look for moderate risk terms using only those from the provided file
        text_lower = row[text_column].lower() if isinstance(row[text_column], str) else ""
        if any(term in text_lower for term in moderate_risk_terms):
            return "Moderate Concern"  # Upgrade if distress signals found
    
    return row['risk_level']  # Keep original classification otherwise


def generate_distribution_plots(df, output_dir):
    """
    Generates exactly three plots:
      1) Distribution of sentiment
      2) Distribution of risk level
      3) Stacked bar chart of sentiment vs. risk level
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a consistent color scheme
    risk_colors = {
        "High-Risk": "#d73027",       # Red
        "Moderate Concern": "#fee090", # Yellow
        "Low Concern": "#4575b4"       # Blue
    }
    
    sentiment_order = ["Negative", "Neutral", "Positive"]
    risk_order = ["High-Risk", "Moderate Concern", "Low Concern"]
    
    print("Generating plots...")
    with tqdm(total=3, desc="Generating plots") as pbar:
        # 1) Sentiment Distribution
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(x='sentiment', data=df, order=sentiment_order, palette="Blues_d")
        plt.title('Sentiment Distribution', fontsize=14)
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Number of Posts', fontsize=12)
        plt.xticks(rotation=0)
        
        # Add counts on top of bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'bottom', fontsize=10)
        
        sentiment_plot_path = os.path.join(output_dir, 'sentiment_distribution.png')
        plt.tight_layout()
        plt.savefig(sentiment_plot_path, dpi=300)
        plt.close()
        pbar.update(1)
        
        # 2) Risk Level Distribution
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(x='risk_level', data=df, order=risk_order, palette=[risk_colors[r] for r in risk_order])
        plt.title('Risk Level Distribution', fontsize=14)
        plt.xlabel('Risk Level', fontsize=12)
        plt.ylabel('Number of Posts', fontsize=12)
        plt.xticks(rotation=0)
        
        # Add counts on top of bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'bottom', fontsize=10)
        
        risk_plot_path = os.path.join(output_dir, 'risk_level_distribution.png')
        plt.tight_layout()
        plt.savefig(risk_plot_path, dpi=300)
        plt.close()
        pbar.update(1)
        
        # 3) Stacked Bar Chart: Sentiment vs Risk Level
        crosstab = pd.crosstab(df['sentiment'], df['risk_level'])
        crosstab = crosstab.reindex(index=sentiment_order, columns=risk_order)
        
        plt.figure(figsize=(10, 6))
        crosstab.plot(kind='bar', stacked=True, color=[risk_colors[r] for r in risk_order])
        plt.title('Sentiment vs Risk Level Distribution', fontsize=14)
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Number of Posts', fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title='Risk Level')
        
        # Add percentage labels
        totals = crosstab.sum(axis=1)
        for i, sentiment in enumerate(crosstab.index):
            cumulative_height = 0
            for j, risk in enumerate(crosstab.columns):
                height = crosstab.loc[sentiment, risk]
                percentage = height / totals[sentiment] * 100
                if height > 0:  # Only label non-zero values
                    plt.text(i, cumulative_height + height/2, 
                            f'{percentage:.1f}%', 
                            ha='center', va='center', 
                            color='black' if risk != 'High-Risk' else 'white',
                            fontweight='bold')
                cumulative_height += height
        
        stacked_bar_path = os.path.join(output_dir, 'sentiment_vs_risk_stacked.png')
        plt.tight_layout()
        plt.savefig(stacked_bar_path, dpi=300)
        plt.close()
        pbar.update(1)
    
    print(f"Plots saved to {output_dir}")
    
    return {
        'sentiment_counts': df['sentiment'].value_counts().to_dict(),
        'risk_counts': df['risk_level'].value_counts().to_dict(),
        'sentiment_risk_crosstab': crosstab.to_dict()
    }


def main():
    """
    Main function that orchestrates the sentiment and risk classification process.
    """
    global high_risk_terms, moderate_risk_terms
    
    args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Load risk terms with progress
    with tqdm(total=2, desc="Loading risk terms") as pbar:
        high_risk_terms = read_terms_from_file(args.high_risk_terms)
        pbar.update(1)
        moderate_risk_terms = read_terms_from_file(args.moderate_risk_terms)
        pbar.update(1)
    
    print(f"Loaded {len(high_risk_terms)} high-risk terms and {len(moderate_risk_terms)} moderate-risk terms")
    
    # Read dataset with progress
    print(f"Reading data from {args.input_file}")
    with tqdm(total=1, desc="Reading data") as pbar:
        df = pd.read_csv(args.input_file)
        pbar.update(1)
    
    if args.text_column not in df.columns:
        raise ValueError(
            f"Text column '{args.text_column}' not found in input file. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    print("Preprocessing text data...")
    # Use progress_apply for preprocessing text
    df['preprocessed_text'] = df[args.text_column].progress_apply(preprocess_text)
    
    print("Building TF-IDF model...")
    vectorizer, _ = build_tfidf_model(df['preprocessed_text'].tolist())
    
    print("Classifying sentiment...")
    # Use progress_apply for sentiment classification
    if args.use_vader:
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        vader_analyzer = SentimentIntensityAnalyzer()
        df['sentiment'] = df[args.text_column].progress_apply(lambda x: classify_sentiment_vader(x, vader_analyzer))
    else:
        df['sentiment'] = df[args.text_column].progress_apply(classify_sentiment_textblob)
    
    print("Detecting risk levels with TF-IDF...")
    # First pass to do risk detection with progress bar
    df['risk_level'] = df.progress_apply(
        lambda row: detect_risk_terms_tfidf(
            row[args.text_column], 
            high_risk_terms, 
            moderate_risk_terms, 
            vectorizer, 
            row['sentiment']
        ), 
        axis=1
    )
    
    # Add logic to ensure risk levels make logical sense with sentiment
    # Run analysis on current distribution
    initial_sentiment_risk_counts = pd.crosstab(df['sentiment'], df['risk_level'])
    print("\nInitial distribution:")
    print(initial_sentiment_risk_counts)
    
    # Apply risk level adjustments with progress bar
    print("Adjusting risk levels...")
    df['risk_level'] = df.progress_apply(lambda row: adjust_risk_level(row, args.text_column), axis=1)
    
    # Show the adjusted distribution
    adjusted_sentiment_risk_counts = pd.crosstab(df['sentiment'], df['risk_level'])
    print("\nAdjusted distribution:")
    print(adjusted_sentiment_risk_counts)
    
    # Generate plots with built-in progress bars in generate_distribution_plots
    stats = generate_distribution_plots(df, args.plot_dir)
    
    print("Saving results...")
    # Save results with progress
    with tqdm(total=2, desc="Saving results") as pbar:
        output_csv_path = os.path.join(args.output_dir, args.output_csv_name)
        df.to_csv(output_csv_path, index=False)
        pbar.update(1)
        
        stats_df = pd.DataFrame({
            'sentiment_counts': pd.Series(stats['sentiment_counts']),
            'risk_counts': pd.Series(stats['risk_counts'])
        })
        stats_path = os.path.join(args.output_dir, 'summary_statistics.csv')
        stats_df.to_csv(stats_path)
        pbar.update(1)
    
    print(f"Results saved to {output_csv_path}")
    print(f"Summary statistics saved to {stats_path}")
    
    # Store sentiment and risk counts for summary output
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    risk_counts = df['risk_level'].value_counts().to_dict()
    
    print("\nSummary:")
    print(f"Total posts processed: {len(df)}")
    
    print("\nSentiment distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\nRisk level distribution:")
    for risk, count in risk_counts.items():
        print(f"  {risk}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\nSentiment vs Risk Level:")
    sentiment_risk_counts = pd.crosstab(df['sentiment'], df['risk_level'])
    print(sentiment_risk_counts)
    print("\nProcess completed successfully.")


if __name__ == "__main__":
    main()