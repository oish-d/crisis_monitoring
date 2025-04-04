#!/usr/bin/env python3

"""
Usage:
  python data_collection.py \
    --keywords KEYWORD_FILE \
    --subreddits SUBREDDITS_FILE \
    --raw_output RAW_OUTPUT_PATH \
    --cleaned_output CLEANED_OUTPUT_PATH \
    --limit NUM_POSTS

Example:
  python data_collection.py \
    --keywords default_keywords.txt \
    --subreddits subreddits.txt \
    --raw_output ./data/raw_data.csv \
    --cleaned_output ./data/cleaned_data.csv \
    --limit 100

This script connects to Reddit, fetches posts for each keyword from specified subreddits, 
and stores two outputs:
  1) A raw CSV of the collected data
  2) A cleaned CSV, where the 'title' and 'content' fields are processed using a text-cleaning function.

Functions:
  parse_arguments():
    Parses command-line arguments.

  fetch_reddit_data(reddit, keywords, subreddits, limit):
    Fetches Reddit posts for each keyword from specified subreddits and returns a list of dictionaries.

  save_data_to_csv(data, output_path):
    Saves a list of dictionaries to a CSV file at the specified path, creating directories if needed.

  clean_text(text, stop_words):
    Cleans input text by removing emojis, URLs, special characters, extra whitespace, and stopwords.

  clean_data(raw_data, stop_words):
    Returns a new list of dictionaries where both 'title' and 'content' are replaced 
    by their cleaned versions.

  main():
    Orchestrates argument parsing, data fetching, saving raw data, cleaning, and saving the cleaned data.
"""

import argparse
import os
import csv
import re
import praw
import nltk
from nltk.corpus import stopwords
import copy
from tqdm import tqdm

# NOTE: These API credentials are included for evaluation purposes only.
# In a production environment, these would be stored securely.
CLIENT_ID = "L0k-UawcE-Idh0O6D5JJLg"
CLIENT_SECRET = "CbAqaABnT7ZoRjUqY4ghDm1gc6Jo8w"
USER_AGENT = "myRedditApp/0.1 by Radiant_Resort_909"

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keywords",
        default="./data/keyword_dictionary.txt",
        help="Path to the file containing keywords, one per line.",
    )
    parser.add_argument(
        "--subreddits",
        default="./data/subreddits.txt",
        help="Path to the file containing subreddits to search, one per line.",
    )
    parser.add_argument(
        "--raw_output",
        default="./data/raw_data.csv",
        help="Path to the raw CSV output file.",
    )
    parser.add_argument(
        "--cleaned_output",
        default="./data/cleaned_data.csv",
        help="Path to the cleaned CSV output file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of Reddit posts per keyword per subreddit.",
    )
    return parser.parse_args()


def fetch_reddit_data(reddit, keywords, subreddits, limit):
    """
    Returns a list of dictionaries containing post data, such as id, title, created_utc, etc.
    Keywords is a list of strings; subreddits is a list of subreddit names; 
    limit is the number of posts per keyword per subreddit.
    """
    collected_data = []
    
    # Create a total progress bar
    total_combinations = len(keywords) * len(subreddits)
    with tqdm(total=total_combinations, desc="Fetching Reddit data") as pbar:
        for sr in subreddits:
            for kw in keywords:
                try:
                    # Search within a specific subreddit
                    subreddit = reddit.subreddit(sr)
                    posts = list(subreddit.search(kw, limit=limit))
                    
                    for submission in posts:
                        post_info = {
                            "id": submission.id,
                            "title": submission.title,
                            "subreddit": submission.subreddit.display_name,
                            "created_utc": submission.created_utc,
                            "num_comments": submission.num_comments,
                            "score": submission.score,
                            "content": submission.selftext,
                            "keyword": kw,
                        }
                        collected_data.append(post_info)
                    
                    # Update progress bar description with current subreddit and keyword
                    pbar.set_description(f"Fetched {len(posts)} posts from r/{sr} for '{kw}'")
                except Exception as e:
                    print(f"Error fetching from r/{sr} with keyword '{kw}': {e}")
                
                # Update progress bar after each subreddit-keyword combination
                pbar.update(1)
    
    return collected_data


def save_data_to_csv(data, output_path):
    """
    Saves a list of dictionaries (data) to a CSV file at output_path.
    Creates the output directory if it does not exist.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not data:
        fieldnames = ["id", "title", "subreddit", "created_utc", "num_comments", "score", "content", "keyword"]
    else:
        fieldnames = list(data[0].keys())
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def clean_text(text, stop_words):
    """
    Cleans input text by:
    - Removing emojis
    - Removing URLs
    - Removing special characters (letters, numbers, spaces remain)
    - Removing extra whitespace
    - Removing English stopwords
    """
    if not text:
        return ""
        
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub("", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    filtered_words = [w for w in words if w.lower() not in stop_words]
    return " ".join(filtered_words)


def clean_data(raw_data, stop_words):
    """
    Returns a new list of dictionaries where both 'title' and 'content' are replaced 
    by their cleaned versions.
    """
    cleaned = []
    # Show progress bar for cleaning data
    for item in tqdm(raw_data, desc="Cleaning data"):
        # Create a deep copy to ensure we're not modifying the original data
        new_item = copy.deepcopy(item)
        new_item["title"] = clean_text(item["title"], stop_words)
        new_item["content"] = clean_text(item["content"], stop_words)
        cleaned.append(new_item)
    return cleaned


def main():
    args = parse_arguments()
    
    # Read keywords from file
    with open(args.keywords, "r", encoding="utf-8") as file:
        keyword_list = [line.strip() for line in file if line.strip()]
    
    # Read subreddits from file
    with open(args.subreddits, "r", encoding="utf-8") as file:
        subreddit_list = [line.strip() for line in file if line.strip()]
    
    print(f"Loaded {len(keyword_list)} keywords and {len(subreddit_list)} subreddits")

    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )
    print("Reddit instance created. User:", reddit.user.me())

    raw_data = fetch_reddit_data(reddit, keyword_list, subreddit_list, args.limit)
    save_data_to_csv(raw_data, args.raw_output)
    print(f"Raw data saved to {args.raw_output}")

    cleaned_data = clean_data(raw_data, stop_words)
    save_data_to_csv(cleaned_data, args.cleaned_output)
    print(f"Cleaned data saved to {args.cleaned_output}")
    print(f"Data collection complete. {len(raw_data)} posts fetched.")


if __name__ == "__main__":
    main()
