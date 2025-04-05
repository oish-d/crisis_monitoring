# Crisis Monitoring Dashboard

A comprehensive solution for extracting, analyzing, and visualizing crisis-related discussions from social media platforms. This project identifies high-risk content using sentiment analysis and NLP techniques, combined with geographic visualization to highlight regional crisis patterns.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B)](https://gsoc-crisis-monitoring-dashboard-oishika.streamlit.app/)

## 📋 Project Overview

This work was developed for the GSoC candidate assessment on the *AI-Powered Behavioral Analysis for Suicide Prevention, Substance Use, and Mental Health Crisis Detection with Longitudinal Geospatial Crisis Trend Analysis* project under ISSR. It consists of three main components:

1. **Data Collection**: Extracts relevant mental health posts from Reddit using predefined keywords
2. **Sentiment & Risk Analysis**: Classifies posts based on sentiment and crisis risk level
3. **Geolocation & Visualization**: Maps crisis discussions geographically to identify regional patterns

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crisis-monitoring-dashboard.git
cd crisis-monitoring-dashboard
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

The main dependencies include:
- praw (Reddit API wrapper)
- nltk (Natural Language Processing)
- pandas (Data manipulation)
- spacy (NLP for location extraction)
- folium (Interactive maps)
- plotly (Data visualization)
- streamlit (Dashboard interface)
- geopy (Geocoding)

## 🔍 Task 1: Social Media Data Extraction & Preprocessing

### Description
The `data_collection.py` script connects to the Reddit API, fetches posts related to mental health using predefined keywords, and cleans the text data by removing emojis, URLs, special characters, and stopwords.

### Input
- Keywords file (`data/keyword_dictionary.txt`) containing search terms related to mental health, substance abuse and suicide
- Subreddits file (`data/subreddits.txt`) containing relevant subreddit names to search
- Command-line parameters for output paths and post limit
  
### Output
- Raw data CSV (`data/raw_data.csv`) containing post ID, title, content, subreddit, timestamp, etc.
- Cleaned data CSV (`data/cleaned_data.csv`) with preprocessed text ready for NLP analysis

### Libraries Used
- `praw`: Reddit API wrapper for Python
- `nltk`: For stopwords removal
- `re`: For regex-based text cleaning
- `tqdm`: For progress tracking

### Usage
```bash
python src/data_collection.py \
  --keywords data/keyword_dictionary.txt \
  --subreddits data/subreddits.txt \
  --raw_output data/raw_data.csv \
  --cleaned_output data/cleaned_data.csv \
  --limit 12
```

⚠️ **API Credentials Note:**
This repository includes Reddit API credentials in the code for evaluation purposes only. In a production environment, these would be stored as environment variables or in a secure configuration file.

## 📊 Task 2: Sentiment & Crisis Risk Classification

### Description
The sentiment_risk_analysis.py script performs sentiment analysis using VADER or TextBlob and classifies posts into risk categories based on crisis-related language patterns. Specifically, high-risk and moderate-risk terms were defined, with posts that don't match these criteria being classified as low risk.

### Input
- Cleaned data CSV from Task 1 (`data/cleaned_data.csv`)
- High-risk terms file (`data/high_risk.txt`) containing terms indicating immediate crisis
- Moderate-risk terms file (`data/moderate_risk.txt`) containing terms indicating distress
- Command-line parameters for output paths and configuration options

### Output
- CSV with sentiment scores and risk classifications (`results/posts_with_sentiment_and_risk.csv`)
- Three visualization plots:
  1. Sentiment distribution (`results/plots/sentiment_distribution.png`)
  2. Risk level distribution (High-Risk, Moderate Concern, and Low Concern) (`results/plots/risk_level_distribution.png`)
  3. Stacked bar chart of sentiment vs. risk level (`results/plots/sentiment_vs_risk_stacked.png`)

### Libraries Used
- `nltk.sentiment.vader`: For sentiment analysis
- `textblob`: Alternative sentiment analyzer
- `sklearn`: For TF-IDF vectorization
- `matplotlib` and `seaborn`: For visualization

### Usage
```bash
python src/sentiment_risk_analysis.py \
  --input_file data/cleaned_data.csv \
  --output_dir results \
  --plot_dir results/plots \
  --high_risk_terms data/high_risk.txt \
  --moderate_risk_terms data/moderate_risk.txt \
  --use_vader True \
  --text_column content
```

## 🗺️ Task 3: Crisis Geolocation & Mapping

### Description
The `crisis_geolocation.py` script extracts location mentions from post content, geocodes them to coordinates, and generates interactive maps and visualizations to identify crisis hotspots.

### Input
- CSV with sentiment and risk data from Task 2 (`results/posts_with_sentiment_and_risk.csv`)
- Non-location words file (`data/non_location_words.txt`) for filtering false positives
- Command-line parameters for risk level filtering and output configuration

### Output
- Geocoded locations CSV (`results/geo_analysis/geocoded_locations.csv`)
- Interactive Folium heatmap (`results/geo_analysis/crisis_heatmap.html`)
- Plotly choropleth map (`results/geo_analysis/crisis_choropleth.html`)
- Top regions visualization (`results/geo_analysis/crisis_choropleth_top_regions.html`)
- Location breakdown data (`results/geo_analysis/location_breakdown.csv`)
- Top locations visualization (`results/geo_analysis/top_locations_chart.png`)
- Top locations summary in JSON format (`results/geo_analysis/top_locations.json`)

### Libraries Used
- `spacy`: For named entity recognition to identify locations
- `geopy`: For geocoding locations to coordinates
- `folium`: For interactive maps and heatmaps
- `plotly`: For choropleth maps

### Usage
```bash
python src/crisis_geolocation.py \
  --input_file results/posts_with_sentiment_and_risk.csv \
  --output_dir results/geo_analysis \
  --risk_level "High-Risk" \
  --text_column content \
  --non_location_file data/non_location_words.txt
```

## 📱 Streamlit Dashboard Demo

The project includes an interactive Streamlit dashboard that visualizes all the analysis results in a user-friendly interface.
### Live Demo

A live demonstration of the dashboard is available at:
[https://gsoc-crisis-monitoring-dashboard-oishika.streamlit.app/](https://gsoc-crisis-monitoring-dashboard-oishika.streamlit.app/)

### Features
- **Dashboard Overview**: Key metrics and visualizations at a glance
- **Data Explorer**: Browse and filter raw, cleaned, and processed data
- **Sentiment Analysis**: Detailed sentiment and risk visualizations with post examples
- **Geospatial Analysis**: Interactive heatmaps and choropleth maps showing crisis hotspots
- **About the Project**: Project overview, workflow, and technologies used
- **About Me**: Academic background, skills, projects and experiences
  
### Usage
```bash
streamlit run app.py
```

### Navigation
The dashboard offers five main sections accessible from the sidebar:

1. **Dashboard Overview**
   - Summary metrics (total posts, high-risk count, negative sentiment count)
   - Sentiment and risk distribution charts
   - Geographic heatmap of high-risk posts
   - Top 5 locations with crisis discussions

2. **Data Explorer**
   - Raw data viewer with subreddit and keyword distributions
   - Cleaned data viewer with text cleaning examples
   - Processed data with sentiment and risk filtering options

3. **Sentiment Analysis**
   - Detailed sentiment and risk visualizations
   - Post explorer with sentiment and risk filters
   - Example posts with sentiment and risk classifications

4. **Geospatial Analysis**
   - Interactive crisis heatmap
   - Global choropleth map showing crisis distribution by country
   - Top regions bar chart
   - Top locations breakdown

5. **About the Project**
   - Project overview and methodology
   - Repository structure
   - Command-line workflow
   - Technologies used
  
6. **About Me**
   - Academic background
   - Skills and experience
     
## 📊 Key Visualizations

The project generates several visualizations to help understand the crisis data:

1. **Sentiment Distribution**: Bar chart showing positive, neutral, and negative post counts
2. **Risk Level Distribution**: Bar chart of high-risk, moderate-concern, and low-concern posts
3. **Sentiment vs. Risk**: Stacked bar chart showing the relationship between sentiment and risk
4. **Crisis Heatmap**: Interactive map showing the geographic concentration of crisis posts
5. **Global Crisis Distribution**: Choropleth map showing crisis posts by country
6. **Top Regions Chart**: Bar chart of the top 15 regions with the highest crisis discussions
7. **Top Locations Chart**: Horizontal bar chart of the top 10 specific locations mentioned

## 🙏 Acknowledgements

I would like to express my gratitude for the opportunity to participate in the GSoC Candidate Assessment for HumanAI organisation. This project has allowed me to demonstrate skills in data extraction, NLP, sentiment analysis, and geospatial visualization while addressing the important topic of mental health crisis monitoring.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
