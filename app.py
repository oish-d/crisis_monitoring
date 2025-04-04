import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import json
import os
from PIL import Image
import base64
from io import BytesIO
from streamlit_folium import folium_static

# Set page configuration
st.set_page_config(
    page_title="Crisis Monitoring Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        text-align: center;
        color: #616161;
    }
    .risk-high {
        color: #d73027;
        font-weight: bold;
    }
    .risk-moderate {
        color: #fee090;
        font-weight: bold;
    }
    .risk-low {
        color: #4575b4;
        font-weight: bold;
    }
    .sentiment-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #9E9E9E;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #F44336;
        font-weight: bold;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Define data loading functions
@st.cache_data
def load_sentiment_risk_data():
    try:
        return pd.read_csv("results/posts_with_sentiment_and_risk.csv")
    except FileNotFoundError:
        st.error("Sentiment and risk data file not found. Please check the file path.")
        return pd.DataFrame()

@st.cache_data
def load_geocoded_data():
    try:
        return pd.read_csv("results/geo_analysis/geocoded_locations.csv")
    except FileNotFoundError:
        st.error("Geocoded data file not found. Please check the file path.")
        return pd.DataFrame()

@st.cache_data
def load_location_breakdown():
    try:
        return pd.read_csv("results/geo_analysis/location_breakdown.csv")
    except FileNotFoundError:
        st.error("Location breakdown file not found. Please check the file path.")
        return pd.DataFrame()

@st.cache_data
def load_raw_data():
    try:
        return pd.read_csv("data/raw_data.csv")
    except FileNotFoundError:
        st.error("Raw data file not found. Please check the file path.")
        return pd.DataFrame()

@st.cache_data
def load_cleaned_data():
    try:
        return pd.read_csv("data/cleaned_data.csv")
    except FileNotFoundError:
        st.error("Cleaned data file not found. Please check the file path.")
        return pd.DataFrame()

@st.cache_data
def load_top_locations():
    try:
        with open("results/geo_analysis/top_locations.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Top locations JSON file not found. Please check the file path.")
        return []

# Function to create heatmap using folium
def create_folium_heatmap(locations_df):
    if len(locations_df) == 0:
        st.warning("No geocoded data available to create heatmap.")
        return None
    
    # Center the map on the mean coordinates
    center_lat = locations_df['latitude'].mean()
    center_lon = locations_df['longitude'].mean()
    
    # Create a base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=3, tiles='CartoDB positron')
    
    # Add a heatmap layer
    heat_data = [[row['latitude'], row['longitude']] for _, row in locations_df.iterrows()]
    HeatMap(heat_data, radius=15).add_to(m)
    
    # Add a marker cluster layer for individual points
    marker_cluster = MarkerCluster().add_to(m)
    
    for _, row in locations_df.iterrows():
        popup_text = f"""
        <b>Location:</b> {row['full_location']}<br>
        <b>Risk Level:</b> {row['risk_level']}<br>
        <b>Sentiment:</b> {row['sentiment']}<br>
        """
        
        # Color markers based on risk level
        if row['risk_level'] == 'High-Risk':
            icon_color = 'red'
        elif row['risk_level'] == 'Moderate Concern':
            icon_color = 'orange'
        else:
            icon_color = 'blue'
            
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color=icon_color, icon='info-sign')
        ).add_to(marker_cluster)
    
    return m

# Function to create choropleth map using plotly
def create_plotly_choropleth(locations_df):
    if len(locations_df) == 0:
        st.warning("No geocoded data available to create choropleth map.")
        return None
    
    # Extract country information from full_location
    def extract_country(location_str):
        if not isinstance(location_str, str):
            return None
        
        # Try to extract the last part which is often the country
        parts = location_str.split(',')
        if parts:
            # Get the last non-empty part
            country_part = next((part.strip() for part in reversed(parts) if part.strip()), None)
            return country_part
        
        return None
    
    # Add country column if it doesn't exist
    if 'country' not in locations_df.columns:
        locations_df['country'] = locations_df['full_location'].apply(extract_country)
    
    # Count occurrences by country
    country_counts = locations_df['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    
    # Filter out None and invalid countries
    country_counts = country_counts[country_counts['country'].notna()]
    
    # Create choropleth map of the world
    fig = px.choropleth(
        country_counts,
        locations='country',
        locationmode='country names',
        color='count',
        color_continuous_scale='Reds',
        title=f'Crisis Posts by Country ({locations_df["risk_level"].iloc[0] if "risk_level" in locations_df.columns else "All Posts"})',
        labels={'count': 'Number of Posts'}
    )
    
    # Make the layout more appealing
    fig.update_layout(
        geo=dict(
            showland=True,
            showlakes=True,
            showcountries=True,
            showocean=True,
            oceancolor='LightBlue',
            lakecolor='LightBlue',
            landcolor='WhiteSmoke',
            countrycolor='Gray'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600
    )
    
    return fig

def create_region_bar_chart(locations_df):
    if len(locations_df) == 0:
        st.warning("No geocoded data available to create region bar chart.")
        return None
    
    # First, get regions from country and city info
    if 'region' not in locations_df.columns:
        locations_df['region'] = locations_df['full_location'].apply(
            lambda loc: loc.split(',')[0].strip() if isinstance(loc, str) and ',' in loc else loc
        )
    
    # Count by region
    region_counts = locations_df['region'].value_counts().reset_index().head(15)
    region_counts.columns = ['region', 'count']
    
    # Create a bar chart of top regions
    fig = px.bar(
        region_counts,
        x='region', 
        y='count',
        title=f'Top 15 Regions with Crisis Posts ({locations_df["risk_level"].iloc[0] if "risk_level" in locations_df.columns else "All Posts"})',
        labels={'count': 'Number of Posts', 'region': 'Region'},
        color='count',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=500)
    
    return fig

# Function to display sentiment distribution plot
def display_sentiment_plot():
    try:
        st.image("results/plots/sentiment_distribution.png", use_column_width=True)
    except FileNotFoundError:
        st.error("Sentiment distribution plot not found.")

# Function to display risk distribution plot
def display_risk_plot():
    try:
        st.image("results/plots/risk_level_distribution.png", use_column_width=True)
    except FileNotFoundError:
        st.error("Risk level distribution plot not found.")

# Function to display sentiment vs risk stacked plot
def display_sentiment_risk_plot():
    try:
        st.image("results/plots/sentiment_vs_risk_stacked.png", use_column_width=True)
    except FileNotFoundError:
        st.error("Sentiment vs risk stacked plot not found.")

# Function to display location chart
def display_location_chart():
    try:
        st.image("results/geo_analysis/top_locations_chart.png", use_column_width=True)
    except FileNotFoundError:
        st.error("Top locations chart not found.")

# Main application
def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/mental-health.png", width=80)
    st.sidebar.title("Navigation")
    
    # Navigation options
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard Overview", "Data Explorer", "Sentiment Analysis", "Geospatial Analysis", "About the Project", "About Me"]
    )
    
    # Load data based on selected page
    if page in ["Dashboard Overview", "Data Explorer", "Sentiment Analysis"]:
        sentiment_risk_df = load_sentiment_risk_data()
    
    if page in ["Dashboard Overview", "Geospatial Analysis"]:
        geocoded_df = load_geocoded_data()
        location_breakdown_df = load_location_breakdown()
        top_locations = load_top_locations()
    
    if page == "Data Explorer":
        raw_df = load_raw_data()
        cleaned_df = load_cleaned_data()
    
    # Dashboard Overview Page
    if page == "Dashboard Overview":
        st.markdown("<h1 class='main-header'>Crisis Monitoring Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Analyzing social media posts for mental health crisis signals</p>", unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{len(sentiment_risk_df)}</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Total Posts Analyzed</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            high_risk_count = len(sentiment_risk_df[sentiment_risk_df['risk_level'] == 'High-Risk'])
            high_risk_percentage = (high_risk_count / len(sentiment_risk_df) * 100) if len(sentiment_risk_df) > 0 else 0
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value risk-high'>{high_risk_count}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>High-Risk Posts ({high_risk_percentage:.1f}%)</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            negative_sentiment_count = len(sentiment_risk_df[sentiment_risk_df['sentiment'] == 'Negative'])
            negative_percentage = (negative_sentiment_count / len(sentiment_risk_df) * 100) if len(sentiment_risk_df) > 0 else 0
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value sentiment-negative'>{negative_sentiment_count}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Negative Sentiment Posts ({negative_percentage:.1f}%)</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            geo_count = len(geocoded_df)
            geo_percentage = (geo_count / len(sentiment_risk_df) * 100) if len(sentiment_risk_df) > 0 else 0
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{geo_count}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Geolocated Posts ({geo_percentage:.1f}%)</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Main dashboard sections
        st.markdown("<h2 class='sub-header'>Sentiment & Risk Analysis</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            display_sentiment_plot()
        
        with col2:
            display_risk_plot()
        
        st.markdown("<h2 class='sub-header'>Sentiment vs Risk Level</h2>", unsafe_allow_html=True)
        display_sentiment_risk_plot()
        
        st.markdown("<h2 class='sub-header'>Geographic Distribution of High-Risk Posts</h2>", unsafe_allow_html=True)
        
        # Display the interactive heatmap
        folium_map = create_folium_heatmap(geocoded_df)
        if folium_map:
            folium_static(folium_map, width=1000, height=500)
        
        # Top 5 locations
        st.markdown("<h2 class='sub-header'>Top 5 Locations with High-Risk Posts</h2>", unsafe_allow_html=True)
        
        if top_locations:
            for i, loc in enumerate(top_locations[:5], 1):
                st.markdown(f"**{i}. {loc['location']}**: {loc['count']} posts")
    
    # Data Explorer Page
    elif page == "Data Explorer":
        st.markdown("<h1 class='main-header'>Data Explorer</h1>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Raw Data", "Cleaned Data", "Processed Data"])
        
        with tab1:
            st.markdown("<h3>Raw Social Media Data</h3>", unsafe_allow_html=True)
            if not raw_df.empty:
                st.dataframe(raw_df, height=400)
                
                # Basic statistics
                st.markdown("<h4>Data Overview</h4>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Posts", len(raw_df))
                    st.metric("Unique Subreddits", raw_df['subreddit'].nunique())
                
                with col2:
                    st.metric("Average Score", round(raw_df['score'].mean(), 2))
                    st.metric("Average Comments", round(raw_df['num_comments'].mean(), 2))
                
                # Distribution of posts by subreddit
                subreddit_counts = raw_df['subreddit'].value_counts().reset_index()
                subreddit_counts.columns = ['subreddit', 'count']
                
                fig = px.bar(
                    subreddit_counts,
                    x='subreddit',
                    y='count',
                    title='Distribution of Posts by Subreddit',
                    labels={'count': 'Number of Posts', 'subreddit': 'Subreddit'},
                    color='count',
                    color_continuous_scale='Blues'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution of posts by keyword
                keyword_counts = raw_df['keyword'].value_counts().reset_index()
                keyword_counts.columns = ['keyword', 'count']
                
                fig = px.bar(
                    keyword_counts,
                    x='keyword',
                    y='count',
                    title='Distribution of Posts by Keyword',
                    labels={'count': 'Number of Posts', 'keyword': 'Keyword'},
                    color='count',
                    color_continuous_scale='Greens'
                )
                
                fig.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.markdown("<h3>Cleaned Text Data</h3>", unsafe_allow_html=True)
            if not cleaned_df.empty:
                st.dataframe(cleaned_df, height=400)
                
                # Show example of text cleaning
                st.markdown("<h4>Text Cleaning Example</h4>", unsafe_allow_html=True)
                
                if len(raw_df) > 0 and len(cleaned_df) > 0:
                    # Get a random index
                    idx = np.random.randint(0, len(raw_df))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("<p style='font-weight: bold;'>Original Text:</p>", unsafe_allow_html=True)
                        st.markdown(f"<div class='card'>{raw_df.iloc[idx]['content']}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<p style='font-weight: bold;'>Cleaned Text:</p>", unsafe_allow_html=True)
                        st.markdown(f"<div class='card'>{cleaned_df.iloc[idx]['content']}</div>", unsafe_allow_html=True)
                    
                    if st.button("Show Another Example"):
                        st.experimental_rerun()
        
        with tab3:
            st.markdown("<h3>Processed Data with Sentiment & Risk Analysis</h3>", unsafe_allow_html=True)
            if not sentiment_risk_df.empty:
                st.dataframe(sentiment_risk_df, height=400)
                
                # Filtered views
                st.markdown("<h4>Filtered Views</h4>", unsafe_allow_html=True)
                
                filter_option = st.selectbox(
                    "Filter posts by:",
                    ["All Posts", "High-Risk Posts", "Negative Sentiment", "High-Risk & Negative Sentiment"]
                )
                
                if filter_option == "High-Risk Posts":
                    filtered_df = sentiment_risk_df[sentiment_risk_df['risk_level'] == 'High-Risk']
                elif filter_option == "Negative Sentiment":
                    filtered_df = sentiment_risk_df[sentiment_risk_df['sentiment'] == 'Negative']
                elif filter_option == "High-Risk & Negative Sentiment":
                    filtered_df = sentiment_risk_df[(sentiment_risk_df['risk_level'] == 'High-Risk') & 
                                                   (sentiment_risk_df['sentiment'] == 'Negative')]
                else:
                    filtered_df = sentiment_risk_df
                
                st.dataframe(filtered_df, height=300)
                st.metric("Number of Posts", len(filtered_df))
    
    # Sentiment Analysis Page
    elif page == "Sentiment Analysis":
        st.markdown("<h1 class='main-header'>Sentiment & Risk Analysis</h1>", unsafe_allow_html=True)
        
        # Sentiment Distribution
        st.markdown("<h2 class='sub-header'>Sentiment Distribution</h2>", unsafe_allow_html=True)
        display_sentiment_plot()
        
        # Risk Level Distribution
        st.markdown("<h2 class='sub-header'>Risk Level Distribution</h2>", unsafe_allow_html=True)
        display_risk_plot()
        
        # Sentiment vs Risk
        st.markdown("<h2 class='sub-header'>Sentiment vs Risk Level</h2>", unsafe_allow_html=True)
        display_sentiment_risk_plot()
        
        # Post Explorer
        st.markdown("<h2 class='sub-header'>Post Explorer</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                ["All", "Positive", "Neutral", "Negative"]
            )
        
        with col2:
            risk_filter = st.selectbox(
                "Filter by Risk Level",
                ["All", "High-Risk", "Moderate Concern", "Low Concern"]
            )
        
        # Apply filters
        filtered_df = sentiment_risk_df.copy()
        
        if sentiment_filter != "All":
            filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]
        
        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
        
        # Display filtered posts
        if not filtered_df.empty:
            for _, row in filtered_df.head(5).iterrows():
                sentiment_class = f"sentiment-{row['sentiment'].lower()}"
                risk_class = f"risk-{'low' if row['risk_level'] == 'Low Concern' else 'moderate' if row['risk_level'] == 'Moderate Concern' else 'high'}"
                
                st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Content:</b> {row['content']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Sentiment:</b> <span class='{sentiment_class}'>{row['sentiment']}</span></p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Risk Level:</b> <span class='{risk_class}'>{row['risk_level']}</span></p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            if len(filtered_df) > 5:
                st.markdown(f"*Showing 5 out of {len(filtered_df)} matching posts*")
        else:
            st.warning("No posts match the selected filters.")
    
    # Geospatial Analysis Page
    elif page == "Geospatial Analysis":
        st.markdown("<h1 class='main-header'>Geospatial Analysis</h1>", unsafe_allow_html=True)
        
        # Heatmap
        st.markdown("<h2 class='sub-header'>Crisis Heatmap</h2>", unsafe_allow_html=True)
        
        folium_map = create_folium_heatmap(geocoded_df)
        if folium_map:
            folium_static(folium_map, width=1000, height=500)
        
        # Choropleth
        st.markdown("<h2 class='sub-header'>Global Crisis Distribution</h2>", unsafe_allow_html=True)
        
        choropleth_fig = create_plotly_choropleth(geocoded_df)
        if choropleth_fig:
            st.plotly_chart(choropleth_fig, use_container_width=True)
        
        # Top Regions
        st.markdown("<h2 class='sub-header'>Top Regions with Crisis Posts</h2>", unsafe_allow_html=True)
        
        region_fig = create_region_bar_chart(geocoded_df)
        if region_fig:
            st.plotly_chart(region_fig, use_container_width=True)
        
        # Top Locations
        st.markdown("<h2 class='sub-header'>Top 10 Locations</h2>", unsafe_allow_html=True)
        display_location_chart()
        
        # Location details table
        if not location_breakdown_df.empty:
            st.dataframe(location_breakdown_df.head(10), height=300)
    
    # About the Project section
    elif page == "About the Project":
        st.markdown("<h1 class='main-header'>About the Project</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>Project Overview</h3>
            <p>This crisis monitoring dashboard was developed as part of the GSoC Candidate Assessment to extract, process, and analyze crisis-related discussions from social media. The project consists of three main components:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>1. Data Collection</h4>
                <p>Using the Reddit API, posts related to mental health distress, substance use, or suicidality were collected based on predefined keywords. The data was cleaned by removing stopwords, emojis, and special characters.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4>2. Sentiment & Risk Analysis</h4>
                <p>The VADER sentiment analysis tool was used to classify posts as Positive, Neutral, or Negative. TF-IDF was used to detect high-risk crisis terms and categorize posts into risk levels: High-Risk, Moderate Concern, and Low Concern.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4>3. Geolocation & Visualization</h4>
            <p>Location data was extracted from post content using NLP-based place recognition. The locations were geocoded and visualized on interactive maps to identify regional crisis patterns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display project structure
        st.markdown("<h3 class='sub-header'>Project Structure</h3>", unsafe_allow_html=True)
        
        st.code("""
    C:.
    |-- data/
    |   |-- cleaned_data.csv
    |   |-- high_risk.txt
    |   |-- keyword_dictionary.txt
    |   |-- moderate_risk.txt
    |   |-- non_location_words.txt
    |   |-- raw_data.csv
    |   |-- subreddits.txt
    |
    |-- results/
    |   |-- posts_with_sentiment_and_risk.csv
    |   |-- summary_statistics.csv
    |   |
    |   |-- geo_analysis/
    |   |   |-- crisis_choropleth.html
    |   |   |-- crisis_heatmap.html
    |   |   |-- geocoded_locations.csv
    |   |   |-- location_breakdown.csv
    |   |   |-- top_locations.json
    |   |   |-- top_locations_chart.png
    |   |
    |   |-- plots/
    |       |-- risk_level_distribution.png
    |       |-- sentiment_distribution.png
    |       |-- sentiment_vs_risk_stacked.png
    |
    |-- src/
    |   |-- crisis_geolocation.py
    |   |-- data_collection.py
    |   |-- sentiment_risk_analysis.py
    |
    |-- app.py (streamlit dashboard)
        """)
        
        # Display workflow
        st.markdown("<h3 class='sub-header'>Workflow</h3>", unsafe_allow_html=True)
        
        # Data Collection Command
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4>Data Collection Command:</h4>", unsafe_allow_html=True)
        st.code("python src/data_collection.py --keywords data/keyword_dictionary.txt --raw_output data/raw_data.csv --cleaned_output data/cleaned_data.csv --subreddits data/subreddits.txt --limit 12")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Sentiment Analysis Command
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4>Sentiment Analysis Command:</h4>", unsafe_allow_html=True)
        st.code("python src/sentiment_risk_analysis.py --input_file data/cleaned_data.csv --output_dir results --plot_dir results/plots --high_risk_terms data/high_risk.txt --moderate_risk_terms data/moderate_risk.txt --use_vader True --text_column content")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Geolocation Command
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4>Geolocation Command:</h4>", unsafe_allow_html=True)
        st.code("python src/crisis_geolocation.py --input_file results/posts_with_sentiment_and_risk.csv --output_dir results/geo_analysis --risk_level \"High-Risk\" --text_column content --non_location_file data/non_location_words.txt")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display technologies used
        st.markdown("<h3 class='sub-header'>Technologies Used</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="card" style="text-align: center;">
                <img src="https://www.python.org/static/community_logos/python-logo-generic.svg" width="100">
                <p><b>Python</b></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card" style="text-align: center;">
                <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.svg" width="100">
                <p><b>Streamlit</b></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="card" style="text-align: center;">
                <img src="https://nltk.org/nltk_data/images/nltk.png" width="100">
                <p><b>NLTK & VADER</b></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="card" style="text-align: center;">
                <img src="https://folium.readthedocs.io/en/latest/_static/folium_logo.jpg" width="100">
                <p><b>Folium & Plotly</b></p>
            </div>
            """, unsafe_allow_html=True)
            
    # About Me Page
    elif page == "About Me":
        st.markdown("<h1 class='main-header'>About Me</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image("https://img.icons8.com/color/240/000000/administrator-female.png", width=200)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h2>Oishika Das</h2>", unsafe_allow_html=True)
            st.markdown("<p><i>Applied Psychology Undergraduate & Mental Health Tech Enthusiast</i></p>", unsafe_allow_html=True)
            st.markdown("""
            <p>I am a 2nd year undergraduate student at Amity University, Kolkata pursuing Applied Psychology. 
            My passion lies at the intersection of psychology and technology, where I work to develop tools 
            that can identify and address mental health crises through data analysis.</p>
            
            <p>With hands-on experience as a crisis responder at Vandrevala Foundation, I bring real-world 
            understanding of immediate mental health interventions to my computational work.</p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Experience section
        st.markdown("<h3 class='sub-header'>Experience & Research</h3>", unsafe_allow_html=True)
        
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Crisis Intervention</h4>", unsafe_allow_html=True)
            st.markdown("""
            <p>I worked with Vandrevala Foundation, a major 24-hour crisis helpline in India, providing first responder 
            support to people in immediate mental health crisis situations. This experience gave me direct insight into 
            mental health emergencies and the critical importance of timely interventions.</p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with exp_col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Research Work</h4>", unsafe_allow_html=True)
            st.markdown("""
            <p>Currently working on a primary research project investigating smartphone addiction among college students. 
            I've presented my ongoing work at a research conference, exploring the psychological impacts of 
            technology dependence and potential intervention strategies.</p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Skills section
        st.markdown("<h3 class='sub-header'>Technical Skills</h3>", unsafe_allow_html=True)
        
        skills_col1, skills_col2, skills_col3 = st.columns(3)
        
        with skills_col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Data Analysis</h4>", unsafe_allow_html=True)
            st.markdown("""
            <ul>
                <li>Python Programming</li>
                <li>Pandas & NumPy</li>
                <li>Data Visualization</li>
                <li>Statistical Analysis</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with skills_col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>NLP & Text Analysis</h4>", unsafe_allow_html=True)
            st.markdown("""
            <ul>
                <li>roBERTa Models</li>
                <li>VADER & TextBlob</li>
                <li>NLTK</li>
                <li>Large Language Models</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with skills_col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Tools & Frameworks</h4>", unsafe_allow_html=True)
            st.markdown("""
            <ul>
                <li>Streamlit Dashboards</li>
                <li>Folium & Geopy</li>
                <li>Matplotlib & Seaborn</li>
                <li>BeautifulSoup & Requests</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Philosophy section
        st.markdown("<h3 class='sub-header'>My Vision</h3>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        <p>I have a deep passion for leveraging computational analysis to create technologies that benefit communities 
        and enhance people's lives in positive ways. My unique background combining psychology, crisis intervention, 
        and technical skills allows me to build bridges between mental health practice and technological solutions.</p>
        
        <p>This crisis monitoring dashboard represents my approach to creating tools that can help identify patterns 
        in mental health crises, potentially enabling earlier and more effective interventions.</p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Community section
        st.markdown("<h3 class='sub-header'>Community Involvement</h3>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        <p>Beyond my academic and technical work, I'm actively involved in community service and holistic practices:</p>
        
        <ul>
            <li><b>ISHA Foundation Volunteer:</b> Organize and manage flagship programs and contribute to their regional publishing department</li>
            <li><b>Hatha Yoga Instructor:</b> Serve as a foundation teacher, conducting independent Hatha Yoga programs in several cities across India</li>
        </ul>
        
        <p>These activities complement my technical work by grounding me in practices that support mental and physical wellbeing.</p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()