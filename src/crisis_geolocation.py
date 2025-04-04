#!/usr/bin/env python3

"""
Script Name: crisis_geolocation.py

This script performs geolocation analysis on crisis-related social media posts.
It extracts location mentions from post text, geocodes them to coordinates,
and generates interactive visualizations including heatmaps and location breakdowns.

Usage:
  python crisis_geolocation.py \
    --input_file INPUT_FILE_PATH \
    --output_dir OUTPUT_DIRECTORY \
    --risk_level RISK_LEVEL \
    --text_column COLUMN_NAME \
    --non_location_file NON_LOCATION_FILE

Example:
  python crisis_geolocation.py \
    --input_file results/posts_with_sentiment_and_risk.csv \
    --output_dir results/geo_analysis \
    --risk_level "High-Risk" \
    --text_column content \
    --non_location_file non_location_words.txt
"""

import argparse
import os
import pandas as pd
import numpy as np
import re
import spacy
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import time

# Try to load spaCy's English model
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    print("Warning: spaCy model not available. Using regex-based location extraction.")
    SPACY_AVAILABLE = False

# Cache for geocoded locations to avoid duplicate API calls
geocoding_cache = {}

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Extract location data and generate crisis heatmaps.")
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True,
        default="results/posts_with_sentiment_and_risk.csv", 
        help="Path to the input CSV file containing posts with sentiment and risk."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        default="results/geo_analysis",
        help="Directory to store the output files and visualizations."
    )
    parser.add_argument(
        "--risk_level", 
        type=str, 
        default="High-Risk", 
        help="Risk level to filter for geolocation (e.g., 'High-Risk', 'Moderate Concern')."
    )
    parser.add_argument(
        "--text_column", 
        type=str, 
        default="content", 
        help="Name of the column containing the text data."
    )
    parser.add_argument(
        "--non_location_file", 
        type=str, 
        default="non_location_words.txt", 
        help="Path to a file containing non-location words to filter out."
    )
    return parser.parse_args()


def extract_locations_spacy(text):
    """
    Extracts location entities from text using spaCy's named entity recognition.
    Returns a list of location strings.
    """
    if not isinstance(text, str) or not text:
        return []
    
    doc = nlp(text)
    locations = []
    
    # Extract entities labeled as GPE (Geopolitical Entity) or LOC (Location)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            locations.append(ent.text.lower())
    
    return locations


def extract_locations_regex(text):
    """
    Extracts potential location mentions using regex patterns.
    This is a fallback when spaCy is not available.
    """
    if not isinstance(text, str) or not text:
        return []
    
    locations = []
    text = text.lower()
    
    # Check for common location phrases with prepositions
    location_patterns = [
        r"in ([a-zA-Z\s]+?,?\s?[a-zA-Z]*)",
        r"at ([a-zA-Z\s]+?,?\s?[a-zA-Z]*)",
        r"from ([a-zA-Z\s]+?,?\s?[a-zA-Z]*)",
        r"near ([a-zA-Z\s]+?,?\s?[a-zA-Z]*)",
        r"around ([a-zA-Z\s]+?,?\s?[a-zA-Z]*)",
        r"living in ([a-zA-Z\s]+?,?\s?[a-zA-Z]*)",
        r"moved to ([a-zA-Z\s]+?,?\s?[a-zA-Z]*)",
        r"based in ([a-zA-Z\s]+?,?\s?[a-zA-Z]*)"
    ]
    
    for pattern in location_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Clean up the extracted location
            location = match.strip()
            if location and len(location) > 2:  # Avoid very short matches
                if location not in ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'our', 'their']:
                    locations.append(location)
    
    # Check for common location indicators
    country_city_indicators = [
        r"city of ([a-zA-Z\s]+)",
        r"town of ([a-zA-Z\s]+)",
        r"([a-zA-Z\s]+) city",
        r"([a-zA-Z\s]+) county",
        r"([a-zA-Z\s]+) province",
        r"province of ([a-zA-Z\s]+)",
        r"region of ([a-zA-Z\s]+)"
    ]
    
    for pattern in country_city_indicators:
        matches = re.findall(pattern, text)
        for match in matches:
            location = match.strip()
            if location and len(location) > 2:
                locations.append(location)
    
    return list(set(locations))  # Remove duplicates


def geocode_location(location_text, geolocator):
    """
    Geocodes a location string to coordinates using the provided geolocator.
    Uses a cache to avoid redundant API calls.
    Returns a tuple of (latitude, longitude, full_location_name) or None if geocoding fails.
    """
    global geocoding_cache
    
    # First check the cache
    if location_text.lower() in geocoding_cache:
        return geocoding_cache[location_text.lower()]
    
    try:
        # Try the geocoding service
        location = geolocator(location_text)
        
        if location and hasattr(location, 'latitude') and hasattr(location, 'longitude'):
            result = (location.latitude, location.longitude, location.address)
            # Store in cache
            geocoding_cache[location_text.lower()] = result
            return result
    except Exception as e:
        print(f"Error geocoding '{location_text}': {e}")
    
    # If geocoding failed, store None in cache to avoid retrying
    geocoding_cache[location_text.lower()] = None
    return None


def extract_and_geocode_locations(df, text_column, risk_level):
    """
    Extracts locations from the specified text column and geocodes them.
    Returns a DataFrame with location information.
    """
    # Filter for posts with the specified risk level
    if risk_level and 'risk_level' in df.columns:
        risk_posts = df[df['risk_level'] == risk_level].copy()
    else:
        risk_posts = df.copy()
    
    print(f"Extracting locations from {len(risk_posts)} posts with risk level '{risk_level}'")
    
    # Set up the geocoder with Nominatim (no API key required)
    geolocator = Nominatim(user_agent="crisis_geolocation_script")
    # Use a rate limiter to avoid hitting API limits (1 second between requests for Nominatim)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    # Extract locations
    if SPACY_AVAILABLE:
        print("Using spaCy for location extraction")
        extract_func = extract_locations_spacy
    else:
        print("Using regex for location extraction")
        extract_func = extract_locations_regex
    
    # Apply location extraction to each post
    tqdm.pandas(desc="Extracting locations")
    risk_posts['extracted_locations'] = risk_posts[text_column].progress_apply(extract_func)
    
    # Create a new dataframe with one row per location mention
    location_rows = []
    
    for idx, row in tqdm(risk_posts.iterrows(), total=len(risk_posts), desc="Geocoding locations"):
        for location in row['extracted_locations']:
            # Skip very short or generic location mentions
            if len(location) < 3 or location in ['here', 'there', 'anywhere', 'nowhere']:
                continue
                
            # Try to geocode
            geo_result = geocode_location(location, geocode)
            
            if geo_result:
                lat, lon, full_location = geo_result
                location_rows.append({
                    'post_id': row.get('id', idx),
                    'sentiment': row.get('sentiment', 'Unknown'),
                    'risk_level': row.get('risk_level', risk_level) if 'risk_level' in row else risk_level,
                    'extracted_location': location,
                    'full_location': full_location,
                    'latitude': lat,
                    'longitude': lon
                })
            
            # Add a small delay to avoid overwhelming the API
            time.sleep(0.2)
    
    # Create a dataframe from the location rows
    locations_df = pd.DataFrame(location_rows)
    
    # Log the results
    if len(locations_df) > 0:
        print(f"Successfully geocoded {len(locations_df)} locations from {len(risk_posts)} posts")
        print(f"Found locations in {len(locations_df['post_id'].unique())} unique posts")
    else:
        print("No locations could be geocoded.")
    
    return locations_df


def filter_non_locations(locations_df, non_location_file):
    """
    Filters out non-location words from the location results.
    This function should be called after the main location extraction is complete.
    """
    print(f"Filtering out non-location words from {non_location_file}...")
    
    # Load the non-location words
    try:
        with open(non_location_file, 'r', encoding='utf-8') as f:
            non_location_words = [line.strip().lower() for line in f if line.strip()]
        print(f"Loaded {len(non_location_words)} non-location words")
    except FileNotFoundError:
        print(f"Warning: Non-location words file '{non_location_file}' not found. Skipping filtering.")
        return locations_df
    
    # Number of locations before filtering
    original_count = len(locations_df)
    
    # Filter out non-location words
    filtered_df = locations_df.copy()
    for word in non_location_words:
        # Filter out exact matches in the 'extracted_location' column
        filtered_df = filtered_df[~filtered_df['extracted_location'].str.lower().eq(word)]
        
        # Also filter out locations that contain these words
        filtered_df = filtered_df[~filtered_df['extracted_location'].str.lower().str.contains(r'\b' + word + r'\b')]
    
    # Number of locations after filtering
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count
    
    print(f"Removed {removed_count} non-location entries ({removed_count/original_count*100:.1f}% of total)")
    
    return filtered_df


def create_folium_heatmap(locations_df, output_path):
    """
    Creates a Folium heatmap visualization of location data.
    Saves the map to the specified output path.
    """
    if len(locations_df) == 0:
        print("No location data available for heatmap.")
        return
    
    # Center the map on the mean coordinates
    center_lat = locations_df['latitude'].mean()
    center_lon = locations_df['longitude'].mean()
    
    # Create a base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles='CartoDB positron')
    
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
    
    # Save the map
    m.save(output_path)
    print(f"Heatmap saved to {output_path}")
    
    return m


def create_plotly_choropleth(locations_df, output_path):
    """
    Creates a Plotly choropleth map of location data aggregated by country/region.
    Saves the map to the specified output path.
    """
    if len(locations_df) == 0:
        print("No location data available for choropleth map.")
        return
    
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
    
    # Add country column
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
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Save the figure
    fig.write_html(output_path)
    print(f"World choropleth map saved to {output_path}")
    
    # Create a more detailed regional breakdown
    # First, get regions from country and city info
    locations_df['region'] = locations_df['full_location'].apply(
        lambda loc: loc.split(',')[0].strip() if isinstance(loc, str) and ',' in loc else loc
    )
    
    # Count by region
    region_counts = locations_df['region'].value_counts().reset_index().head(15)
    region_counts.columns = ['region', 'count']
    
    # Create a bar chart of top regions
    fig_bar = px.bar(
        region_counts,
        x='region', 
        y='count',
        title=f'Top 15 Regions with Crisis Posts ({locations_df["risk_level"].iloc[0] if "risk_level" in locations_df.columns else "All Posts"})',
        labels={'count': 'Number of Posts', 'region': 'Region'},
        color='count',
        color_continuous_scale='Reds'
    )
    
    region_bar_path = os.path.splitext(output_path)[0] + '_top_regions.html'
    fig_bar.write_html(region_bar_path)
    print(f"Top regions chart saved to {region_bar_path}")
    
    return fig


def create_location_breakdown(locations_df, output_dir):
    """
    Creates visualizations showing the breakdown of top locations.
    Saves the visualizations to the specified output directory.
    """
    if len(locations_df) == 0:
        print("No location data available for location breakdown.")
        return
    
    # Count occurrences of each location
    location_counts = locations_df['full_location'].value_counts().reset_index()
    location_counts.columns = ['location', 'count']
    
    # Get the top 5 locations (plus a few more for the visualization)
    top_locations = location_counts.head(10)
    
    # Create a bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.barh(top_locations['location'], top_locations['count'], color='darkred')
    plt.xlabel('Number of Crisis Posts')
    plt.ylabel('Location')
    plt.title(f'Top 10 Locations with Highest Crisis Discussions ({locations_df["risk_level"].iloc[0] if "risk_level" in locations_df.columns else "All Posts"})')
    plt.tight_layout()
    
    # Add count labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width}', 
                 ha='left', va='center')
    
    # Save the chart
    chart_path = os.path.join(output_dir, 'top_locations_chart.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Top locations chart saved to {chart_path}")
    
    # Create a detailed CSV with all locations
    csv_path = os.path.join(output_dir, 'location_breakdown.csv')
    location_counts.to_csv(csv_path, index=False)
    print(f"Location breakdown saved to {csv_path}")
    
    # Create a JSON file with the top locations including coordinates
    top_locations_with_coords = []
    for _, location_row in top_locations.iterrows():
        location_name = location_row['location']
        # Get coordinate information for this location
        location_info = locations_df[locations_df['full_location'] == location_name].iloc[0]
        top_locations_with_coords.append({
            'location': location_name,
            'count': int(location_row['count']),
            'latitude': float(location_info['latitude']),
            'longitude': float(location_info['longitude']),
            'risk_level': location_info['risk_level']
        })
    
    json_path = os.path.join(output_dir, 'top_locations.json')
    with open(json_path, 'w') as f:
        json.dump(top_locations_with_coords, f, indent=2)
    print(f"Top locations JSON saved to {json_path}")
    
    # Print the top 5 locations
    print("\nTop 5 locations with highest crisis discussions:")
    for i, loc in enumerate(top_locations_with_coords[:5], 1):
        print(f"{i}. {loc['location']}: {loc['count']} posts")
    
    return top_locations_with_coords


def main():
    """
    Main function that orchestrates the geolocation and mapping process.
    """
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Reading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    if args.text_column not in df.columns:
        raise ValueError(
            f"Text column '{args.text_column}' not found in input file. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    if 'risk_level' not in df.columns:
        print("Warning: No 'risk_level' column found. Processing all posts.")
        risk_level = None
    elif args.risk_level not in df['risk_level'].unique():
        print(f"Warning: Risk level '{args.risk_level}' not found in data. Available levels: {df['risk_level'].unique()}")
        risk_level = None
    else:
        risk_level = args.risk_level
    
    # Extract and geocode locations
    locations_df = extract_and_geocode_locations(df, args.text_column, risk_level)
    
    # Filter out non-location words if the file exists
    locations_df = filter_non_locations(locations_df, args.non_location_file)
    
    # Save the locations dataframe
    locations_csv_path = os.path.join(args.output_dir, 'geocoded_locations.csv')
    locations_df.to_csv(locations_csv_path, index=False)
    print(f"Geocoded locations saved to {locations_csv_path}")
    
    # Create visualizations
    if len(locations_df) > 0:
        # Folium heatmap
        heatmap_path = os.path.join(args.output_dir, 'crisis_heatmap.html')
        create_folium_heatmap(locations_df, heatmap_path)
        
        # Plotly choropleth
        choropleth_path = os.path.join(args.output_dir, 'crisis_choropleth.html')
        create_plotly_choropleth(locations_df, choropleth_path)
        
        # Location breakdown
        top_locations = create_location_breakdown(locations_df, args.output_dir)
    else:
        print("No locations could be geocoded. Unable to create visualizations.")
    
    print("\nProcess completed successfully.")


if __name__ == "__main__":
    main()