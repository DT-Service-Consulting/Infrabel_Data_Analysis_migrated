import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
import json
import pandas as pd
#import streamlit as st
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def visualize_departure_analysis(mean_delay_per_hour):
    """
    Visualizes the mean delay per departure hour in a bar plot.

    Args:
    mean_delay_per_hour (pd.Series): Series containing the mean delay per departure hour.
    """
    # Plot the mean delay per hour
    plt.figure(figsize=(10, 6))
    mean_delay_per_hour.plot(kind='bar')
    plt.title('Average Delay at Departure by Hour of the Day')
    plt.xlabel('Hour of Departure')
    plt.ylabel('Average Delay (minutes)')
    plt.xticks(rotation=90)
    plt.show()

def visualize_line_delays(avg_delays):
    """
    Creates an interactive bar plot of average delays for each departure line.

    Args:
    avg_delays (pd.DataFrame): The DataFrame containing average delays by departure line.
    """
    # Create the interactive plot
    fig = px.bar(
        avg_delays,
        x='Departure line',
        y=['Delay at arrival', 'Delay at departure'],
        barmode='group',
        title='Average Delays by Departure Line',
        labels={'value': 'Average Delay (in minutes)', 'Departure line': 'Departure Line'},
    )

    # Show the plot
    fig.show()
    #st.plotly_chart(fig)




def create_map(top_lines, top_10_places, places_coordinates, df_et, map_filename="belgium_etcs_map.html"):
    """
    Creates a Folium map visualizing the top 10 stopping places and lines with delays.

    Args:
    top_lines (DataFrame): DataFrame containing the top 10 lines with their delays.
    top_10_places (list): List of tuples containing the top 10 stopping places and their delays.
    places_coordinates (dict): A dictionary mapping stopping places to their coordinates.
    df_et (DataFrame): DataFrame containing ETCS line data.
    map_filename (str): The name of the output HTML file for the map.
    """
    # Color mapping based on ETCS levels
    color_map = {
        "ETCS L1 FS": "blue",
        "ETCS L2 FS": "orange",
        'ETCS L1 LS': "yellow",
        'ETCS 1+2': "green",
        'TVM-430' :"cyan"
    }

    # Create a Folium map centered around Belgium
    belgium_map = folium.Map(location=[50.8503, 4.3517], zoom_start=8)

    # Add circles for each place with delays
    for place, delay in top_10_places:
        folium.CircleMarker(
            location=places_coordinates[place],
            radius=delay * 3,  # Scale radius by delay
            popup=f'{place}: {delay:.2f} minutes delay',
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(belgium_map)

    # Plot lines on the map
    for index, entry in df_et.iterrows():
        # Extract coordinates from GeoShape
        coordinates = json.loads(entry["GeoShape"])["coordinates"][0]
        
        # Create a list of points for the polyline
        points = [(lat, lon) for lon, lat in coordinates]

        # Get the ETCS level and color
        etcs_level = entry["ETCS level"]
        color = color_map.get(etcs_level, "gray")  # Default to gray if not found

        # Create a polyline for the line
        folium.PolyLine(
            locations=points,
            color=color,
            weight=5,
            opacity=0.7,
            popup=f'Line: {entry["Line"]}, Track: {entry["Track"]}, ID: {entry["ID"]}'
        ).add_to(belgium_map)

        # Check if the line is in the top lines
        if entry["Line"] in top_lines["Departure line"].values:
            # Calculate midpoint of the coordinates
            mid_index = len(points) // 2
            midpoint = points[mid_index]

            # Add a marker at the midpoint
            folium.Marker(
                location=midpoint,
                popup=f'Top Line: {entry["Line"]}',
                icon=folium.Icon(color='red')
            ).add_to(belgium_map)

    # Save the map to an HTML file
    belgium_map.save(map_filename)


def visualize_delay_etcs(df_delay_etcs):
    """
    Visualizes the average delay at departure by line and ETCS status.

    Args:
    df_delay_etcs (DataFrame): DataFrame containing average delays and ETCS deployment status.
    """
    plt.figure(figsize=(25, 4))
    sns.scatterplot(x='Departure line', y='Avg Delay at Departure', hue='ETCS status', data=df_delay_etcs)
    plt.xticks(rotation=90)
    plt.title('Average Delay at Departure by Line and ETCS Status')
    plt.show()

def visualize_correlation(df_length):
    """
    Calculate and visualize the correlation matrix.

    Args:
    df_length (DataFrame): DataFrame containing lengths and delays.
    """
    correlation = df_length[['Total_Length_m', 'Avg Delay at Departure', 'Avg Delay at Arrival']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()
    print(correlation)


def plot_incident_counts(incident_counts):
    """
    Plot the counts of incidents.

    Args:
    incident_counts (Series): A series with incident types and their counts.
    """
    plt.figure(figsize=(15, 8))
    sns.barplot(x=incident_counts.values, y=incident_counts.index, palette='viridis')
    plt.title('Most Common Incident Types Causing Delays and Cancellations')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Incident Type')
    plt.show()


def plot_top_lines_with_most_incidents(line_incidents, top_n=10):
    """
    Plot the top N lines with the most incidents.

    Args:
    line_incidents (Series): A series with line numbers and their incident counts.
    top_n (int): Number of top lines to plot.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x=line_incidents.head(top_n).index, y=line_incidents.head(top_n).values, palette='viridis')
    plt.title(f'Top {top_n} Lines with Most Incidents')
    plt.xlabel('Line')
    plt.ylabel('Incident Count')
    plt.xticks(rotation=0)
    plt.show()
def create_map_complete(hotspots, places_coordinates, top_10_places, df_et, top_lines):
    """
    Create a Folium map highlighting incident hotspots.

    Args:
    hotspots (DataFrame): DataFrame containing hotspots.
    places_coordinates (dict): Dictionary with place names and their coordinates.
    top_10_places (list): List of top places with delays.
    df_et (DataFrame): DataFrame containing track geometries.
    top_lines (list): List of top lines to plot.
    """
    belgium_map = folium.Map(location=[50.8503, 4.3517], zoom_start=8)
    
    # Add circles for top 10 places with delays
    for place, delay in top_10_places:
        folium.CircleMarker(
            location=places_coordinates[place],
            radius=delay * 3,  # Scale radius by delay
            popup=f'{place}: {delay:.2f} minutes delay',
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(belgium_map)

    # Track the lines already plotted for incidents
    plotted_lines = set()

    # Plot lines on the map
    for index, entry in df_et.iterrows():
        # Extract coordinates from GeoShape
        coordinates = json.loads(entry["GeoShape"])["coordinates"][0]
        points = [(lat, lon) for lon, lat in coordinates]

        # Add polyline for the line
        folium.PolyLine(
            locations=points,
            color='gray',  # Default color for lines
            weight=5,
            opacity=0.7,
            popup=f'Line: {entry["Line"]}, Track: {entry["Track"]}, ID: {entry["ID"]}'
        ).add_to(belgium_map)

        # Add markers for hotspots
        if entry["Line"] in hotspots["Line"].values and entry["Line"] not in plotted_lines:
            r = hotspots.loc[hotspots.Line == entry["Line"]]
            mid_index = len(points) // 2
            midpoint = points[mid_index]

            folium.CircleMarker(
                location=midpoint,
                radius=int(r["Incident Count"]),
                popup=f'Hotspot: {entry["Line"]}',
                color='magenta',
                fill=True,
                fill_color='magenta'
            ).add_to(belgium_map)

            plotted_lines.add(entry["Line"])

    # Save the map to an HTML file
    belgium_map.save("belgium_incident_hotspots_map.html")

def plot_incident_heatmap(df_inc, top_n=15):
    """
    Plot a heatmap of incidents by line and type.

    Args:
    df_inc (DataFrame): DataFrame containing incident data.
    top_n (int): Number of top lines to consider for heatmap.
    """
    station_delay_counts = df_inc.groupby(['Line', 'Incident description.2']).size().unstack(fill_value=0)
    top_lines = station_delay_counts.sum(axis=1).nlargest(top_n).index
    top_line_delays = station_delay_counts.loc[top_lines]

    plt.figure(figsize=(10, 10))
    sns.heatmap(top_line_delays, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Number of Incidents'})
    plt.title('Heatmap of Primary Types of Delays at Key Lines')
    plt.xlabel('Incident Description')
    plt.ylabel('Line')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.show()

def analyze_punctuality_trends(df_inc_ects):
    """
    Analyze trends in punctuality over time based on ETCS levels.
    
    Args:
    df_inc_ects (DataFrame): Merged DataFrame with ETCS levels.
    """
    # Convert 'Incident date' to datetime format
    df_inc_ects['Incident date'] = pd.to_datetime(df_inc_ects['Incident date'])

    # Extract month-year for trend analysis
    df_inc_ects['YearMonth'] = df_inc_ects['Incident date'].dt.to_period('M')

    # Create a new column to classify lines as ETCS or Non-ETCS
    df_inc_ects['ETCS Status'] = df_inc_ects['ETCS level'].apply(
        lambda x: 'Non-ETCS' if x == "No ETCS" else 'ETCS'
    )

    # Grouping by Year-Month and ETCS Status
    trend_data = df_inc_ects.groupby(['YearMonth', 'ETCS Status']).agg({
        'Minutes of delay': 'mean',
        'Number of cancelled trains': 'mean'
    }).reset_index()

    # Convert 'YearMonth' back to a datetime format for plotting
    trend_data['YearMonth'] = trend_data['YearMonth'].dt.to_timestamp()

    # Create combined plots for Average Delay and Average Cancellations
    plt.figure(figsize=(12, 6))

    # Define winter months (December, January, February)
    winter_months = trend_data['YearMonth'].dt.month.isin([12, 1, 2])

    # Subplot for Average Delay
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
    sns.lineplot(data=trend_data, x='YearMonth', y='Minutes of delay', hue='ETCS Status', marker='o')
    plt.title('Trend of Average Delay Over Time: ETCS vs Non-ETCS Lines')
    plt.ylabel('Average Delay (Minutes)')
    plt.xlabel('Date (Year-Month)')
    plt.xticks(rotation=45)
    plt.legend(title='ETCS Status')

    # Highlight winter months by shading the background
    for i in range(len(trend_data)):
        if winter_months[i]:
            plt.axvspan(trend_data['YearMonth'].iloc[i], trend_data['YearMonth'].iloc[i], color='lightblue', alpha=0.3)

    plt.tight_layout()

    # Subplot for Average Cancellations
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
    sns.lineplot(data=trend_data, x='YearMonth', y='Number of cancelled trains', hue='ETCS Status', marker='o')
    plt.title('Trend of Average Cancellations Over Time: ETCS vs Non-ETCS Lines')
    plt.ylabel('Average Number of Cancellations')
    plt.xlabel('Date (Year-Month)')
    plt.xticks(rotation=45)
    plt.legend(title='ETCS Status')

    # Highlight winter months by shading the background
    for i in range(len(trend_data)):
        if winter_months[i]:
            plt.axvspan(trend_data['YearMonth'].iloc[i], trend_data['YearMonth'].iloc[i], color='lightblue', alpha=0.7)

    plt.tight_layout()
    plt.show()


# Function to plot feature importance
def plot_feature_importance(pipeline):
    """
    Plot the top 10 important features based on the RandomForest model.

    Args:
    pipeline (Pipeline): Trained model pipeline.
    """
    model = pipeline.named_steps['model']
    encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']

    # Get feature names from the OneHotEncoder and numeric features
    encoded_features = encoder.get_feature_names_out(['Line_y', 'Place.2', 'Incident description.2', 'ETCS level'])
    numeric_features = ['Total_Length_m']
    all_feature_names = numeric_features + list(encoded_features)

    # Get feature importances
    importances = model.feature_importances_

    # Create DataFrame and sort by importance
    feature_importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    # Display the top 10 important features
    print(feature_importance_df.head(10))

    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 10 Important Features')
    plt.gca().invert_yaxis()  # Invert y-axis to show top feature at the top
    plt.show()