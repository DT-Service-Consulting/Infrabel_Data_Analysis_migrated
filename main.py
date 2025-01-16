#import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Set the aesthetic style of the plots
sns.set(style="whitegrid")
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from shapely.geometry import LineString, Point
import ast
import geopandas as gpd
import json
import folium
import contextily as ctx
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from preprocess import *
from visualization import *
from analysis import *

def main():
#Preprocessing the data

    df_reg = preprocess_reg_data('Data\Regular data.csv')
    df_inc = preprocess_inc_data('Data\Incidents.csv')
    df_et = preprocess_etcs_data('Data\ETCS.csv')

    #Exploratory Data Analysis

    #1. Is there any relation between hours of departure and delay?
    mean_delay_per_hour, correlation= analyze_departure_data(df_reg)
    visualize_departure_analysis(mean_delay_per_hour)

    #2. Is there any relation between the departure line and delays?
    avg_delays = analyze_line_delays(df_reg)
    visualize_line_delays(avg_delays)

    #3. Are the stopping places with highest delays corresponds to "lines" with highest delays?
    top_delays = analyze_stopping_place_delays(df_reg)
    places_coordinates, top_10_places = preprocess_top_10_places()
    top_lines = analyze_top_lines_with_delays(df_reg)
    create_map(top_lines, top_10_places, places_coordinates, df_et, map_filename="belgium_etcs_map.html")

    #4.Has the ETCS deployments improved punctuality?
    df_delay_etcs = preprocess_delay_data(df_reg, df_et)
    df_delay_etcs = analyze_etcs_status(df_delay_etcs)
    visualize_delay_etcs(df_delay_etcs)
    etcs_delays, no_etcs_delays = separate_delays_by_etcs_status(df_delay_etcs)
    u_stat, p_value = perform_mann_whitney_u_test(etcs_delays, no_etcs_delays)
    interpret_results(u_stat, p_value, alpha=0.05)

    #5. Is there any relation with the length of the tracks to the delay minutes?
    gdf_tracks = create_geo_dataframe(df_et)
    gdf_tracks = calculate_track_lengths(gdf_tracks)# Step 2: Calculate track lengths
    print(gdf_tracks[['ID', 'ETCS level', 'Line', 'Track', 'Length', 'Length_m']])# Step 3: Display the lengths
    df_dist = aggregate_track_lengths(gdf_tracks)# Step 4: Aggregate lengths
    print(df_dist)# Step 5: Display the resulting DataFrame
    df_length = merge_with_delay_data(df_delay_etcs, df_dist)# Step 6: Merge with delay data
    visualize_correlation(df_length) # Step 7: Visualize correlation

    #6.  What are the most common types of incidents causing delays and cancellations? Which incidents lead to more severe disruptions?
    incident_counts = count_incident_occurrences(df_inc)# Step 1: Count incident occurrences
    plot_incident_counts(incident_counts)# Step 2: Plot the incident counts

    #7.What are the top lines with most incidents?
    line_incidents = count_incidents_by_line(df_inc)#Step 1: Count incidents by line
    plot_top_lines_with_most_incidents(line_incidents, top_n=10)# Step 2: Plot the top 10 lines with the most incidents

    #8.Identifying the incident hotspots
    incident_summary = summarize_incidents(df_inc)
    hotspots = identify_hotspots(incident_summary)# Step 2: Identify hotspots
    places_coordinates, top_10_places = preprocess_top_10_places()
    top_lines = top_lines
    create_map_complete(hotspots, places_coordinates, top_10_places, df_et, top_lines)
    plot_incident_heatmap(df_inc)

    #9.Does ECTS helps in the recovery time after an accident?
    df_inc_ects = merge_datasets(df_inc, df_length)
    incident_line_counts = analyze_incidents_by_etcs(df_inc_ects)# Step 2: Analyze incidents by ETCS
    etcs_delay = analyze_delays_by_etcs(df_inc_ects)# Step 3: Analyze delays by ETCS
    incident_types_for_specific_etcs(df_inc_ects)# Step 4: Analyze incident types for specific ETCS levels
    analyze_recovery_time(df_inc_ects)# Step 5: Analyze recovery time
    analyze_top_incident_types(df_inc_ects)# Step 6: Analyze top incident types

    # 10.Does the ECTS helps improving punctuality over time?
    analyze_punctuality_trends(df_inc_ects)

    #Predictive Modelling
    X, y = preprocess_data(df_inc_ects)
    preprocessor = create_preprocessing_pipeline() 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = train_model(X_train, y_train, preprocessor)
    mse, r2 = evaluate_model(pipeline, X_test, y_test)
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R-squared: {r2:.2f}')
    plot_feature_importance(pipeline)

if __name__ == '__main__':
    main()

