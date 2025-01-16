import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def analyze_line_delays(df):
    """
    Analyzes the average delays by grouping data by departure line.

    Args:
    df (pd.DataFrame): The input DataFrame containing delay information.

    Returns:
    pd.DataFrame: A DataFrame containing the average delays for each departure line.
    """
    # Group by Departure Line and Calculate Average Delays
    avg_delays = df.groupby('Departure line')[['Delay at arrival', 'Delay at departure']].mean().reset_index()

    return avg_delays


def analyze_stopping_place_delays(df):
    """
    Analyzes the delays by grouping the data by stopping places and calculating average delays.

    Args:
    df (pd.DataFrame): The input DataFrame containing delay information.

    Returns:
    pd.DataFrame: A DataFrame containing the top 10 stopping places with the highest average delays.
    """
    # Group by stopping place and calculate average delays
    avg_delays = df.groupby('Stopping place').agg(
        avg_delay_departure=('Delay at departure', 'mean'),
        avg_delay_arrival=('Delay at arrival', 'mean')
    ).reset_index()

    # Sort by average departure delay and select the top 10
    top_delays = avg_delays.sort_values(by='avg_delay_departure', ascending=False).head(10)

    return top_delays


def analyze_top_lines_with_delays(df):
    """
    Analyzes the average delays for departure lines and returns the top 10 lines.

    Args:
    df (DataFrame): DataFrame containing train delay information.

    Returns:
    DataFrame: A DataFrame containing the top 10 departure lines with the highest average delays.
    """
    # Group by Departure Line and calculate average delays
    avg_line_delays = df.groupby('Departure line').agg(
        avg_delay_departure=('Delay at departure', 'mean'),
        avg_delay_arrival=('Delay at arrival', 'mean'),
        corresponding_stopping_places=('Stopping place', lambda x: ', '.join(set(x)))  # List unique stopping places
    ).reset_index()

    # Sort by average departure delay and select the top 10
    top_lines = avg_line_delays.sort_values(by='avg_delay_departure', ascending=False).head(10)

    # Print the top 10 departure lines with the highest average delays
    print("Top 10 Departure Lines with Highest Average Departure Delays:")
    print(top_lines[['Departure line', 'avg_delay_departure', 'corresponding_stopping_places']])

    return top_lines


def analyze_etcs_status(df_delay_etcs):
    """
    Adds ETCS status to the DataFrame.

    Args:
    df_delay_etcs (DataFrame): DataFrame containing average delays and ETCS deployment status.

    Returns:
    DataFrame: The updated DataFrame with ETCS status.
    """
    # Creating a column with "Status" to know if having ETCS or not affects the delay timings
    df_delay_etcs['ETCS status'] = df_delay_etcs['ETCS level'].apply(lambda x: 'No ETCS' if x == "No ETCS" else 'ETCS')
    
    return df_delay_etcs

def separate_delays_by_etcs_status(df_delay_etcs):
    """
    Separates average delays based on ETCS status.

    Args:
    df_delay_etcs (DataFrame): DataFrame containing average delays and ETCS deployment status.

    Returns:
    tuple: Two Series containing delays for ETCS and No ETCS.
    """
    etcs_delays = df_delay_etcs[df_delay_etcs['ETCS status'] == 'ETCS']['Avg Delay at Departure']
    no_etcs_delays = df_delay_etcs[df_delay_etcs['ETCS status'] == 'No ETCS']['Avg Delay at Departure']
    
    return etcs_delays, no_etcs_delays



def perform_mann_whitney_u_test(etcs_delays, no_etcs_delays):
    """
    Performs Mann-Whitney U Test on the provided delays.

    Args:
    etcs_delays (Series): Delays with ETCS.
    no_etcs_delays (Series): Delays without ETCS.

    Returns:
    tuple: U-statistic and p-value from the test.
    """
    u_stat, p_value = stats.mannwhitneyu(etcs_delays, no_etcs_delays, alternative='two-sided')
    return u_stat, p_value

def interpret_results(u_stat, p_value, alpha=0.05):
    """
    Interprets the results of the Mann-Whitney U Test.

    Args:
    u_stat (float): U-statistic from the test.
    p_value (float): P-value from the test.
    alpha (float): Significance level for the test.

    Returns:
    str: Interpretation of the test results.
    """
    print(f"U-statistic: {u_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < alpha:
        print( "Reject the null hypothesis: There is a significant difference in departure delays.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference in departure delays.")


def calculate_track_lengths(gdf_tracks):
    """
    Calculate lengths of the tracks in meters.

    Args:
    gdf_tracks (GeoDataFrame): GeoDataFrame containing track geometries.

    Returns:
    GeoDataFrame: GeoDataFrame with lengths in meters.
    """
    gdf_tracks['Length'] = gdf_tracks.length  # Length in degrees (geodetic)
    gdf_tracks = gdf_tracks.to_crs(epsg=3857)  # Convert to Web Mercator (meters)
    gdf_tracks['Length_m'] = gdf_tracks.length  # Length in meters
    return gdf_tracks

def aggregate_track_lengths(gdf_tracks):
    """
    Group by lines and calculate total track lengths.

    Args:
    gdf_tracks (GeoDataFrame): GeoDataFrame with track lengths.

    Returns:
    DataFrame: DataFrame with total lengths for each line.
    """
    df_dist = gdf_tracks.groupby('Line').agg({
        'Length_m': 'sum'  # Sum the lengths of tracks in meters
    }).reset_index()

    # Rename the columns for clarity
    df_dist.rename(columns={'Length_m': 'Total_Length_m'}, inplace=True)
    return df_dist

def merge_with_delay_data(df_delay_etcs, df_dist):
    """
    Merge the total lengths with the delay data.

    Args:
    df_delay_etcs (DataFrame): DataFrame with delay information.
    df_dist (DataFrame): DataFrame with total lengths.

    Returns:
    DataFrame: Merged DataFrame with lengths and delays.
    """
    df_dist['Line'] = df_dist['Line'].replace({
        '01': '0/1',
        '02': '0/2',
        '03': '0/3'
    })
    return pd.merge(df_delay_etcs, df_dist, on='Line', how='left')


def count_incident_occurrences(df_inc):
    """
    Count occurrences of each incident type.

    Args:
    df_inc (DataFrame): DataFrame containing incident descriptions.

    Returns:
    Series: A series with incident types as index and their counts as values.
    """
    return df_inc['Incident description.2'].value_counts()


def count_incidents_by_line(df_inc):
    """
    Count the number of incidents for each line.

    Args:
    df_inc (DataFrame): DataFrame containing incident data.

    Returns:
    Series: A series with lines as index and their incident counts as values.
    """
    return df_inc['Line'].value_counts()




def summarize_incidents(df_inc):
    """
    Summarize incident data by line.

    Args:
    df_inc (DataFrame): DataFrame containing incident data.

    Returns:
    DataFrame: Summary of incidents grouped by line.
    """
    incident_summary = df_inc.groupby("Line").agg({
        "Incident date": "count",  # Count incidents
        "Minutes of delay": "mean",  # Mean delays
        "Number of cancelled trains": "sum"  # Sum cancellations
    }).rename(columns={"Incident date": "Incident Count"})
    
    return incident_summary.reset_index()

def identify_hotspots(incident_summary, incident_threshold=10):
    """
    Identify hotspots based on incident counts.

    Args:
    incident_summary (DataFrame): Summary of incidents by line.
    incident_threshold (int): Minimum number of incidents to qualify as a hotspot.

    Returns:
    DataFrame: Hotspots with counts exceeding the threshold.
    """
    hotspots = incident_summary[incident_summary["Incident Count"] > incident_threshold]
    return hotspots.reset_index()


def merge_datasets(df_inc, df_length):
    """
    Merge incident and ETCS datasets and fill missing ETCS levels.
    
    Args:
    df_inc (DataFrame): DataFrame containing incident data.
    df_length (DataFrame): DataFrame containing ETCS level information.
    
    Returns:
    DataFrame: Merged DataFrame with ETCS levels.
    """
    df_inc_ects = pd.merge(df_inc, df_length, left_on="Line", right_on="Departure line", how='left')
    df_inc_ects["ETCS level"] = df_inc_ects["ETCS level"].fillna("No ETCS")
    return df_inc_ects

def analyze_incidents_by_etcs(df_inc_ects):
    """
    Analyze incidents based on ETCS levels.
    
    Args:
    df_inc_ects (DataFrame): Merged DataFrame with ETCS levels.
    
    Returns:
    DataFrame: Incident counts and line counts by ETCS level.
    """
    # Group by 'ETCS level' and count the total number of incidents
    incident_counts = df_inc_ects.groupby('ETCS level').size().reset_index(name='Total Incidents')

    # Count the number of unique lines for each ETCS level
    line_counts = df_inc_ects.groupby('ETCS level')['Line_x'].nunique().reset_index(name='Number of Lines')

    # Merge the two dataframes to get incidents and lines together
    incident_line_counts = pd.merge(incident_counts, line_counts, on='ETCS level')

    # Calculate the weighted number of incidents per line
    incident_line_counts['Weighted Incidents per Line'] = incident_line_counts['Total Incidents'] / incident_line_counts['Number of Lines']

    # Plot the weighted number of incidents per line by ETCS level
    plt.figure(figsize=(15, 6))
    sns.barplot(data=incident_line_counts, x='ETCS level', y='Weighted Incidents per Line', order=incident_line_counts['ETCS level'].value_counts().index)
    plt.title('Weighted Number of Incidents per Line by ETCS Level')
    plt.xticks(rotation=90)
    plt.xlabel('ETCS Level')
    plt.ylabel('Weighted Incidents per Line')
    plt.show()
    
    return incident_line_counts

def analyze_delays_by_etcs(df_inc_ects):
    """
    Analyze delays based on ETCS levels.
    
    Args:
    df_inc_ects (DataFrame): Merged DataFrame with ETCS levels.
    
    Returns:
    DataFrame: Average and total delays by ETCS level.
    """
    # Group by ETCS level and calculate average delay
    etcs_delay = df_inc_ects.groupby('ETCS level')['Minutes of delay'].agg(['mean', 'sum', 'count']).reset_index()

    # Rename columns
    etcs_delay.columns = ['ETCS Level', 'Average Delay (Minutes)', 'Total Delay (Minutes)', 'Incident Count']

    # Sort values by Average Delay for visualization
    etcs_delay = etcs_delay.sort_values(by='Average Delay (Minutes)', ascending=False)

    # Plot the average delay by ETCS level
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Average Delay (Minutes)', y='ETCS Level', data=etcs_delay, palette='viridis')
    plt.title('Average Delay by ETCS Level')
    plt.axvline(etcs_delay['Average Delay (Minutes)'].mean(), color='red', linestyle='--', label='Overall Average')
    plt.legend()
    plt.show()
    
    return etcs_delay

def incident_types_for_specific_etcs(df_inc_ects):
    """
    Analyze incident types for specific ETCS levels.
    
    Args:
    df_inc_ects (DataFrame): Merged DataFrame with ETCS levels.
    """
    # Filter for specific ETCS levels
    etcs_l1_fs_incidents = df_inc_ects[df_inc_ects['ETCS level'].str.contains("ETCS L1 LS|ETCS L2 FS|ETCS L1 FS")]

    # Count occurrences of each incident type
    incident_counts = etcs_l1_fs_incidents['Incident description.2'].value_counts().reset_index()
    incident_counts.columns = ['Incident Type', 'Count']

    # Plot the results
    plt.figure(figsize=(12, 6))
    sns.barplot(data=incident_counts, x='Count', y='Incident Type', palette='Blues')
    plt.title('Incident Types for Selected ETCS Levels')
    plt.xlabel('Number of Incidents')
    plt.ylabel('Incident Type')
    plt.show()

def analyze_recovery_time(df_inc_ects):
    """
    Analyze average delay based on ETCS levels.
    
    Args:
    df_inc_ects (DataFrame): Merged DataFrame with ETCS levels.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_inc_ects, x='ETCS level', y='Minutes of delay', estimator=np.mean, palette='Blues')
    plt.title('Average Delay by ETCS Level')
    plt.ylabel('Average Delay (Minutes)')
    plt.xlabel('ETCS Level')
    plt.xticks(rotation=90)
    plt.show()

def analyze_top_incident_types(df_inc_ects):
    """
    Analyze the top incident types by ETCS level.
    
    Args:
    df_inc_ects (DataFrame): Merged DataFrame with ETCS levels.
    """
    # Count the incidents by description and ETCS level
    incident_types = df_inc_ects.groupby(['Incident description.2', 'ETCS level']).size().reset_index(name='Counts')

    # Plotting the top incident types
    top_incidents = incident_types.nlargest(25, 'Counts')

    plt.figure(figsize=(12, 12))
    sns.barplot(data=top_incidents, x='Counts', y='Incident description.2', hue='ETCS level', palette='viridis')
    plt.title('Top Incident Types by ETCS Level')
    plt.xlabel('Number of Incidents')
    plt.ylabel('Incident Description')
    plt.show()


# Function to train the model
def train_model(X_train, y_train, preprocessor):
    """
    Train a RandomForestRegressor model with the provided data.

    Args:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target variable.
    preprocessor (ColumnTransformer): Preprocessing pipeline.

    Returns:
    pipeline (Pipeline): Trained model pipeline.
    """
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', RandomForestRegressor(n_estimators=200, random_state=42))])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline
# Function to evaluate the model
def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate the model's performance on test data.

    Args:
    pipeline (Pipeline): Trained model pipeline.
    X_test (pd.DataFrame): Testing features.
    y_test (pd.Series): Testing target variable.
    
    Returns:
    mse (float): Mean Squared Error.
    r2 (float): R-squared value.
    """
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2
