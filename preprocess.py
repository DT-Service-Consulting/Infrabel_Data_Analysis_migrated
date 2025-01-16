import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
import json
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
def preprocess_reg_data(filepath):
    """
    Load and preprocess data for analysis.

    Args:
    filepath (str): Path to the dataset.

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    # Load data
    df = pd.read_csv(filepath, sep =';')
    #converting the data formats for easy manipulations
    df['Date of departure'] = pd.to_datetime(df['Date of departure'])
    df['Planned arrival date'] = pd.to_datetime(df['Planned arrival date'])
    df['Planned departure date'] = pd.to_datetime(df['Planned departure date'])
    df['Actual arrival date'] = pd.to_datetime(df['Actual arrival date'])
    df['Actual departure date'] = pd.to_datetime(df['Actual departure date'])
    df['Departure line'] = df['Departure line'].astype(str)  # Ensure the line numbers are strings
    #converting seconds to minutes for clarity
    df['Delay at arrival'] = df['Delay at arrival'] / 60
    df['Delay at departure'] = df['Delay at departure'] / 60
#############################################################
    df.info()
########################################################
    df.describe()
########################################################
    #Checking the null value columns
    missing_values = df.isnull().sum()

    #Filter columns with missing values
    missing_columns = missing_values[missing_values > 0]

    # Display columns with missing values
    print("Columns with missing values:")
    print(missing_columns)
########################################################
    #showing the null value columns
    # Step 1: Create a boolean mask for missing values
    missing_mask = df.isnull().any(axis=1)

    # Step 2: Filter the DataFrame to show rows with at least one null value
    rows_with_nulls = df[missing_mask]

    # Display the rows with null values
    print("Rows with null values:")
    print(rows_with_nulls)
    #########################################################
    df.dropna(inplace=True)

    return df

def preprocess_inc_data(filepath):
    """
    Load and preprocess data for analysis.

    Args:
    filepath (str): Path to the dataset.

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    # Load data
    df = pd.read_csv(filepath, sep =';')
    df['Incident date'] = pd.to_datetime(df['Incident date']) #Ensuring dates are in same format
    df['Line'] = df['Line'].astype(str)  # Ensure the line numbers are strings
    df.info()

    return df


#Create GeoDataFrame for ETCS Lines using GeoShape
def parse_geo_shape(geo_shape_str):
    """Parse GeoShape string and return LineString."""
    geo_shape_json = json.loads(geo_shape_str)
    coordinates = geo_shape_json['coordinates'][0]  # assuming 'coordinates' is a list of points
    return LineString(coordinates)


def preprocess_etcs_data(filepath):
    """
    Load and preprocess data for analysis.

    Args:
    filepath (str): Path to the dataset.

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    # Load data
    df = pd.read_csv(filepath, sep =';')
    df['Line'] = df['Line'].astype(str)  # Ensure the line numbers are strings
    # Apply the function to create geometries for the lines
    df['geometry'] = df['GeoShape'].apply(parse_geo_shape)
    df.info()

    return df

def analyze_departure_data(df):
    """
    Analyzes the departure data to calculate the mean delay per departure hour
    and the correlation between hour of departure and delay.

    Args:
    df (pd.DataFrame): The preprocessed DataFrame with departure information.

    Returns:
    pd.Series, float: A Series containing the mean delay per hour and 
                      the correlation between departure hour and delay.
    """
    # Calculate the mean delay per hour
    df['Actual departure time'] = pd.to_datetime(df['Actual departure time'], format='%H:%M:%S')

# Extract the hour from the 'Actual departure time'
    df['Departure hour'] = df['Actual departure time'].dt.hour
    mean_delay_per_hour = df.groupby('Departure hour')['Delay at departure'].mean()

    # Calculate the correlation between hour of departure and delay
    correlation = df['Departure hour'].corr(df['Delay at departure'])

    return mean_delay_per_hour, correlation

def preprocess_top_10_places():
    """
    Prepares the coordinates and delays for the top 10 stopping places.

    Returns:
    places_coordinates (dict): A dictionary mapping each place to its coordinates.
    top_10_places (list): A list of tuples containing the stopping place and the associated delay.
    """
    # Coordinates for the top 10 stopping places
    places_coordinates = {
        'HERGENRATH': [50.7115, 6.0116],
        'GOUVY': [50.1899, 5.9335],
        'VIELSALM': [50.2791, 5.9241],
        'COO': [50.3943, 5.8998],
        'TROIS-PONTS': [50.3761, 5.8663],
        'AYWAILLE': [50.4751, 5.6761],
        'STOUMONT': [50.4454, 5.8261],
        'OVERPELT': [51.2142, 5.4486],
        'LOMMEL': [51.2300, 5.3139],
        'NOORDERKEMPEN': [51.3465, 4.7126]
    }

    # Top 10 stopping places with average departure delays
    top_10_places = [
        ('HERGENRATH', 15.406818),
        ('GOUVY', 8.388172),
        ('VIELSALM', 7.832381),
        ('COO', 7.830476),
        ('TROIS-PONTS', 7.279524),
        ('AYWAILLE', 7.188571),
        ('STOUMONT', 6.954762),
        ('OVERPELT', 6.538725),
        ('LOMMEL', 6.451471),
        ('NOORDERKEMPEN', 6.044872)
    ]

    return places_coordinates, top_10_places


def preprocess_delay_data(df_reg, df_et):
    """
    Preprocesses the delay data and ETCS deployment data.

    Args:
    df_reg (DataFrame): DataFrame containing train delay information.
    df_et (DataFrame): DataFrame containing ETCS level information.

    Returns:
    DataFrame: A merged DataFrame containing average delays and ETCS deployment status.
    """
    # Getting the average delay timing of each track from df_reg
    df_avg_delay = df_reg.groupby('Departure line').agg({
        'Delay at departure': 'mean',
        'Delay at arrival': 'mean'
    }).reset_index()

    # Rename columns for clarity
    df_avg_delay.columns = ['Departure line', 'Avg Delay at Departure', 'Avg Delay at Arrival']

    # Getting the Deployment of ECTS status on each line from df_et
    df_etcs = df_et.groupby('Line').agg({
        'ETCS level': lambda x: ', '.join(x.unique())  # Returns all unique ETCS levels as a comma-separated string
    }).reset_index()

    # Adjust line numbers for consistency
    df_etcs['Line'] = df_etcs['Line'].replace({
        '01': '0/1',
        '02': '0/2',
        '03': '0/3'
    })

    # Merging the two datasets on departure line to see if ETCS has been applied or not
    df_delay_etcs = pd.merge(df_avg_delay, df_etcs, left_on=["Departure line"], right_on=["Line"], how="left")

    # Filling the lines with No ETCS as "No ETCS"
    df_delay_etcs["ETCS level"] = df_delay_etcs["ETCS level"].fillna("No ETCS")

    return df_delay_etcs


def create_geo_dataframe(df_et):
    """
    Create a GeoDataFrame from the DataFrame containing track geometries.

    Args:
    df_et (DataFrame): DataFrame containing track geometries.

    Returns:
    GeoDataFrame: GeoDataFrame with track geometries.
    """
    df_et['geometry'] = df_et['GeoShape'].apply(parse_geo_shape)
    return gpd.GeoDataFrame(df_et, geometry='geometry', crs='EPSG:4326')


# Preprocessing function
def preprocess_data(df):
    """
    Preprocess the input data by converting categorical columns to category types and handling missing values.

    Args:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    X (pd.DataFrame): Features DataFrame.
    y (pd.Series): Target variable (Minutes of delay).
    """
    # Convert categorical variables to category dtype
    categorical_cols = ['Line_x', 'Place.2', 'Incident description.2', 'ETCS level']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # Convert the target variable to numeric
    df['Minutes of delay'] = pd.to_numeric(df['Minutes of delay'], errors='coerce')

    # Define features and target variable
    X = df[['Line_y', 'Place.2', 'Incident description.2', 'ETCS level', 'Total_Length_m']]
    y = df['Minutes of delay']
    
    return X, y
# Function to create a preprocessing pipeline
def create_preprocessing_pipeline():
    """
    Create a pipeline for preprocessing numerical and categorical data.

    Returns:
    preprocessor (ColumnTransformer): Preprocessing pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values for numeric features
                ('scaler', StandardScaler())  # Standardize 'Total_Length_m'
            ]), ['Total_Length_m']),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values for categorical features
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # Encode categorical variables
            ]), ['Line_y', 'Place.2', 'Incident description.2', 'ETCS level'])  # Categorical variables
        ])
    
    return preprocessor