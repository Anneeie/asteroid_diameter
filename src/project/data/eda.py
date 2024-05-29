import numpy as np
import pandas as pd
import math

df = pd.read_csv('C:\\Users\\user\\PycharmProjects\\final_project\\src\\project\\data\\asteroid diameter.csv',
                 low_memory=False)
df.drop(['full_name', 'name', 'producer', 'G', 'rot_per', 'BV', 'UB', 'spec_B', 'spec_T', 'diameter_sigma', 'epoch_mjd',
         'epoch', 'tp_cal', 'tp', 'per_y', 'per', 'data_arc'], axis=1, inplace=True)
df['first_year_as_int'] = pd.to_datetime(df['first_obs']).dt.year.astype(int)
df['last_year_as_int'] = pd.to_datetime(df['last_obs']).dt.year.astype(int)
new_names = ['semi_major_axis', 'eccentricity', 'inclination', 'longitude_of_the_ascending_node',
             'argument_of_perihelion', 'perihelion_distance', 'aphelion_distance', 'orbit_condition_code',
             'num_obs_used_infit', 'absolute_magnitude', 'spkid', 'number_of_known_satellites', 'diameter',
             'geometric_albedo', 'mean_anomaly', 'mean_motion', 'earth_minimum_orbit', 'earth_minimum_orbit_LD',
             ' jupiter_minimum_orbit', 'jupiter_tisserand', 'sigma_eccentricity', 'sigma_semi_major_axis',
             'sigma_perihelion_distance', 'sigma_inclination', ' sigma_ascending_node', 'sigma_argument_perihelion',
             'sigma_mean_anomaly', 'sigma_aphelion_distance', 'sigma_mean_motion', 'sigma_perihelion_passage',
             'sigma_sidereal_orbital', 'rms', 'years_ago']


def remove_outliers(df, percentile_threshold=99):
    """
    Remove outliers from a DataFrame based on the specified percentile threshold.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - percentile_threshold (int): Percentile threshold for outlier detection.

    Returns:
    - pd.DataFrame: DataFrame without outliers.
    - pd.DataFrame: DataFrame containing only numerical columns.
    """
    # Calculate the percentile threshold for each column
    numerical_variables = df.select_dtypes(include='number')
    threshold_values = np.percentile(numerical_variables, percentile_threshold, axis=0)

    # Identify rows where any value exceeds the percentile threshold
    outliers_mask = (numerical_variables > threshold_values).any(axis=1)

    # Count the number of outliers in each row
    outliers_count_per_row = outliers_mask.sum()

    # Create a DataFrame without outliers
    df_no_outliers = df[~outliers_mask]

    # Extract numerical columns from the cleaned DataFrame
    df_no_outliers = df_no_outliers.select_dtypes(include='number')

    return df_no_outliers


# Example usage:
# Assuming 'df' is your original DataFrame


def change_column_names(df, new_column_names):
    """
       Change the column names of a DataFrame.

       Parameters:
       - dataframe (pandas.DataFrame): The input DataFrame whose column names will be changed.
       - new_column_names (list): A list of new column names. The length of this list must match the number of columns in the input DataFrame.

       Returns:
       pandas.DataFrame: A new DataFrame with the column names updated.

       Raises:
       ValueError: If the number of new column names does not match the number of columns in the DataFrame.
       
    """
    if len(new_column_names) != len(df.columns):
        raise ValueError("Number of new column names must match the number of columns in the DataFrame.")

    renamed_dataframe = df.rename(columns=dict(zip(df.columns, new_column_names)))
    return renamed_dataframe


def select_highly_correlated_features(df, target_column, correlation_threshold=0.3, drop_threshold=0.85,
                                      output_file=None):
    """
    Select highly correlated features based on a given target column and correlation thresholds.

    Parameters:
    - df_numeric (pd.DataFrame): The input DataFrame containing numerical features.
    - target_column (str): The target column for which correlation is calculated.
    - correlation_threshold (float, optional): The initial correlation threshold to identify relevant features. Default is 0.3.
    - drop_threshold (float, optional): The threshold for dropping highly correlated features. Default is 0.85.
    - output_file (str or None, optional): If provided, the resulting DataFrame will be saved to this CSV file. Default is None.

    Returns:
    pd.DataFrame: A DataFrame with selected features based on correlation thresholds.

    """
    # Calculate correlation with the target column
    data_corr = df.corr()
    cor_target = abs(data_corr[target_column])
    relevant_features = cor_target[cor_target > correlation_threshold]

    # Extract relevant features and recalculate correlation matrix
    data_corr = df[relevant_features.index].corr()
    correlation_matrix = data_corr

    # Set a threshold for dropping highly correlated features
    correlated_columns = set()

    # Identify highly correlated columns
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > drop_threshold:
                colname = correlation_matrix.columns[i]
                correlated_columns.add(colname)

    # Drop highly correlated columns and handle missing values
    df_filtered = df[relevant_features.index].drop(columns=correlated_columns).dropna()

    # Reset index
    df_filtered.index = np.arange(0, len(df_filtered))

    # Save to CSV if specified
    if output_file:
        df_filtered.to_csv(output_file, index=False)

    return df_filtered


# df_numeric = remove_outliers(df, percentile_threshold=99)
df_numeric = df.drop(['first_obs', 'last_obs', 'last_year_as_int'], axis=1)
df_numeric['years_ago'] = 2023 - df_numeric['first_year_as_int']
df_numeric = df_numeric.drop(['first_year_as_int'], axis=1)
# df_num = change_column_names(df_numeric, new_names)
# df_num = select_highly_correlated_features(df_num, 'diameter')

df_num_for_other_train = change_column_names(df_numeric, new_names)

print(df_num_for_other_train.size)
