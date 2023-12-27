import numpy as np
import pandas as pd
import math
df =pd.read_csv('C:\\Users\\user\PycharmProjects\\final_project\src\project\data\\asteroid diameter.csv')
df.drop(['full_name','name','producer','G','rot_per','BV','UB','spec_B','spec_T', 'diameter_sigma', 'epoch_mjd', 'epoch','tp_cal','tp', 'per_y','per', 'data_arc'], axis=1,inplace=True)
df['first_year_as_int'] = pd.to_datetime(df['first_obs']).dt.year.astype(int)
df['last_year_as_int'] = pd.to_datetime(df['last_obs']).dt.year.astype(int)

# Define the percentile threshold (99th percentile in this case)
percentile_threshold = 99

# Calculate the 99th percentile for each column
numerical_variables = df.select_dtypes(include='number')
threshold_values = np.percentile(numerical_variables, percentile_threshold, axis=0)

# Identify rows where any value exceeds the percentile threshold
outliers_mask = (numerical_variables > threshold_values).any(axis=1)

# Count the number of outliers in each row
outliers_count_per_row = outliers_mask.sum()
df_no_outliers = df[~outliers_mask]
df_cleand = df_no_outliers.drop(['first_obs','last_obs', 'last_year_as_int'], axis=1)
df_cleand['years_ago'] = 2023-df_cleand['first_year_as_int']
df_cleand = df_cleand.drop(['first_year_as_int'], axis=1)
df_num = df_cleand.select_dtypes(include='number')

def change_column_names(dataframe, new_column_names):
    """
       Change the column names of a DataFrame.

       Parameters:
       - dataframe (pandas.DataFrame): The input DataFrame whose column names will be changed.
       - new_column_names (list): A list of new column names. The length of this list must match
                                  the number of columns in the input DataFrame.

       Returns:
       pandas.DataFrame: A new DataFrame with the column names updated.

       Raises:
       ValueError: If the number of new column names does not match the number of columns in the DataFrame.

       """
    if len(new_column_names) != len(dataframe.columns):
        raise ValueError("Number of new column names must match the number of columns in the DataFrame.")

    renamed_dataframe = dataframe.rename(columns=dict(zip(dataframe.columns, new_column_names)))
    return renamed_dataframe

new_names = ['semi_major_axis','eccentricity', 'inclination', 'longitude_of_the_ascending_node', 'argument_of_perihelion', 'perihelion_distance', 'aphelion_distance', 'orbit_condition_code', 'num_obs_used_infit', 'absolute_magnitude', 'spkid','number_of_known_satellites', 'diameter', 'geometric_albedo', 'mean_anomaly', 'mean_motion', 'earth_minimum_orbit', 'earth_minimum_orbit_LD', ' jupiter_minimum_orbit', 'jupiter_tisserand', 'sigma_eccentricity', 'sigma_semi_major_axis','sigma_perihelion_distance', 'sigma_inclination', ' sigma_ascending_node', 'sigma_argument_perihelion', 'sigma_mean_anomaly', 'sigma_aphelion_distance', 'sigma_mean_motion', 'sigma_perihelion_passage', 'sigma_sidereal_orbital','rms', 'years_ago']

df_num = change_column_names(df_num, new_names)

#Selecting highly correlated features
data_corr= df_num.corr()
cor_target = abs(data_corr["diameter"])
relevant_features = cor_target[cor_target > 0.3]
data_corr = df_num[relevant_features.index].corr()
correlation_matrix = data_corr

last_data = df_num[relevant_features.index]
# Set a threshold for correlation
threshold = 0.85
correlated_columns = set()

# Identify highly correlated columns
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            correlated_columns.add(colname)
df_filtered = last_data.drop(columns=correlated_columns)
df_filtered = df_filtered.dropna()
df_filtered.index = np.arange(0, len(df_filtered))
df_filtered.to_csv('C:\\Users\\user\PycharmProjects\\final_project\src\project\data\data.csv')

