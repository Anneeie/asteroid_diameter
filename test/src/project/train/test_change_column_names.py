import sys
import os
sys.path.append(os.path.abspath('C:\\Users\\user\\PycharmProjects\\final_project\\src\\project\\data'))
from eda import *
new_names_1 = ['X', 'Y']
new_names_2 = ['semi_major_axis', 'eccentricity', 'inclination', 'longitude_of_the_ascending_node', 'argument_of_perihelion', 'perihelion_distance', 'aphelion_distance', 'orbit_condition_code', 'num_obs_used_infit', 'absolute_magnitude', 'spkid','number_of_known_satellites', 'diameter', 'geometric_albedo', 'mean_anomaly', 'mean_motion', 'earth_minimum_orbit', 'earth_minimum_orbit_LD', ' jupiter_minimum_orbit', 'jupiter_tisserand', 'sigma_eccentricity', 'sigma_semi_major_axis','sigma_perihelion_distance', 'sigma_inclination', ' sigma_ascending_node', 'sigma_argument_perihelion', 'sigma_mean_anomaly', 'sigma_aphelion_distance', 'sigma_mean_motion', 'sigma_perihelion_passage', 'sigma_sidereal_orbital','rms', 'years_ago']
try:
    changed_df_1 = change_column_names(df_num, new_names_1)
    # No ValueError should be raised
except ValueError:
    raise AssertionError("Test case 1 failed. ValueError should not be raised.")

assert list(changed_df_1.columns) == new_names_1

