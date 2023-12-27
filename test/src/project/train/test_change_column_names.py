import sys
sys.path.append('C:\\Users\\user\\PycharmProjects\\final_project\\src\\project\\train')
from eda import change_column_names, df_num
new_names_1 = ['X', 'Y']
try:
    changed_df_1 = change_column_names(df_num, new_names_1)
    # No ValueError should be raised
except ValueError:
    raise AssertionError("Test case 1 failed. ValueError should not be raised.")

assert list(changed_df_1.columns) == new_names_1




