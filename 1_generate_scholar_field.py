import pandas as pd

# Load the first 1800 rows from the CSV file
file1_path = "/cluster/work/lawecon/Work/lixiang/authid_year_field_count.csv"
df = pd.read_csv(file1_path)

# Step 1: Group by 'authid' and 'col_name', and sum 'col_value'
df_summed = df.groupby(['authid', 'col_name'])['col_value'].sum().reset_index()

# Step 2: Filter to keep rows where 'col_name' starts with '_'
df_filtered = df_summed[df_summed['col_name'].str.startswith('_')]

# Step 3: Sort by 'authid' and 'col_value' in descending order, so that the highest `col_value` appears first
df_sorted = df_filtered.sort_values(by=['authid', 'col_value'], ascending=[True, False])

# Step 4: For each `authid`, pick the first row that starts with '_'
df_most_frequent = df_sorted.groupby('authid').head(1).reset_index(drop=True)

# Step 5: Rename 'col_name' to 'subfield_most_frequent'
df_most_frequent.rename(columns={'col_name': 'subfield_most_frequent'}, inplace=True)

# Step 6: Generate 'subfield_most_frequent_two_digit' from 'subfield_most_frequent'
df_most_frequent['subfield_most_frequent_two_digit'] = df_most_frequent['subfield_most_frequent'].str[:3]

# Step 7: Drop the 'col_value' column as it's not needed in the final result
df_most_frequent = df_most_frequent.drop(columns=['col_value'], errors='ignore')

# Step 8: Save the new DataFrame as a CSV file
generated_file_path = "/cluster/work/lawecon/Work/lixiang/generated_file.csv"
df_most_frequent.to_csv(generated_file_path, index=False)
