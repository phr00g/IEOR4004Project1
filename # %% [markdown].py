# %% [markdown]
# ## Data Cleaning

# %% [markdown]
# We first find the missing data.

# %%
import pandas as pd
import sys

df = pd.read_csv('child_care_regulated.csv')

missing_count = df.isnull().sum()
total_rows = len(df)
missing_percentage = (missing_count / total_rows) * 100

missing_report = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing Percentage': missing_percentage.round(2)
})

missing_report = missing_report[missing_report['Missing Count'] > 0].sort_values(
    by='Missing Count', ascending=False
)

print("--- Missing Value Report for child_care_regulated.csv ---")
print(f"Total rows processed: {total_rows}")

if missing_report.empty:
    print("\n[Result]: No missing values found in any column.")
else:
    print("\n[Result]: Columns with missing data:")
    print(missing_report.to_markdown())


# %% [markdown]
# Then, we process some strange data, like zipcode.

# %%
import re


ZIP_CODE_COL = 'zip_code'
CLEANED_ZIP_COL = 'zip_code_cleaned'

# 1. Data Cleaning Prep: Convert to string and clean non-digit characters
df['zip_temp'] = df[ZIP_CODE_COL].astype(str).str.replace(r'\.0$', '', regex=True)
df['zip_temp'] = df['zip_temp'].str.replace(r'\D', '', regex=True)
df.loc[df[ZIP_CODE_COL].isna(), 'zip_temp'] = pd.NA

long_zipcodes_mask = (df['zip_temp'].str.len() > 5) & (df['zip_temp'].notna())
long_zipcodes_examples = df[long_zipcodes_mask].head(10)[[ZIP_CODE_COL, 'zip_temp']]

print("--- 1. Example of Zip Codes longer than 5 digits ---")
if long_zipcodes_examples.empty:
    print("No zip codes found with length greater than 5 digits.")
else:
    print(long_zipcodes_examples.to_string(header=True, index=False))

# 2. Core Processing: Truncate to 5 digits and store in the new column
df[CLEANED_ZIP_COL] = df['zip_temp'].apply(
    lambda x: x[:5] if pd.notna(x) and len(str(x)) > 5 else x
)

df = df.drop(columns=['zip_temp'])

# 3. Output Range of the Cleaned Zip Codes
zip_min = df[CLEANED_ZIP_COL].min()
zip_max = df[CLEANED_ZIP_COL].max()

print("\n--- 2. Zip Code Cleaning Summary ---")
truncated_count = (df[CLEANED_ZIP_COL].str.len() == 5) & (long_zipcodes_mask)
print(f"Total entries truncated to 5 digits: {truncated_count.sum()}")
print(f"Cleaned '{CLEANED_ZIP_COL}' Min (Lexicographical): {zip_min}")
print(f"Cleaned '{CLEANED_ZIP_COL}' Max (Lexicographical): {zip_max}")


# %%
df.head()

# %%
# Assuming 'df' is the DataFrame currently in use (from child_care_regulated.csv)
# Assuming the 'zip_code_cleaned' column has already been created in a previous step.

# --- 1. Drop Rows with Missing Geo-Coordinates ---
initial_rows = len(df)
df_cleaned = df.dropna(subset=['latitude', 'longitude']).copy()

rows_removed = initial_rows - len(df_cleaned)
print("--- 1. Missing Geographical Data Handling ---")
print(f"Rows dropped due to missing 'latitude' or 'longitude': {rows_removed}")
print(f"Remaining rows for analysis: {len(df_cleaned)}")


# --- 2. Identify and Drop Redundant Columns (Retaining 'children_capacity') ---

# Columns to keep for modeling, capacity calculation, and location constraints:
# zip_code_cleaned (essential key), capacity columns, geo-coordinates, and IDs.
COLUMNS_TO_KEEP = [
    'facility_id',
    'zip_code_cleaned', # Retain cleaned zip_code
    'infant_capacity',
    'toddler_capacity',
    'preschool_capacity',
    'school_age_capacity',
    'children_capacity',
    'total_capacity',
    'latitude',
    'longitude'
]

# Calculate columns to drop
all_columns = set(df_cleaned.columns)
columns_to_drop = list(all_columns - set(COLUMNS_TO_KEEP))

# Drop the columns
df_cleaned = df_cleaned.drop(columns=columns_to_drop, errors='ignore')

df_cleaned.to_csv('child_care_regulated_cleaned.csv', index=False)

df_cleaned.head()

# %% [markdown]
# We processs the population data 

# %%
import pandas as pd
import numpy as np

df_pop = pd.read_csv('population.csv')

# --- 1. Data Preparation and Cleaning (Including 4-Digit Fix) ---

df_pop['zip_code_cleaned'] = df_pop['zipcode'].astype(str).str.replace(r'\.0$', '', regex=True)
df_pop['zip_code_cleaned'] = df_pop['zip_code_cleaned'].str.replace(r'\D', '', regex=True)
df_pop.loc[df_pop['zipcode'].isna(), 'zip_code_cleaned'] = pd.NA

four_digit_mask = (df_pop['zip_code_cleaned'].str.len() == 4) & (df_pop['zip_code_cleaned'].notna())
fixed_count = four_digit_mask.sum()

df_pop.loc[four_digit_mask, 'zip_code_cleaned'] = '0' + df_pop.loc[four_digit_mask, 'zip_code_cleaned']


COL_0_4 = '-5'
COL_5_9 = '5-9'
COL_10_14 = '10-14'

# Fill any NaNs in the population columns with 0 before calculation
pop_cols_to_use = [COL_0_4, COL_5_9, COL_10_14]
df_pop[pop_cols_to_use] = df_pop[pop_cols_to_use].fillna(0)


# --- 2. Calculate Population 0-5 and 0-12 using Linear Interpolation ---

# 2a. Calculate Population 0-5 (0-4 + 1/5 of 5-9)
df_pop['Population_0_5'] = (
    df_pop[COL_0_4] + 
    df_pop[COL_5_9] * (1/5)
)

# 2b. Calculate Population 0-12 (0-4 + 5-9 + 3/5 of 10-14)
df_pop['Population_0_12'] = (
    df_pop[COL_0_4] + 
    df_pop[COL_5_9] + 
    df_pop[COL_10_14] * (3/5)
)


df_pop_calculated = df_pop[['zip_code_cleaned', 'Population_0_5', 'Population_0_12']].copy()
df_pop_calculated[['Population_0_5', 'Population_0_12']] = df_pop_calculated[['Population_0_5', 'Population_0_12']].round(0).astype(int)
df_pop_calculated.to_csv('population_calculated.csv', index=False)

print("\nPreview of the calculated population data (showing fixed zip codes if applicable):")
print(df_pop_calculated.head(5).to_markdown(index=False))

# %% [markdown]
# Find high demand and normal demand area

# %%
import pandas as pd

HIGH_EMPLOYMENT_THRESHOLD = 0.60  
HIGH_INCOME_THRESHOLD = 60000     

def clean_zip_column_safe(df, zip_col):
    
    """Cleans and fixes 4-digit zip codes by prepending '0', and truncates long ones."""
    cleaned_col_name = f'{zip_col}_cleaned'
    df[cleaned_col_name] = df[zip_col].astype(str).str.replace(r'\.0$', '', regex=True)
    df[cleaned_col_name] = df[cleaned_col_name].str.replace(r'\D', '', regex=True)
    
    four_digit_mask = (df[cleaned_col_name].str.len() == 4) & (df[cleaned_col_name].notna())
    df.loc[four_digit_mask, cleaned_col_name] = '0' + df.loc[four_digit_mask, cleaned_col_name]
    
    long_zip_mask = (df[cleaned_col_name].str.len() > 5) & (df[cleaned_col_name].notna())
    df.loc[long_zip_mask, cleaned_col_name] = df.loc[long_zip_mask, cleaned_col_name].str[:5]
    
    return df

# ---  Load data ---

df_income = pd.read_csv('avg_individual_income.csv')
df_income = clean_zip_column_safe(df_income, 'ZIP code')
df_income = df_income.rename(columns={df_income.columns[1]: 'Average_Income_Raw'}) 
df_income = df_income[[f"{'ZIP code'}_cleaned", 'Average_Income_Raw']].rename(
    columns={f"{'ZIP code'}_cleaned": 'zip_code_cleaned', 'Average_Income_Raw': 'Average_Income'}
)

df_employment = pd.read_csv('employment_rate.csv')
df_employment = clean_zip_column_safe(df_employment, 'zipcode' )
df_employment = df_employment.rename(columns={df_employment.columns[1]: 'Employment_Rate_Raw'}) 
df_employment = df_employment[[f"{'zipcode'}_cleaned", 'Employment_Rate_Raw']].rename(
    columns={f"{'zipcode'}_cleaned": 'zip_code_cleaned', 'Employment_Rate_Raw': 'Employment_Rate'}
)

# %%
# Merge the two datasets on the *common cleaned zip code column*
df_demand_factors = pd.merge(
    df_income,
    df_employment,
    on='zip_code_cleaned',
    how='outer'  # Use outer merge to keep all zip codes from both files
)

# Handle missing data after merge by filling with neutral values 
# (These values are chosen to NOT trigger High Demand when data is missing)
df_demand_factors['Employment_Rate'] = df_demand_factors['Employment_Rate'].fillna(0.0) 
df_demand_factors['Average_Income'] = df_demand_factors['Average_Income'].fillna(float('inf'))


# %%
# Determine High Demand Status (True if either condition is met)
df_demand_factors['Is_High_Demand'] = (
    (df_demand_factors['Employment_Rate'] >= HIGH_EMPLOYMENT_THRESHOLD) | 
    (df_demand_factors['Average_Income'] <= HIGH_INCOME_THRESHOLD)
)

# Convert boolean to 'High' / 'Normal' string classification
df_demand_factors['Demand_Classification'] = df_demand_factors['Is_High_Demand'].map({
    True: 'High', 
    False: 'Normal'
})

print("--- Demand Classification Summary ---")
print(f"Total unique zip codes processed: {len(df_demand_factors)}")
print(f"Number of High Demand Areas: {df_demand_factors['Is_High_Demand'].sum()}")

# Select and rename final columns for output
df_classification = df_demand_factors[[
    'zip_code_cleaned', 
    'Employment_Rate', 
    'Average_Income', 
    'Demand_Classification'
]].copy()

df_classification.to_csv('demand_classification.csv', index=False)

print(f"\nClassification results saved to: {'demand_classification.csv'}")
print("\nPreview of Demand Classification:")
print(df_classification.head(5).to_markdown(index=False))

df_demand_classification = df_classification


