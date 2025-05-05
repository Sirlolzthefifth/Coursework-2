#Cleaning and preprocessing of owid-covid dataset
import pandas as pd
import numpy as np

# Loading the raw dataset
df = pd.read_csv('/Users/Zof/Downloads/owid-covid-data.csv')

# Filtering data to include only countries from Europe
df_europe = df[df['continent'] == 'Europe'].copy()

# Output the number of European countries before exclusions
print('Initial number of European countries:', df_europe['location'].nunique())

# Defining subnational or aggregate entities to exclude
entries_to_drop = ['Russia', 'England', 'Scotland', 'Wales', 'Northern Ireland',
                   'Faroe Islands', 'Gibraltar', 'Guernsey', 'Isle of Man', 'Jersey']

# Excluding entries using vectorised filtering
df_europe = df_europe[~df_europe['location'].isin(entries_to_drop)]

# Displaying updated list of countries after exclusions
print('European countries (post-cleaning):', sorted(df_europe['location'].unique()))
print('Total remaining countries:', df_europe['location'].nunique())

# Retaining only selected columns relevant for analysis
columns_to_keep = [
    'continent', 'location', 'date', 'population',
    'new_cases', 'new_deaths',
    'new_cases_per_million', 'new_deaths_per_million',
    'people_fully_vaccinated_per_hundred',
    'stringency_index', 'gdp_per_capita'
]
df_europe = df_europe[columns_to_keep]

# Converting date column to datetime format
df_europe['date'] = pd.to_datetime(df_europe['date'])

# Deriving a 'month' column as human-readable string format (e.g., 'April 2020')
df_europe['month'] = df_europe['date'].dt.to_period('M').dt.to_timestamp()
df_europe['month'] = df_europe['month'].dt.strftime('%B %Y')

# Calculating average new cases per month per country
monthly_avg = df_europe.groupby(['location', 'month'])['new_cases'].mean().reset_index()
monthly_avg.rename(columns={'new_cases': 'new_cases_month_avg'}, inplace=True)

# Merging monthly averages back into the main DataFrame
df_europe = df_europe.merge(monthly_avg, on=['location', 'month'], how='left')

# Displaying percentage of missing values by column
missing = df_europe.isnull().mean() * 100
print('Missing value percentages:', missing.sort_values(ascending=False))

# Checking for any fully duplicated rows
if df_europe.duplicated().any():
    print('Duplicate rows found.')
else:
    print('No duplicate rows found.')

# Checking for duplicates based on (location, date) combinations
dupes_loc_date = df_europe.duplicated(subset=['date', 'location'])
if dupes_loc_date.any():
    print('Duplicate (location, date) combinations:', dupes_loc_date.sum())
else:
    print('No duplicate (location, date) combinations found.')

# --- DATA CLEANING ---

# Filling vaccine data with 0 only for dates before vaccine rollout (Dec 2020)
vaccine_start = pd.to_datetime('2020-12-01')
df_europe['people_fully_vaccinated_per_hundred'] = df_europe.apply(
    lambda row: 0 if pd.isna(row['people_fully_vaccinated_per_hundred']) and row['date'] < vaccine_start
    else row['people_fully_vaccinated_per_hundred'], axis=1
)

# Forward-filling 'stringency_index' within each country
df_europe['stringency_index'] = df_europe.groupby('location')['stringency_index'].ffill()

# Filling missing stringency values with 0 only for early pre-March 2020 entries
cutoff = pd.to_datetime("2020-03-01")
df_europe['stringency_index'] = df_europe.apply(
    lambda row: 0 if pd.isna(row['stringency_index']) and row['date'] < cutoff else row['stringency_index'],
    axis=1
)

# Replacing blank or zero GDP values with NaN before filling
df_europe['gdp_per_capita'] = df_europe['gdp_per_capita'].replace(['', ' ', 0], np.nan)
df_europe['gdp_per_capita'] = df_europe.groupby('location')['gdp_per_capita'].ffill()

# Imputing new cases and deaths with forward-fill and linear interpolation
df_europe['new_cases'] = df_europe.groupby('location')['new_cases'].ffill().interpolate()
df_europe['new_deaths'] = df_europe.groupby('location')['new_deaths'].ffill().interpolate()

# Recalculating per-million indicators using cleaned values
df_europe['new_cases_per_million'] = df_europe['new_cases'] / df_europe['population'] * 1e6
df_europe['new_deaths_per_million'] = df_europe['new_deaths'] / df_europe['population'] * 1e6

# Filling missing monthly average values with 0
df_europe['new_cases_month_avg'] = df_europe['new_cases_month_avg'].fillna(0)

# Displaying any remaining missing values
missing = df_europe.isnull().sum()
print("Remaining missing values after cleaning:", missing[missing > 0])

# Exporting the cleaned and preprocessed dataset to CSV
df_europe.to_csv('/Users/Zof/Downloads/europe_covid_cleaned9.csv', index=False, na_rep='NaN')
print('Cleaned dataset successfully saved.')
