from pathlib import Path
import numpy as np
import pandas as pd
import pycountry
from geotext import GeoText

app_dir = Path(__file__).parent
merged_df = pd.read_csv("merged_df.csv")
merged_df['Region of Origin'] = [str(region).replace(' (P)', '') for region in merged_df['Region of Origin']]

df = pd.read_csv("data.csv")
# Ensure df['Region of Incident'] and df['Incident Year'] are properly formatted
df['Incident Year'] = pd.to_numeric(df['Incident Year'], errors='coerce')  # Convert to numeric, handle errors
df.dropna(subset=['Incident Year'], inplace=True)  # Remove rows where 'Incident Year' could not be converted

#Clean Data
def clean_data(df):    
    
    threshold = len(df) * 0.5
    df_clean = df.dropna(thresh=threshold, axis=1)

    # Assuming 'column_name' is the name of your column
    split_by_countries = df_clean.copy()
    split_by_countries = df_clean['Country of Origin'].str.split(',', expand=True)
    split_by_countries.columns = ['Country of Origin ' + str(col + 1) for col in split_by_countries.columns]
    split_by_countries = split_by_countries.replace('unknown', np.nan)

    # Reset the index to use the index as an identifier for melting
    split_by_countries.reset_index(inplace=True)

    # Melt the DataFrame
    melted_df = split_by_countries.melt(id_vars='index', value_vars=split_by_countries.columns[1:], value_name='Country of Origin')

    # Remove NaN values
    melted_df = melted_df.dropna(subset=['Country of Origin'])

    # Merge with the original DataFrame to get the other columns
    final_df = pd.merge(df_clean, melted_df, left_index=True, right_on='index')

    # Drop unnecessary columns
    split_by_countries = final_df.drop(['index', 'variable'], axis=1)

    # Getting the country code for country or origin
    def country_to_code(country_name):
        try:
            return pycountry.countries.get(name=country_name).alpha_3
        except AttributeError:
            return np.nan

    df_clean['Country of Origin Code'] = split_by_countries['Country of Origin_y'].apply(country_to_code)
    df_clean = df_clean[df_clean['Country of Origin'] != 'Unknown']

    df_location = pd.DataFrame(df_clean['Location of Incident'])
    # Function to extract country name
    def extract_country(location):
        places = GeoText(location)
        # Get countries as a list, could be more than one country mentioned
        countries = places.countries
        # Join all found countries with ', ' if there's more than one, or just return the single country
        return countries[0] if countries else "Unknown"

    # Apply the function to create a new column
    df_clean['Country of Incident'] = df_clean['Location of Incident'].apply(extract_country)
            
    # For handling unknown countries
    location_to_country = {
        'Arizona': 'USA',
        'Texas': 'USA',
        'Türkiye': 'Türkiye',
        'California': 'USA',
        'Central Mediterranean': 'Central Mediterranean',
        'Morroco': 'Morroco',
        'Liby': 'Sudanese-Libyan'
    }

    # Define a function to apply the mapping
    def replace_unknown(row):
        if row['Country of Incident'] == 'Unknown':
            for location, country in location_to_country.items():
                if location in row['Location of Incident']:
                    return country
        return row['Country of Incident']

    # Apply the function to the 'countries' column
    df_clean['Country of Incident'] = df_clean.apply(replace_unknown, axis=1)
    
    return df_clean

df = clean_data(df)
