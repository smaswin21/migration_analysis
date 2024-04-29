# lets put everything into one function
import os
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv() 

def get_news() -> dict:
    """We will use a news API to get the latest news on migrants. 
    In main() we will then filter the articles to only include those with the words 'missing' or 'dead' in the description"""

    yesterday = datetime.now() - timedelta(days=1)
    yesterday = yesterday.strftime('%Y-%m-%d')
    last_week = datetime.now() - timedelta(days=7)

    api_key = os.getenv('NEWS_API_KEY') 
    params = {
          'q': 'migrants',
          'apiKey': api_key, # We hardcode the API key here for simplicity (for enhanced security use environment variables)
          'language': 'en',
          'from': last_week, # Last week will be used for the demo. Ideally this should be changed to yesterday.
          'to': yesterday
    }

    response = requests.get('https://newsapi.org/v2/everything', params=params)
    data = response.json()
    return data 

def query_llm(filtered_articles) -> str:
    """We will query the llm model to get the JSON data for the news articles. 
    In this way we can derive information from the articles we couldnt extract directly"""


    prompt = f""" you are a journalist researching migration. You want to produce data from news articles. Given this list of news articles, return a JSON with each article's information in this format:
    json = {{ "Article 1": {{
        region_of_continent: string
        incident_date: "year-month-day"
        incident_year: int
        incident_month: "month"
        number_of_dead: int
        number_of_missing: int
        number_of_males: int
        countries_of_origin: string
        region_of_origin: string
        cause_of_death: string
        country_of_incident: string
        route_taken_by_migrants: string
        border_crossing_location: string  
        UNSD_Geographical_Grouping: string
        source_name: string
        url: string,
    }}
    ...
    }} Return just the JSON.
    Input list of news articles: {filtered_articles}
    """

    data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": prompt,
            "max_tokens": 2000,
            "temperature": 0.5
    }

    openai_api_key = os.getenv('OPENAI_API_KEY')

    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
    
    response = requests.post("https://api.openai.com/v1/completions", headers=headers, json=data)

    if response.status_code == 200:
        summary_data = response.json()

    news_json = summary_data['choices'][0]['text'].strip()
    return news_json


def quality_check(news_df) -> pd.DataFrame:
    """ We will evaluate the quality of the news data based on the presence or lack of cross-references.
    As per IMO standards, we will assign a source quality of 1 to articles with all fields present
    and 3 to articles that have been cross-referenced."""

    # Add a column for source quality
    news_df.insert(16, 'Source Quality', 1)
    remaining_duplicates_mask = news_df.duplicated(subset=['number_of_dead', 'number_of_missing', 'incident_date'], keep=False)
    remaining_duplicates = news_df[remaining_duplicates_mask]

    # Set source quality to 3 for remaining duplicates
    news_df.loc[remaining_duplicates.index, 'Source Quality'] = 3
    news_df.drop_duplicates(subset=['number_of_dead', 'number_of_missing', 'incident_date'], keep='first', inplace=True)
    return news_df

    
def cleaning_round(news_df, df_clean) -> pd.DataFrame:
    """We will do a first round of cleaning on the news data.
    1. Add a column for source quality
    2. Drop duplicates
    3. Set source quality to 3 for remaining duplicates
    4. Add a column for total number of dead and missing
    5. Rename columns as per the clean data"""

    # Add a column for source quality & evaluate quality
    news_df = quality_check(news_df)

    # Number of dead and missing
    news_df.insert(6, 'total number of dead and missing', news_df['number_of_dead'] + news_df['number_of_missing'])

    # Rename columns to match clean data
    news_df.columns = ['Region of Incident', 'Incident Date', 'Incident Year', 'Month', 'Number of Dead', 'Minmum Estimated Number of Missing', 'Total Number of Dead and Missing', 'Number of Males', 'Country of Origin', 'Region of Origin', 'Cause of Death', 'Country of Incidet', 'Migration Route', 'Location of Incident', 'UNSD Geographical Grouping ', 'Information Source', 'Url', 'Source Quality']
    news_df['Cause of Death'] = news_df['Cause of Death'].replace('Unknown', 'Mixed or unknown')

    # Adding missing columns
    news_df.insert(0, 'Main ID', '')
    news_df.insert(1, 'Incident ID', '')
    news_df.insert(2, 'Incident Type', 'Incident')
    news_df.insert(17, 'Coordinates', 'NaN')

    # Get the last 4 digits of the 'Main ID'
    last_row = df_clean.iloc[-2]
    last_main_id = str(last_row['Main ID'])[-4:]
    for i, row in news_df.iterrows():
        last_main_id = int(last_main_id) + 1
        news_df.loc[i, 'Main ID'] = f"{row['Incident Year']}MMP{last_main_id}"
    news_df['Incident ID'] = news_df['Main ID']

    return news_df


def main() -> None:
    df_clean = pd.read_csv('data/df_clean.csv')
    news_data = get_news()

    # Introduce a filter to retain only high quality articles without missing information
    filtered_articles = []
    for article in news_data['articles']:
        if all(article.get(field) is not None for field in ['author', 'description', 'url']):
            if "missing" in article.get('description') or "dead" in article.get('description'):
                filtered_articles.append(article)
    
    news_data['articles'] = filtered_articles
    news_data['totalResults'] = len(filtered_articles)

    news_json = query_llm(filtered_articles)

    # Debugging line
    print("Received JSON:", news_json)  

    # We turn our news data into a dataframe
    news_df = pd.DataFrame(json.loads(news_json)).T

    # Cleaning step
    news_df = cleaning_round(news_df, df_clean)

    # Saving just the added data 
    news_df.to_csv('data/news_data.csv', mode='a', header=False)

    # Merging step of the data 
    df_clean = pd.concat([df_clean, news_df], ignore_index=True)
    df_clean.to_csv('data/clean_data_merged.csv', index=False)

main()




     
