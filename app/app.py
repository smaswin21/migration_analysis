import plotly.express as px
from palmerpenguins import load_penguins
from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget  
import pandas as pd
from shared import merged_df, df
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from geotext import GeoText
import random
import pycountry
import numpy as np
import IPython.display as display
import plotly.graph_objects as go
from xgboost import XGBRegressor
import seaborn as sns
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import os
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv() 
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv() 

penguins = load_penguins()

try:
    path='network_graph.html'
    with open(path, "r") as file:
        html_content = file.read()
        
    path_legend='LegendCard.html'
    with open(path_legend, "r") as file:
        html_legend = file.read()
except FileNotFoundError:
    html_content = "No network graph available."

# app_ui = ui.page_fluid(
#     ui.input_dark_mode(id='mode', mode="dark"),
#     ui.layout_column_wrap(
#     ui.input_slider(id="date", label="Year range:", min = 2014, max = 2023,value = (2014,2023), ticks=False, sep=''),
#     ui.input_selectize('region', 'Choose a Region:', list(merged_df['Region of Origin'].unique()), 
#                        selected= [str(region) for region in merged_df['Region of Origin'].unique() if 'Africa' in str(region)], multiple=True),
#     ),
#     output_widget("line_chart"),
#     print(ui.HTML(ui.output_ui("network_graph"))),
#     # ui.output_plot(ui.output_text("network_graph")),
#     ui.output_plot("network_graph"),

# )

app_ui = ui.page_fillable(
    ui.navset_card_tab(
        # Tab for Input Controls
        ui.nav_panel(
            "Dashboard",
            ui.input_dark_mode(id='mode', mode="dark"),
            
            ui.div(            
                # Main layout with selection inputs and responsive columns
                ui.layout_column_wrap(
                    # Selection inputs with added padding for spacing
                    ui.input_selectize('region_incident_select', 'Choose a Region of Incident:', list(df['Region of Incident'].unique()), 
                                    selected=[str(region) for region in df['Region of Incident'].unique() if 'America' in str(region)], 
                                    multiple=True),
                    ui.input_select(
                        "year_range",
                        "Select Time Range",
                        choices=["Total", "2-Year Range", "5-Year Range"],
                        selected="Total"
                    ),
                ),
            ),

            ui.div(
                # Data display cards wrapped in layout for added padding
                ui.layout_column_wrap(
                    ui.card(
                            ui.card_header("Country of Origin"),
                            ui.output_data_frame("summary_statistics")
                        ),
                    ui.card(
                            ui.card_header("Migration Route"),
                            ui.output_data_frame("summary_statistics1")
                        ),
                    ui.card(
                            ui.card_header("Country of Incident"),
                            ui.output_data_frame("summary_statistics2")
                        ),
                    ui.card(
                            ui.card_header("Cause of Death"),
                            ui.output_data_frame("summary_statistics3")
                        )
                ),
            ), 
            
            ui.div(
                ui.layout_column_wrap(
                    ui.input_selectize('region_origin_select', 'Choose a Region of Origin:', list(merged_df['Region of Origin'].unique()), 
                        selected=[str(region) for region in merged_df['Region of Origin'].unique() if 'Africa' in str(region)], 
                        multiple=True),
                ),
            ),
            ui.div(  
            output_widget("line_chart"),
            ui.layout_column_wrap(
                ui.input_slider(id="date", label="Year range:", min=2014, max=2023, value=(2014, 2023), ticks=False, sep='')
                ),
            ),
        ),
        # Tab for Line Chart
        ui.nav_panel(
            "Forecast",
            ui.layout_column_wrap(
                ui.input_selectize(id='route_selection', label='Choose a Route:', choices=list(merged_df['Migration Route'].unique()), 
                                    selected=[str(route) for route in merged_df['Migration Route'].unique() if 'Mediterranean' in str(route)], 
                                    multiple=True),
            ui.input_task_button("btn_forecast", "Compute the Forecast"),
            ),
            output_widget("forecast_graph"),
        ),
        # Tab for Network Graph
        ui.nav_panel(
            "Network Graph",
            # ui.input_task_button("btn_network_graph", "Compute the Network Graph"),
            ui.HTML(html_content),  
            ui.layout_column_wrap(
                ui.column(12, ui.card(
                    ui.card_header("Legend"),
                    ui.HTML(html_legend),
                )),
            ),
        ),
        # Tab for Sources
        ui.nav_panel(
            "Sources Scrapper",
            ui.input_task_button("btn_scrape2", "Get New Data"),
            ui.card(
                ui.card_header("Scrapping New Data"),
                ui.output_data_frame("scrape_source2"),
            ),
        ),
        id="tab",
    )
)

def server(input, output, session):
    @render_widget  
    def line_chart():  
        #Clean Data
        merged_df = pd.read_csv("merged_df.csv")
        merged_df = merged_df.groupby(['Incident Year', 'Migration Route', 'Region of Origin'])['Total Number of Dead and Missing'].sum().reset_index()
        merged_df = merged_df.drop(merged_df[merged_df['Migration Route'] == 'Mixed'].index)
        merged_df['Region of Origin'] = [str(region).replace(' (P)', '') for region in merged_df['Region of Origin']]
        
        #Log
        # print(input.region())
        
        # Filter
        merged_df = merged_df[(merged_df['Incident Year'] >= input.date()[0]) & (merged_df['Incident Year'] <= input.date()[1])]
        merged_df = merged_df[(merged_df['Region of Origin'].isin(input.region_origin_select()))]  
        
        #Groupby
        merged_df=merged_df.groupby(['Incident Year', 'Migration Route'])['Total Number of Dead and Missing'].sum().reset_index()
        
        line_chart = px.line(
            data_frame=merged_df,
            x="Incident Year",
            y='Total Number of Dead and Missing',
            line_group="Migration Route",
            color="Migration Route",
            line_shape="spline", 
            render_mode="svg",
            markers=True,
        ).update_layout(
            title={"text": "Evolution of Illegal Immigration", "x": 0.5},
            hovermode='x unified',
            plot_bgcolor='#1D2021' if input.mode() == 'dark' else '#FFFFFF',
            paper_bgcolor='#1D2021' if input.mode() == 'dark' else '#FFFFFF',
            title_font=dict(color="white", size=20),  # Explicitly set title font and color
            xaxis=dict(  # Customize x-axis appearance
                title="Years",
                title_font=dict(color='#FFFFFF' if input.mode() == 'dark' else '#1D2021'),
                tickfont=dict(color='#FFFFFF' if input.mode() == 'dark' else '#1D2021'),
                linecolor='#FFFFFF' if input.mode() == 'dark' else '#1D2021',  # Set x-axis line color to white
                tickcolor='#FFFFFF' if input.mode() == 'dark' else '#1D2021',  # Set x-axis tick color to white
                showgrid=False  # Optionally hide the grid for a cleaner look
            ),
            yaxis=dict(  # Customize y-axis appearance
                title="Count",
                title_font=dict(color='#FFFFFF' if input.mode() == 'dark' else '#1D2021'),
                tickfont=dict(color='#FFFFFF' if input.mode() == 'dark' else '#1D2021'),
                linecolor='#FFFFFF' if input.mode() == 'dark' else '#1D2021',  # Set y-axis line color to white
                tickcolor='#FFFFFF' if input.mode() == 'dark' else '#1D2021',  # Set y-axis tick color to white
                showgrid=False  # Optionally hide the grid for a cleaner look
            ),
            legend=dict(title_text='Legend', font=dict(color='#FFFFFF' if input.mode() == 'dark' else '#1D2021', size=12), bgcolor='rgba(0,0,0,0)'),
            
        )
        return line_chart  
    
    @render.plot
    @reactive.event(input.btn_network_graph)
    def network_graph():
        
        df_clean = pd.read_csv("hackathon-challenge-2/data/Missing_Migrants_Global_Figures_allData.csv")

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

            split_by_countries['Country of Origin Code'] = split_by_countries['Country of Origin_y'].apply(country_to_code)

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
                        
            
            df_network = df_clean.groupby(['Country of Origin', 'Migration Route', 'Cause of Death', 'Country of Incident', 'Region of Incident']).agg({
                'Total Number of Dead and Missing': 'sum'
            }).reset_index()
            
            return df_network
        
        # Calling Clean data
        df_network = clean_data(df_clean)
        
        # Defining colors for each cause of death
        color_map = {}
        
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF']

        for cause in df_clean['Cause of Death']:
            color_map[str(cause)] = random.choice(colors)
        
        # Create the network graph
        G = nx.Graph()

        attributes = ['Region of Incident', 'Country of Incident', 'Migration Route']
        # Add nodes and edges with attributes
        for i, row in df_network.iterrows():
            origin = str(row['Country of Origin'])
            route = str(row['Migration Route'])
            incident = str(row['Country of Incident'])
            deaths = row['Total Number of Dead and Missing']
            cause = str(row['Cause of Death'])

            # Adding nodes
            G.add_node(origin, title=origin, group=cause, color=color_map.get(cause, 'grey'))
            G.add_node(route, title=route, group=cause, color=color_map.get(cause, 'grey'))
            G.add_node(incident, title=incident, group=cause, color=color_map.get(cause, 'grey'))
            
            # Adding edges
            title_text = f"Deaths: {deaths}"
            G.add_edge(origin, route, title=title_text, width=deaths*10) 
            G.add_edge(route, incident, title=title_text, width=deaths*10)

        # Convert to PyVis network
        nt = Network(notebook=False, height="750px", width="100%", bgcolor="#222222", font_color="white")
        nt.from_nx(G)
        nt_html = nt.generate_html()

        # print(nt_html)
        # display.HTML(nt_html)
        
        # nt.show("network_graph.html")
        nt.save_graph("network_graph.html")
        
        return 
    
    @render_widget 
    @reactive.event(input.btn_forecast)
    def forecast_graph():
        def predict_future_impacts_by_route(data_path, target, routes, features=None, date_columns=None, test_size_days=10, year=2024, noise_scale=0.01):
            # Load the dataset
            df = pd.read_csv(data_path)

            # Date processing
            if 'Date' not in df.columns and {'year', 'month', 'day'}.issubset(df.columns):
                df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                raise ValueError("No appropriate column for 'Date'.")

            df.set_index('Date', inplace=True)

            # Data type conversions
            object_cols = df.select_dtypes(include='object').columns
            for col in object_cols:
                df[col] = df[col].astype('category')

            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

            # Define features if not specified
            if features is None:
                features = df.columns.drop(target)

            # Initialize the results dictionary
            results = {}

            # Process each specified route
            for route in routes:
                print(f"Processing route: {route}")
                route_df = df[df['Migration Route'] == route]
                
                if route_df.empty:
                    print(f"No data for route: {route}")
                    continue
                
                # Split the data
                split_date = route_df.index.max() - pd.Timedelta(days=int(test_size_days))
                train = route_df[route_df.index <= split_date]
                test = route_df[route_df.index > split_date]

                # Train the XGBRegressor
                model = XGBRegressor(enable_categorical=True, alpha=0.1, reg_lambda=1.0)
                model.fit(train[features], train[target])

                # Predictions with randomness for each month in the specified year
                start_date = pd.Timestamp(f'{year}-01-01')
                dates_year = pd.date_range(start=start_date, periods=12, freq='MS')

                predictions_year = {}
                for date in dates_year:
                    last_features = train.iloc[-1][features].copy()
                    last_features['year'] = date.year
                    last_features['month'] = date.month
                    for col in numeric_cols:
                        if col in last_features and last_features[col] >= 0:
                            scale = noise_scale * abs(last_features[col])
                            last_features[col] += np.random.normal(0, scale)
                    last_features_df = pd.DataFrame([last_features], index=[date])
                    last_features_df[object_cols] = last_features_df[object_cols].apply(lambda x: x.astype('category'))
                    prediction = model.predict(last_features_df[features])[0]
                    predictions_year[date.strftime('%Y-%m')] = prediction

                # Store results for the route
                results[route] = predictions_year
                print(f"Predictions for route {route} in {year} with added variation:")
                for date, prediction in predictions_year.items():
                    print(f"{date}: {prediction:.2f}")

            return results
        
        #Calling the function
        path='df_web.csv'
        print(type(input.route_selection))
        routes = ['Central Mediterranean', 'Western Balkans', 'Estern Mediterranean']
        # routes = input.route_selection.tolist()
        predictions_by_route = predict_future_impacts_by_route(path, 'total_dead_and_missing', routes)


        data = []
        for route, predictions in predictions_by_route.items():
            for date, prediction in predictions.items():
                data.append({"Date": date, "Predicted total_dead_and_missing": prediction, "Migration Route": route})

        df_predictions = pd.DataFrame(data)
        # Plotting results for each route
        # Set colors based on the mode
        mode = input.mode()
        plot_bgcolor = '#1D2021' if mode == 'dark' else '#FFFFFF'
        paper_bgcolor = '#1D2021' if mode == 'dark' else '#FFFFFF'
        text_color = '#FFFFFF' if mode == 'dark' else '#1D2021'
        grid_color = '#FFFFFF' if mode == 'dark' else '#1D2021'

        line_chart = px.line(
            data_frame=df_predictions,
            x="Date",
            y='Predicted total_dead_and_missing',
            color="Migration Route",
            line_shape="spline",
            markers=True,
            title="Predictions for Different Migration Routes in 2024 with Variation"
        ).update_layout(
            plot_bgcolor=plot_bgcolor,
            paper_bgcolor=paper_bgcolor,
            title_font=dict(color=text_color, size=20),
            xaxis=dict(
                title="Date",
                title_font=dict(color=text_color),
                tickfont=dict(color=text_color),
                linecolor=text_color,
                tickcolor=text_color,
                showgrid=True,
                gridcolor=grid_color
            ),
            yaxis=dict(
                title="Predicted total_dead_and_missing",
                title_font=dict(color=text_color),
                tickfont=dict(color=text_color),
                linecolor=text_color,
                tickcolor=text_color,
                showgrid=True,
                gridcolor=grid_color
            ),
            legend=dict(
                title_text='Migration Route',
                font=dict(color=text_color, size=12),
                bgcolor='rgba(0,0,0,0)' if mode == 'dark' else 'rgba(255,255,255,0.5)'
            )
        )
        
        return line_chart
        
    @reactive.Calc
    def filtered_df():
        region_filter = input.region_incident_select()
        time_range = input.year_range()

        # filtered = df[df["Region of Incident"] == region_filter] if region_filter else df
        filtered = df[(df['Region of Incident'].isin(input.region_incident_select()))]  

        current_year = df["Incident Year"].max()
        if time_range == "2-Year Range":
            year_range = [current_year - 1, current_year]
        elif time_range == "5-Year Range":
            year_range = [current_year - 4, current_year]
        else:
            year_range = [df["Incident Year"].min(), df["Incident Year"].max()]

        return filtered[(filtered["Incident Year"] >= year_range[0]) & (filtered["Incident Year"] <= year_range[1])]

    @render.data_frame
    def summary_statistics():
        data = filtered_df()
        grouped_data = data.groupby("Country of Origin")["Total Number of Dead and Missing"].sum()
        top_10 = grouped_data.sort_values(ascending=False).head(3)
        return top_10.reset_index()

    @render.data_frame
    def summary_statistics1():
        data = filtered_df()
        grouped_data = data.groupby("Migration Route")["Total Number of Dead and Missing"].sum()
        top_10 = grouped_data.sort_values(ascending=False).head(3)
        return top_10.reset_index()

    @render.data_frame
    def summary_statistics2():
        data = filtered_df()
        grouped_data = data.groupby("Country of Incident")["Total Number of Dead and Missing"].sum()
        top_10 = grouped_data.sort_values(ascending=False).head(3)
        return top_10.reset_index()

    @render.data_frame
    def summary_statistics3():
        data = filtered_df()
        grouped_data = data.groupby("Cause of Death")["Total Number of Dead and Missing"].sum()
        top_10 = grouped_data.sort_values(ascending=False).head(3)
        return top_10.reset_index()    
    
    # Must be debugged
    @reactive.event(input.btn_scrape)
    def scrape_source() -> df:

        
        def get_news() -> dict:
            
            """We will use a news API to get the latest news on migrants. 
            In main() we will then filter the articles to only include those with the words 'missing' or 'dead' in the description"""

            yesterday = datetime.now() - timedelta(days=1)
            yesterday = yesterday.strftime('%Y-%m-%d')
            last_week = datetime.now() - timedelta(days=7)

            api_key = os.getenv('NEWS_API_KEY') 
            params = {
                'q': 'migrants',
                'apiKey': 'api_key', # We hardcode the API key here for simplicity (for enhanced security use environment variables)
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

            openai_api_key = os.getenv('OPENAI_API_KEY')

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



        df_clean = pd.read_csv('clean_data.csv')
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
        # df_clean.to_csv('data/clean_data_merged.csv', index=False)
        
        return df_clean
    
    # Hardcode the data for the presentation 
    @render.data_frame
    @reactive.event(input.btn_scrape2)
    def scrape_source2() -> df: 
        df = pd.read_csv('news_data.csv')
        return df.reset_index()

app = App(app_ui, server)