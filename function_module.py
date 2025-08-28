"""
Function Module for Mexican Inflation Analysis

This module contains all classes and functions used in the Mexican Inflation Analysis project.
It includes the InflationAnalyzer class for data processing and the InflationDashboard class 
for interactive visualization.

Author: Extracted from InflationAnalysis.py and dashboard.py
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import calendar
import datetime as dt
from scipy.stats import percentileofscore
import warnings

# Dashboard specific imports
try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None
    px = None
    go = None
    print("⚠️  Streamlit not available. Dashboard functionality disabled.")

from datetime import datetime, timedelta
import io
import base64

# Suppress matplotlib deprecation warning
warnings.filterwarnings('ignore', category=UserWarning)
plt.style.use('seaborn-v0_8')


class InflationAnalyzer:
    """
    A comprehensive class for analyzing Mexican inflation data.
    
    This class handles data collection from INEGI's API, processes weights,
    calculates contributions, and generates visualizations for inflation analysis.
    """
    
    def __init__(self, api_key: str, series_file: str = 'SeriesInflation_ids.xlsx', 
                 weights_file: str = 'ponderadores.xlsx'):
        """
        Initialize the InflationAnalyzer.
        
        Args:
            api_key (str): INEGI API key
            series_file (str): Path to Excel file containing series IDs
            weights_file (str): Path to Excel file containing weights data
        """
        self.api_key = api_key
        self.series_file = series_file
        self.weights_file = weights_file
        
        # Data containers
        self.inflation_data = pd.DataFrame()
        self.final_data = pd.DataFrame()
        self.weights_raw = pd.DataFrame()
        self.weights_final = {}
        self.weights_100 = {}
        
        # Analysis results
        self.yoy_contributions = {}
        self.mom_contributions = {}
        
    def get_bie_data(self, serie: str, historic: str = 'false') -> pd.DataFrame:
        """
        Fetch data from INEGI's BIE API.
        
        Args:
            serie (str): Series ID to fetch
            historic (str): Whether to fetch historic data
            
        Returns:
            pd.DataFrame: DataFrame with series data
        """
        if isinstance(serie, int):
            serie = str(serie)
        
        url = (f'https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/'
               f'INDICATOR/{serie}/es/0700/{historic}/BIE/2.0/{self.api_key}?type=json')
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                content = json.loads(response.content)
                observations = content['Series'][0]['OBSERVATIONS']
                
                obs = []
                dates = []
                
                for observation in observations:
                    if observation['OBS_VALUE'] == '':
                        value = np.nan
                    else:
                        value = float(observation['OBS_VALUE'])
                    
                    obs.append(value)
                    dates.append(observation['TIME_PERIOD'])
                
                df = pd.DataFrame({'Serie': obs}, index=dates)
                index = pd.to_datetime(df.index, format='%Y/%m')
                df.index = [dt.date(y, m, calendar.monthrange(y, m)[1]) 
                           for y, m in zip(index.year, index.month)]
                df.sort_index(inplace=True)
                
                return df
            else:
                print(f'Error with API response for series {serie}')
                return pd.DataFrame()
                
        except Exception as e:
            print(f'Error processing series {serie}: {e}')
            return pd.DataFrame()
    
    def load_series_data(self) -> None:
        """Load series IDs and fetch all inflation data."""
        try:
            # Load series IDs
            i_tickers = pd.read_excel(self.series_file, index_col=0)
            series = i_tickers.Serie.to_list()
            
            print("Fetching inflation data from INEGI API...")
            
            # Fetch data for all series
            inflation_data = pd.DataFrame()
            for i, serie in enumerate(series):
                print(f"Fetching series {i+1}/{len(series)}: {serie}")
                df = self.get_bie_data(serie, historic='false')
                inflation_data = pd.concat([inflation_data, df], axis=1)
            
            inflation_data.columns = i_tickers.index
            inflation_data.index = pd.to_datetime(inflation_data.index, format='%Y-%m-%d')
            
            self.inflation_data = inflation_data
            print("Data fetching completed.")
            
        except Exception as e:
            print(f"Error loading series data: {e}")
    
    def load_weights(self) -> None:
        """Load and process weights data."""
        try:
            # Load weights
            weights = pd.read_excel(self.weights_file, index_col=0, 
                                  skiprows=9, sheet_name='ObjetoGasto')
            self.weights_raw = weights
            
            # Process weights for each category
            weights_final = {}
            weights_100 = {}
            
            for rubro in weights.columns[1:]:
                pesos = weights[['INPC', rubro]]
                pesos = pesos[pesos[rubro] == 'X']
                pesos = pesos[['INPC']] / 100
                
                weights_final[rubro] = pesos['INPC']
                weights_100[rubro] = pesos['INPC'] / pesos['INPC'].sum()
            
            self.weights_final = weights_final
            self.weights_100 = weights_100
            
            print("Weights processing completed.")
            
        except Exception as e:
            print(f"Error loading weights: {e}")
    
    def prepare_final_data(self) -> None:
        """Prepare final dataset for analysis."""
        try:
            final_data = pd.DataFrame()
            
            for ids in self.weights_raw.index.to_list():
                if ids in self.inflation_data.columns:
                    o = self.inflation_data[[ids]]
                    if len(o.columns) > 1:
                        o = o.iloc[:, 0]
                    final_data = pd.concat([final_data, o], axis=1)
            
            final_data.index = pd.to_datetime(final_data.index, format='%Y-%m-%d')
            
            # Filter from 2000 onwards
            final_data = final_data.loc['2000':]
            final_data.index = final_data.index.strftime("%Y-%b")
            
            self.final_data = final_data
            print("Final data preparation completed.")
            
        except Exception as e:
            print(f"Error preparing final data: {e}")
    
    def calculate_contributions(self) -> None:
        """Calculate YoY and MoM contributions for all categories."""
        try:
            yoy_contributions = {}
            mom_contributions = {}
            
            for key in self.weights_100.keys():
                weights = self.weights_100[key]
                items = weights.index.to_list()
                
                # Filter data for available items
                available_items = [item for item in items if item in self.final_data.columns]
                data = self.final_data[available_items]
                
                # Calculate contributions
                yoy_contributions[key] = (data.pct_change(12) * weights[available_items]).T
                mom_contributions[key] = (data.pct_change().fillna(0) * weights[available_items]).T
            
            self.yoy_contributions = yoy_contributions
            self.mom_contributions = mom_contributions
            
            print("Contribution calculations completed.")
            
        except Exception as e:
            print(f"Error calculating contributions: {e}")
    
    @staticmethod
    def percentile_score(df: pd.DataFrame, method: str = 'weak') -> pd.DataFrame:
        """
        Calculate percentile scores for the last data point in each series.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Percentile calculation method
            
        Returns:
            pd.DataFrame: Percentile scores
        """
        percentiles = []
        for col in df.columns:
            x = percentileofscore(df[col], df[col].iloc[-1], method) / 100
            percentiles.append(x)
        
        return pd.DataFrame(percentiles, index=df.columns)
    
    def calculate_percentiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling percentiles for time series data.
        
        Args:
            data (pd.DataFrame): Input time series data
            
        Returns:
            pd.DataFrame: Rolling percentiles
        """
        results = pd.DataFrame()
        
        for i, idx in enumerate(data.columns[:-12]):
            filter_data = data.T.iloc[i:(i + 12)]
            filter_data.fillna(0, inplace=True)
            percs = self.percentile_score(filter_data)
            percs = percs[0].sort_values(ascending=True)
            percs.index = np.arange(0, len(percs))
            results = pd.concat([results, percs], axis=1)
        
        results.columns = data.columns[12:]
        return results
    
    def plot_contribution_heatmap(self, category: str, title_suffix: str = "") -> None:
        """
        Plot contribution heatmap for a specific category.
        
        Args:
            category (str): Category to plot
            title_suffix (str): Additional title text
        """
        if category not in self.yoy_contributions:
            print(f"Category {category} not found in contributions.")
            return
        
        data = self.yoy_contributions[category]
        plt.figure(figsize=(20, 4))
        sns.heatmap(data, cmap='Blues')
        plt.title(f'{category} {title_suffix}')
        plt.show()
    
    def plot_percentile_heatmap(self, category: str) -> None:
        """
        Plot percentile heatmap for a specific category.
        
        Args:
            category (str): Category to plot
        """
        if category not in self.yoy_contributions:
            print(f"Category {category} not found in contributions.")
            return
        
        data = self.yoy_contributions[category]
        perc = self.calculate_percentiles(data)
        
        plt.figure(figsize=(20, 3))
        sns.heatmap(perc, cmap='Blues')
        plt.title(category)
        plt.yticks([])
        plt.grid(True)
        plt.show()
    
    def plot_all_percentile_heatmaps(self) -> None:
        """Plots percentile heatmaps for all available categories."""
        print("Generating percentile heatmaps for all categories...")
        for category in self.yoy_contributions.keys():
            self.plot_percentile_heatmap(category)

    def plot_top_contributors(self, top_n: int = 10) -> None:
        """
        Analyzes and plots the top and bottom contributors to YoY inflation for the most recent period.
        
        Args:
            top_n (int): The number of top and bottom contributors to display.
        """
        print(f"Analyzing top {top_n} contributors to inflation...")
        try:
            # Granular inflation calculation (from notebook cells 21-23)
            items_all = self.weights_final['IndiceGeneral'].index.to_list()
            selected = self.weights_raw['INPC'].index.to_list()
            componentes = [p for p in selected[1:] if p not in items_all]
            pesos_generales = self.weights_raw.loc[componentes, 'INPC']
            
            items_df = pd.DataFrame()
            for col in pesos_generales.index.to_list():
                if col in self.final_data.columns:
                    items_df = pd.concat([items_df, self.final_data[[col]]], axis=1)

            inflation_granular = items_df.pct_change(12).loc['2018-Jan':] * pesos_generales
            inflation_granular.dropna(how='all', axis=1, inplace=True)

            # Contribution analysis (from notebook cells 25-27)
            data = inflation_granular.iloc[-1].sort_values(ascending=False).dropna()
            data = data[data != 0]
            
            most = data.index[:top_n].to_list()
            least = data.index[-top_n:].to_list()
            
            incidents = pd.concat([data[most], data[least]], axis=0)
            as_total = incidents / data.sum()
            incidents = pd.concat([incidents, as_total], axis=1)
            incidents.columns=['Contributions', 'As_Total_100']
            incidents = incidents.sort_values(by='As_Total_100', ascending=False)
            
            most_least = most + least
            classification = ['Green' if x in self.weights_100['Total_Subyacente'].index else 'Blue' for x in most_least]
            
            # Plotting (from notebook cell 28)
            (100 * incidents['As_Total_100'].sort_values(ascending=True)).plot.barh(
                figsize=(12, 10), 
                title=f'Incidencias {inflation_granular.iloc[-1].name}',
                color=classification,
                alpha=0.5
            )
            plt.ylabel('Percentage contribution to Inflation', size=18)
            plt.yticks(size=14)
            plt.xticks(size=12)
            plt.show()

        except Exception as e:
            print(f"Could not generate top contributors plot: {e}")

    def analyze_core_inflation_components(self) -> tuple:
        """
        Analyze core inflation components (goods vs services).
        
        Returns:
            tuple: (goods_contribution, services_contribution, total_core)
        """
        try:
            # Core weights
            core_weight = self.weights_raw['Total_Subyacente']['IndiceGeneral']
            goods = self.weights_raw['Total_Mercancias_Subyacente']['IndiceGeneral']
            services = self.weights_raw['Total_Servicios']['IndiceGeneral']
            
            goods_weight = goods / core_weight
            services_weight = services / core_weight
            
            # Goods contribution
            goods_list = self.weights_raw[['INPC', 'Total_Mercancias_Subyacente']]
            goods_list = goods_list[goods_list['Total_Mercancias_Subyacente'] == 'X']['INPC']
            goods_list_100 = goods_list / goods_list.sum()
            goods_w_contribution = goods_list_100 * goods_weight
            
            # Services contribution
            services_list = self.weights_raw[['INPC', 'Total_Servicios']]
            services_list = services_list[services_list['Total_Servicios'] == 'X']['INPC']
            services_list_100 = services_list / services_list.sum()
            services_w_contribution = services_list_100 * services_weight
            
            # Calculate YoY contributions
            yoy_change_goods = self.final_data[goods_list_100.index].pct_change(12)
            goods_contribution = (yoy_change_goods.loc['2018-Jan':] * goods_w_contribution).sum(axis=1)
            
            yoy_change_services = self.final_data[services_list_100.index].pct_change(12)
            services_contribution = (yoy_change_services.loc['2018-Jan':] * services_w_contribution).sum(axis=1)
            
            total_core = pd.concat([goods_contribution, services_contribution], axis=1)
            total_core.columns = ['Goods', 'Services']
            
            return goods_contribution, services_contribution, total_core
            
        except Exception as e:
            print(f"Error analyzing core inflation components: {e}")
            return None, None, None
    
    def plot_core_components(self) -> None:
        """Plot core inflation components (goods vs services)."""
        _, _, total_core = self.analyze_core_inflation_components()
        
        if total_core is not None:
            (total_core * 100).plot.bar(stacked=True, figsize=(18, 6), 
                                       color=['#004481', '#2DCCCD'])
            plt.title('Contributions - Goods & Services to Core Inflation', size=16)
            plt.legend(bbox_to_anchor=(1.1, 0.85))
            plt.xticks(size=12)
            plt.yticks(size=14)
            plt.show()

    def analyze_goods_inflation_components(self) -> pd.DataFrame:
        """
        Analyzes goods inflation components (food vs non-food).
        
        Returns:
            pd.DataFrame: DataFrame with food and non-food contributions.
        """
        try:
            # Weights (from notebook cells 42-43)
            goods_total = self.weights_raw['Total_Mercancias_Subyacente']['IndiceGeneral']
            food = self.weights_raw['AlimentosBebidasTabaco_Mercancias_Subyacente']['IndiceGeneral']
            nonfood = self.weights_raw['NoAlimenticias_Mercacias_Subyacente']['IndiceGeneral']
            
            food_weight = food / goods_total
            nonfood_weight = nonfood / goods_total

            # Food contribution (from notebook cell 44)
            food_list = self.weights_raw[self.weights_raw['AlimentosBebidasTabaco_Mercancias_Subyacente'] == 'X']['INPC']
            food_list_100 = food_list / food_list.sum()
            food_w_contribution = food_list_100 * food_weight
            
            # Non-food contribution (from notebook cell 45)
            nonfood_list = self.weights_raw[self.weights_raw['NoAlimenticias_Mercacias_Subyacente'] == 'X']['INPC']
            nonfood_list_100 = nonfood_list / nonfood_list.sum()
            nonfood_w_contribution = nonfood_list_100 * nonfood_weight
            
            # Calculate YoY contributions (from notebook cells 46-48)
            yoy_change_food = self.final_data[food_list_100.index].pct_change(12)
            food_contribution_goods = (yoy_change_food.loc['2018-Jan':] * food_w_contribution).sum(axis=1)
            
            yoy_change_nonfood = self.final_data[nonfood_list_100.index].pct_change(12)
            nonfood_contribution_goods = (yoy_change_nonfood.loc['2018-Jan':] * nonfood_w_contribution).sum(axis=1)
            
            total_goods = pd.concat([food_contribution_goods, nonfood_contribution_goods], axis=1)
            total_goods.columns = ['Food', 'NonFood']
            return total_goods
            
        except Exception as e:
            print(f"Error analyzing goods inflation components: {e}")
            return pd.DataFrame()

    def plot_goods_components(self) -> None:
        """Plots goods inflation components (food vs non-food)."""
        total_goods = self.analyze_goods_inflation_components()
        if not total_goods.empty:
            (total_goods * 100).plot.bar(stacked=True, figsize=(18, 6), color=['#004481', '#2DCCCD'])
            plt.title('Contributions - Food & Non-Food to Goods Inflation', size=16)
            plt.legend(bbox_to_anchor=(1.1, 0.85))
            plt.xticks(size=12)
            plt.yticks(size=14)
            plt.show()

    def analyze_services_inflation_components(self) -> pd.DataFrame:
        """
        Analyzes services inflation components (education, housing, others).
        
        Returns:
            pd.DataFrame: DataFrame with services component contributions.
        """
        try:
            # Weights (from notebook cells 52-53)
            services_total = self.weights_raw['Total_Servicios']['IndiceGeneral']
            education = self.weights_raw['Educacion_Servicios_Subyacente']['IndiceGeneral']
            housing = self.weights_raw['Vivienda_Servicios_Subyacente']['IndiceGeneral']
            others = self.weights_raw['Otros_Servicios_Subyacente']['IndiceGeneral']

            education_weight = education / services_total
            housing_weight = housing / services_total
            others_weight = others / services_total

            # Component contributions (from notebook cells 54-56)
            education_list = self.weights_raw[self.weights_raw['Educacion_Servicios_Subyacente'] == 'X']['INPC']
            education_w_contribution = (education_list / education_list.sum()) * education_weight

            housing_list = self.weights_raw[self.weights_raw['Vivienda_Servicios_Subyacente'] == 'X']['INPC']
            housing_w_contribution = (housing_list / housing_list.sum()) * housing_weight

            others_list = self.weights_raw[self.weights_raw['Otros_Servicios_Subyacente'] == 'X']['INPC']
            others_w_contribution = (others_list / others_list.sum()) * others_weight

            # Calculate YoY contributions (from notebook cells 57-59)
            yoy_change_edu = self.final_data[education_list.index].pct_change(12)
            education_contrib = (yoy_change_edu.loc['2018-Jan':] * education_w_contribution).sum(axis=1)

            yoy_change_housing = self.final_data[housing_list.index].pct_change(12)
            housing_contrib = (yoy_change_housing.loc['2018-Jan':] * housing_w_contribution).sum(axis=1)

            yoy_change_others = self.final_data[others_list.index].pct_change(12)
            others_contrib = (yoy_change_others.loc['2018-Jan':] * others_w_contribution).sum(axis=1)

            total_services = pd.concat([education_contrib, housing_contrib, others_contrib], axis=1)
            total_services.columns = ['Education', 'Housing', 'Others']
            return total_services

        except Exception as e:
            print(f"Error analyzing services inflation components: {e}")
            return pd.DataFrame()

    def plot_services_components(self) -> None:
        """Plots services inflation components."""
        total_services = self.analyze_services_inflation_components()
        if not total_services.empty:
            (total_services * 100).plot.bar(stacked=True, figsize=(18, 6), color=['#004481', '#2DCCCD', '#D8BE75'])
            plt.title('Contributions - Services Components to Services Inflation', size=16)
            plt.legend(bbox_to_anchor=(1.1, 0.85))
            plt.xticks(size=12)
            plt.yticks(size=14)
            plt.show()

    def plot_combined_core_components(self) -> None:
        """Plots a combined view of all core inflation sub-components."""
        total_goods = self.analyze_goods_inflation_components()
        total_services = self.analyze_services_inflation_components()
        
        if not total_goods.empty and not total_services.empty:
            total = pd.concat([total_services, total_goods], axis=1)
            (total * 100).plot.bar(stacked=True, figsize=(18, 6), color=['#004481', '#2DCCCD', '#1973B8', '#A5A5A5', '#D8BE75'])
            plt.title('Contributions - Core Inflation Components', size=16)
            plt.legend(bbox_to_anchor=(1.1, 0.85))
            plt.xticks(size=12)
            plt.yticks(size=14)
            plt.show()

    def identify_outliers(self, df: pd.DataFrame, window: int = 36, 
                         threshold: float = 2) -> pd.DataFrame:
        """
        Identify outliers using expanding window approach.
        
        Args:
            df (pd.DataFrame): Input dataframe
            window (int): Window size for outlier detection
            threshold (float): Z-score threshold
            
        Returns:
            pd.DataFrame: Boolean dataframe indicating outliers
        """
        # Suppress the performance warning for this specific operation
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 
                                  message='DataFrame is highly fragmented', 
                                  category=pd.errors.PerformanceWarning)
            
            # Use a dictionary to collect outlier series, then create DataFrame at once
            outliers_dict = {}
            
            for column in df.columns:
                if df[column].isna().any():
                    continue
                
                outliers = pd.Series(index=df.index, dtype=bool, name=column)
                outliers[:] = False  # Initialize all as False
                
                for i in range(1, window + 1):
                    if len(df) < i:
                        continue
                        
                    current_window = df[column][:-i]
                    if len(current_window) == 0:
                        continue
                        
                    mean = current_window.mean()
                    std = current_window.std()
                    
                    if std == 0:
                        outliers.iloc[-i] = False
                    else:
                        point = df[column].iloc[-i]
                        z_score = (point - mean) / std
                        outliers.iloc[-i] = np.abs(z_score) > threshold
                
                # Only keep the last 'window' observations
                outliers_dict[column] = outliers[-window:]
            
            # Create DataFrame from dictionary all at once to avoid fragmentation
            if outliers_dict:
                outliers_df = pd.DataFrame(outliers_dict)
                return outliers_df.dropna()
            else:
                return pd.DataFrame()
    
    def save_data(self, full_file: str = 'FullInflation.xlsx', 
                  relevant_file: str = 'RelevantInflation.xlsx') -> None:
        """
        Save processed data to Excel files.
        
        Args:
            full_file (str): Filename for full inflation data
            relevant_file (str): Filename for relevant inflation data
        """
        try:
            if not self.inflation_data.empty:
                self.inflation_data.to_excel(full_file)
                print(f"Full inflation data saved to {full_file}")
            
            if not self.final_data.empty:
                self.final_data.to_excel(relevant_file)
                print(f"Relevant inflation data saved to {relevant_file}")
                
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def run_full_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        print("Starting full inflation analysis...")
        
        # Load and process data
        self.load_series_data()
        self.load_weights()
        self.prepare_final_data()
        self.calculate_contributions()
        
        # Save processed data
        self.save_data()
        
        print("Analysis completed successfully!")
        
        # Generate some key visualizations
        self.generate_all_visualizations()
        
    def _calculate_pct_proportion(self, data: pd.DataFrame) -> pd.DataFrame:
        """Helper to calculate proportion of items within YoY change ranges."""
        results = pd.DataFrame()
        for i in data.index:
            line = data.loc[i].dropna()
            a = line[line < 0.02].count()
            b = line[(line >= 0.02) & (line < 0.04)].count()
            c = line[(line >= 0.04) & (line < 0.06)].count()
            d = line[(line >= 0.06) & (line < 0.08)].count()
            e = line[line >= 0.08].count()
            proportions = pd.DataFrame([a, b, c, d, e], columns=[i], index=['<2%', '2-4%', '4-6%', '6-8%', '>8%'])
            if len(line.index) > 0:
                proportions /= len(line.index)
            results = pd.concat([results, proportions], axis=1)
        return results

    def plot_inflation_distribution_area_charts(self) -> None:
        """Plots area charts showing the distribution of inflation components among YoY ranges."""
        print("Generating inflation distribution area charts...")
        
        # Define better colors for the 5 categories
        colors = ['#004481', '#2DCCCD', '#1973B8', '#A5A5A5', '#D8BE75']
        
        for key in list(self.weights_100.keys()):
            items = self.weights_100[key].index.to_list()
            available_items = [item for item in items if item in self.final_data.columns]
            
            if not available_items:
                print(f"No available items for category: {key}")
                continue

            # Calculate YoY changes
            yoy_changes = self.final_data[available_items].pct_change(12).loc['2018-Jan':]
            
            # Skip if no data
            if yoy_changes.empty:
                print(f"No YoY data for category: {key}")
                continue
                
            results = self._calculate_pct_proportion(yoy_changes)
            
            if results.empty or results.shape[1] == 0:
                print(f"No proportion results for category: {key}")
                continue

            try:
                # Create the plot
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Transpose and convert to percentage
                plot_data = (results.T * 100).fillna(0)
                
                # Ensure we have proper date index
                if hasattr(plot_data.index, 'strftime'):
                    # If already datetime, format it
                    plot_data.index = pd.to_datetime(plot_data.index).strftime('%Y-%m')
                else:
                    # Convert string dates to proper format
                    try:
                        plot_data.index = pd.to_datetime(plot_data.index).strftime('%Y-%m')
                    except:
                        pass  # Keep original index if conversion fails
                
                # Create stacked area plot
                plot_data.plot.area(ax=ax, stacked=True, color=colors[:len(plot_data.columns)], alpha=0.8)
                
                # Formatting
                ax.set_title(f'% of Inflation Components among YoY ranges - {key}', fontsize=16, pad=20)
                ax.set_xlabel('Date', fontsize=14)
                ax.set_ylabel('Percentage of Components', fontsize=14)
                
                # Format legend
                ax.legend(title='YoY Rate Ranges', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
                
                # Format x-axis
                ax.tick_params(axis='x', rotation=45, labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                
                # Set y-axis limits
                ax.set_ylim(0, 100)
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Tight layout
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Error plotting {key}: {e}")
                continue
            
    def generate_all_visualizations(self) -> None:
        """Generates and displays all visualizations from the analysis."""
        print("\n--- Generating All Visualizations ---")
        
        # Contribution heatmaps
        self.plot_contribution_heatmap('IndiceGeneral', title_suffix="YoY Contributions")
        
        # Percentile heatmaps for all categories
        self.plot_all_percentile_heatmaps()
        
        # Top/bottom contributors
        self.plot_top_contributors()
        
        # Core inflation component breakdowns
        self.plot_core_components() # Goods vs Services
        self.plot_goods_components() # Food vs Non-Food
        self.plot_services_components() # Education, Housing, Others
        self.plot_combined_core_components() # All core components combined
        
        # Inflation distribution area charts
        self.plot_inflation_distribution_area_charts()
        
        # Outlier analysis plot
        print("\nAnalyzing outliers...")
        if not self.final_data.empty:
            yoy_changes = self.final_data.pct_change(12).loc['2018-Jan':]
            outliers = self.identify_outliers(yoy_changes)
            if not outliers.empty:
                outlier_percentage = (outliers.sum(1) / len(outliers.columns) * 100)
                outlier_percentage.plot(kind='bar', figsize=(16, 6), 
                                      title='Percentage of Outliers in Mexican Inflation Components')
                plt.show()
        print("\n--- Visualization Generation Complete ---")


class InflationDashboard:
    """
    A comprehensive dashboard for Mexican inflation analysis.
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit is required for dashboard functionality. Install with: pip install streamlit")
        self.analyzer = None
        self.data_loaded = False
        
    @st.cache_data
    def load_data(_self, api_key: str):
        """
        Load and process inflation data (cached for performance).
        
        Args:
            api_key (str): INEGI API key
        """
        try:
            analyzer = InflationAnalyzer(api_key)
            
            # Show progress
            with st.spinner('Loading inflation data from INEGI API...'):
                analyzer.load_series_data()
                analyzer.load_weights()
                analyzer.prepare_final_data()
                analyzer.calculate_contributions()
            
            return analyzer
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def show_data_summary(self, analyzer):
        """Display data summary statistics."""
        if analyzer.final_data.empty:
            st.warning("No data available")
            return
            
        st.subheader("📊 Data Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Series", len(analyzer.final_data.columns))
        
        with col2:
            st.metric("Date Range", f"{analyzer.final_data.index[0]} to {analyzer.final_data.index[-1]}")
        
        with col3:
            st.metric("Total Observations", len(analyzer.final_data))
        
        with col4:
            st.metric("Weight Categories", len(analyzer.weights_100.keys()))
    
    def plot_time_series(self, data: pd.Series, title: str):
        """
        Create interactive time series plot.
        
        Args:
            data (pd.Series): Time series data
            title (str): Plot title
        """
        fig = go.Figure()
        
        # Convert index to datetime if it's not already
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                x_values = pd.to_datetime(data.index, format='%Y-%b')
            except:
                x_values = data.index
        else:
            x_values = data.index
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=data.values,
            mode='lines',
            name=title,
            line=dict(width=2, color='#004481')
        ))
        
        fig.update_layout(
            title=f"Time Series: {title}",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_histogram(self, data: pd.Series, title: str):
        """
        Create histogram plot.
        
        Args:
            data (pd.Series): Data for histogram
            title (str): Plot title
        """
        fig = px.histogram(
            x=data.dropna().values,
            nbins=30,
            title=f"Distribution: {title}",
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title="Value",
            yaxis_title="Frequency",
            height=400
        )
        
        return fig
    
    def plot_yoy_changes(self, data: pd.Series, title: str):
        """
        Create Year-over-Year changes plot.
        
        Args:
            data (pd.Series): Time series data
            title (str): Plot title
        """
        yoy_data = data.pct_change(12) * 100  # Convert to percentage
        
        fig = go.Figure()
        
        # Convert index to datetime if it's not already
        if not isinstance(yoy_data.index, pd.DatetimeIndex):
            try:
                x_values = pd.to_datetime(yoy_data.index, format='%Y-%b')
            except:
                x_values = yoy_data.index
        else:
            x_values = yoy_data.index
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=yoy_data.values,
            mode='lines',
            name='YoY Change (%)',
            line=dict(width=2, color='#2DCCCD')
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        
        fig.update_layout(
            title=f"Year-over-Year Changes: {title}",
            xaxis_title="Date",
            yaxis_title="YoY Change (%)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def calculate_statistics(self, data: pd.Series):
        """
        Calculate comprehensive statistics for a time series.
        
        Args:
            data (pd.Series): Time series data
            
        Returns:
            dict: Dictionary of statistics
        """
        clean_data = data.dropna()
        yoy_changes = data.pct_change(12).dropna()
        
        stats = {
            'count': len(clean_data),
            'mean': clean_data.mean(),
            'median': clean_data.median(),
            'std': clean_data.std(),
            'min': clean_data.min(),
            'max': clean_data.max(),
            'q25': clean_data.quantile(0.25),
            'q75': clean_data.quantile(0.75),
            'latest_value': clean_data.iloc[-1] if len(clean_data) > 0 else None,
            'yoy_latest': yoy_changes.iloc[-1] * 100 if len(yoy_changes) > 0 else None,
            'yoy_mean': yoy_changes.mean() * 100 if len(yoy_changes) > 0 else None,
            'yoy_std': yoy_changes.std() * 100 if len(yoy_changes) > 0 else None
        }
        
        return stats
    
    def show_statistics_table(self, stats: dict):
        """Display statistics in a formatted table."""
        stats_df = pd.DataFrame([
            ['Count', f"{stats['count']:,}"],
            ['Mean', f"{stats['mean']:.2f}"],
            ['Median', f"{stats['median']:.2f}"],
            ['Std Dev', f"{stats['std']:.2f}"],
            ['Minimum', f"{stats['min']:.2f}"],
            ['Maximum', f"{stats['max']:.2f}"],
            ['25th Percentile', f"{stats['q25']:.2f}"],
            ['75th Percentile', f"{stats['q75']:.2f}"],
            ['Latest Value', f"{stats['latest_value']:.2f}"],
            ['Latest YoY Change (%)', f"{stats['yoy_latest']:.2f}" if stats['yoy_latest'] else "N/A"],
            ['Avg YoY Change (%)', f"{stats['yoy_mean']:.2f}" if stats['yoy_mean'] else "N/A"],
            ['YoY Volatility (%)', f"{stats['yoy_std']:.2f}" if stats['yoy_std'] else "N/A"]
        ], columns=['Statistic', 'Value'])
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    def plot_contribution_analysis(self, analyzer, category: str):
        """
        Create contribution analysis visualization.
        
        Args:
            analyzer: InflationAnalyzer instance
            category (str): Category to analyze
        """
        if category not in analyzer.yoy_contributions:
            st.warning(f"No contribution data available for {category}")
            return
        
        contrib_data = analyzer.yoy_contributions[category]
        
        # Get the latest contributions
        latest_contrib = contrib_data.iloc[:, -1].sort_values(ascending=True)
        
        # Create horizontal bar chart
        fig = px.bar(
            x=latest_contrib.values,
            y=latest_contrib.index,
            orientation='h',
            title=f"Latest Contributions: {category}",
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title="Contribution",
            yaxis_title="Component",
            height=max(400, len(latest_contrib) * 20)
        )
        
        return fig
    
    def create_correlation_matrix(self, analyzer, selected_series: list):
        """
        Create correlation matrix for selected series.
        
        Args:
            analyzer: InflationAnalyzer instance
            selected_series (list): List of selected series
        """
        if len(selected_series) < 2:
            st.warning("Please select at least 2 series for correlation analysis")
            return
        
        # Get data for selected series
        corr_data = analyzer.final_data[selected_series].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_data,
            title="Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto',
            template='plotly_white'
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def export_data(self, analyzer, selected_series: str):
        """
        Export data functionality.
        
        Args:
            analyzer: InflationAnalyzer instance
            selected_series (str): Selected series name
        """
        if selected_series and selected_series in analyzer.final_data.columns:
            data = analyzer.final_data[selected_series].dropna()
            
            # Create download link
            csv = data.to_csv()
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{selected_series}.csv">Download {selected_series} data as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    def run_dashboard(self):
        """Main dashboard application."""
        st.title("🇲🇽 Mexican Inflation Analysis Dashboard")
        st.markdown("Interactive analysis of Mexican inflation components using INEGI data")
        
        # Sidebar configuration
        st.sidebar.header("Configuration")
        
        # API Key input
        api_key = st.sidebar.text_input(
            "INEGI API Key",
            value="4f988b8a-9fe0-8498-a864-7d45e96af34f",
            type="password",
            help="Enter your INEGI API key"
        )
        
        # Load data button
        if st.sidebar.button("Load Data", type="primary"):
            self.analyzer = self.load_data(api_key)
            if self.analyzer:
                self.data_loaded = True
                st.success("Data loaded successfully!")
            else:
                st.error("Failed to load data")
                return
        
        # Check if data is loaded
        if not self.data_loaded or not self.analyzer:
            st.info("👈 Please load data using the sidebar to begin analysis")
            return
        
        # Show data summary
        self.show_data_summary(self.analyzer)
        
        # Check if analyzer has data
        if self.analyzer.final_data.empty:
            st.warning("⚠️ No data available. Please load data first.")
            return
        
        # Get available series for all tabs
        available_series = list(self.analyzer.final_data.columns)
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Time Series Analysis", 
            "📊 Statistics", 
            "🔄 Contributions", 
            "🔗 Correlations",
            "📁 Data Export"
        ])
        
        # Time Series Analysis Tab
        with tab1:
            st.header("Time Series Analysis")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Series selection
                selected_series = st.selectbox(
                    "Select Time Series",
                    available_series,
                    help="Choose a time series to analyze"
                )
                
                # Analysis options
                show_yoy = st.checkbox("Show YoY Changes", value=True)
                show_histogram = st.checkbox("Show Distribution", value=True)
            
            with col2:
                if selected_series:
                    data = self.analyzer.final_data[selected_series]
                    
                    # Time series plot
                    fig_ts = self.plot_time_series(data, selected_series)
                    st.plotly_chart(fig_ts, use_container_width=True)
                    
                    # Additional plots
                    if show_yoy or show_histogram:
                        subcol1, subcol2 = st.columns(2)
                        
                        if show_yoy:
                            with subcol1:
                                fig_yoy = self.plot_yoy_changes(data, selected_series)
                                st.plotly_chart(fig_yoy, use_container_width=True)
                        
                        if show_histogram:
                            with subcol2:
                                fig_hist = self.plot_histogram(data, selected_series)
                                st.plotly_chart(fig_hist, use_container_width=True)
        
        # Statistics Tab
        with tab2:
            st.header("Statistical Analysis")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_series_stats = st.selectbox(
                    "Select Series for Statistics",
                    available_series,
                    key="stats_series"
                )
            
            with col2:
                if selected_series_stats:
                    data = self.analyzer.final_data[selected_series_stats]
                    stats = self.calculate_statistics(data)
                    
                    st.subheader(f"Statistics: {selected_series_stats}")
                    self.show_statistics_table(stats)
        
        # Contributions Tab
        with tab3:
            st.header("Contribution Analysis")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Category selection
                available_categories = list(self.analyzer.yoy_contributions.keys())
                selected_category = st.selectbox(
                    "Select Category",
                    available_categories,
                    help="Choose a category for contribution analysis"
                )
            
            with col2:
                if selected_category:
                    fig_contrib = self.plot_contribution_analysis(self.analyzer, selected_category)
                    if fig_contrib:
                        st.plotly_chart(fig_contrib, use_container_width=True)
        
        # Correlations Tab
        with tab4:
            st.header("Correlation Analysis")
            
            # Multi-select for series
            selected_for_corr = st.multiselect(
                "Select Series for Correlation Analysis",
                available_series,
                default=available_series[:5] if len(available_series) >= 5 else available_series,
                help="Select multiple series to analyze correlations"
            )
            
            if len(selected_for_corr) >= 2:
                fig_corr = self.create_correlation_matrix(self.analyzer, selected_for_corr)
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True)
        
        # Data Export Tab
        with tab5:
            st.header("Data Export")
            
            col1, col2 = st.columns(2)
            
            with col1:
                export_series = st.selectbox(
                    "Select Series to Export",
                    available_series,
                    key="export_series"
                )
                
                if st.button("Generate Download Link"):
                    self.export_data(self.analyzer, export_series)
            
            with col2:
                st.info("""
                **Export Options:**
                - Individual time series as CSV
                - Statistical summaries
                - Contribution data
                """)
