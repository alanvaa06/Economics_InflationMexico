#!/usr/bin/env python3
"""
Mexican Inflation Analysis Tool

This script performs a comprehensive analysis of Mexican inflation data using INEGI's API.
It implements methodologies for data collection, processing, contribution calculation,
percentile analysis, and outlier detection.

Methodologies reference: Inflation_Analysis.ipynb
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
import os
import argparse
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inflation_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# Set plot style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')

class InflationAnalyzer:
    """
    A class for analyzing Mexican inflation data.
    Handles data extraction, transformation, and visualization generation.
    """

    def __init__(self, api_key: str, output_dir: str = 'output',
                 series_file: str = 'SeriesInflation_ids.xlsx',
                 weights_file: str = 'ponderadores.xlsx'):
        """
        Initialize the InflationAnalyzer.

        Args:
            api_key (str): INEGI API key
            output_dir (str): Directory to save outputs
            series_file (str): Path to Excel file containing series IDs
            weights_file (str): Path to Excel file containing weights data
        """
        self.api_key = api_key
        self.output_dir = output_dir
        self.series_file = series_file
        self.weights_file = weights_file

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)

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
                logger.error(f'API returned status code {response.status_code} for series {serie}')
                return pd.DataFrame()

        except Exception as e:
            logger.error(f'Error processing series {serie}: {e}')
            return pd.DataFrame()

    def load_series_data(self) -> None:
        """Load series IDs and fetch all inflation data."""
        try:
            logger.info(f"Loading series IDs from {self.series_file}")
            i_tickers = pd.read_excel(self.series_file, index_col=0)
            series = i_tickers.Serie.to_list()

            logger.info("Fetching inflation data from INEGI API...")

            # Fetch data for all series
            data_list = []
            total_series = len(series)

            for i, serie in enumerate(series):
                if (i + 1) % 10 == 0:
                    logger.info(f"Fetching series {i+1}/{total_series}")
                df = self.get_bie_data(serie, historic='false')
                if not df.empty:
                    # Rename the column to match the ticker index name
                    df.columns = [i_tickers.index[i]]
                    data_list.append(df)

            if data_list:
                inflation_data = pd.concat(data_list, axis=1)
                inflation_data.index = pd.to_datetime(inflation_data.index, format='%Y-%m-%d')
                self.inflation_data = inflation_data
                logger.info("Data fetching completed.")
            else:
                logger.warning("No data fetched from API.")
                self.inflation_data = pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading series data: {e}")
            raise

    def load_weights(self) -> None:
        """Load and process weights data."""
        try:
            logger.info(f"Loading weights from {self.weights_file}")
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

            logger.info("Weights processing completed.")

        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            raise

    def prepare_final_data(self) -> None:
        """Prepare final dataset for analysis."""
        try:
            logger.info("Preparing final dataset...")
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
            logger.info("Final data preparation completed.")

        except Exception as e:
            logger.error(f"Error preparing final data: {e}")
            raise

    def calculate_contributions(self) -> None:
        """Calculate YoY and MoM contributions for all categories."""
        try:
            logger.info("Calculating contributions...")
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

            logger.info("Contribution calculations completed.")

        except Exception as e:
            logger.error(f"Error calculating contributions: {e}")
            raise

    @staticmethod
    def percentile_score(df: pd.DataFrame, method: str = 'weak') -> pd.DataFrame:
        """Calculate percentile scores for the last data point in each series."""
        percentiles = []
        for col in df.columns:
            x = percentileofscore(df[col], df[col].iloc[-1], method) / 100
            percentiles.append(x)

        return pd.DataFrame(percentiles, index=df.columns)

    def calculate_percentiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling percentiles for time series data."""
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

    def save_contribution_heatmap(self, category: str, title_suffix: str = "") -> None:
        """Save contribution heatmap for a specific category."""
        if category not in self.yoy_contributions:
            logger.warning(f"Category {category} not found in contributions.")
            return

        data = self.yoy_contributions[category]
        plt.figure(figsize=(20, 4))
        sns.heatmap(data, cmap='Blues')
        plt.title(f'{category} {title_suffix}')

        filename = f"heatmap_contribution_{category}.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved contribution heatmap: {filename}")

    def save_percentile_heatmap(self, category: str) -> None:
        """Save percentile heatmap for a specific category."""
        if category not in self.yoy_contributions:
            logger.warning(f"Category {category} not found in contributions.")
            return

        data = self.yoy_contributions[category]
        perc = self.calculate_percentiles(data)

        plt.figure(figsize=(20, 3))
        sns.heatmap(perc, cmap='Blues')
        plt.title(category)
        plt.yticks([])
        plt.grid(True)

        filename = f"heatmap_percentile_{category}.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved percentile heatmap: {filename}")

    def save_top_contributors(self, top_n: int = 10) -> None:
        """Analyzes and saves the top and bottom contributors to YoY inflation."""
        logger.info(f"Analyzing top {top_n} contributors to inflation...")
        try:
            # Granular inflation calculation
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

            # Contribution analysis
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

            # Plotting
            plt.figure(figsize=(12, 10))
            (100 * incidents['As_Total_100'].sort_values(ascending=True)).plot.barh(
                title=f'Incidencias {inflation_granular.iloc[-1].name}',
                color=classification,
                alpha=0.5
            )
            plt.ylabel('Percentage contribution to Inflation', size=18)
            plt.yticks(size=14)
            plt.xticks(size=12)

            filename = "top_contributors.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved top contributors plot: {filename}")

        except Exception as e:
            logger.error(f"Could not generate top contributors plot: {e}")

    def analyze_core_inflation_components(self) -> tuple:
        """Analyze core inflation components (goods vs services)."""
        try:
            core_weight = self.weights_raw['Total_Subyacente']['IndiceGeneral']
            goods = self.weights_raw['Total_Mercancias_Subyacente']['IndiceGeneral']
            services = self.weights_raw['Total_Servicios']['IndiceGeneral']

            goods_weight = goods / core_weight
            services_weight = services / core_weight

            goods_list = self.weights_raw[['INPC', 'Total_Mercancias_Subyacente']]
            goods_list = goods_list[goods_list['Total_Mercancias_Subyacente'] == 'X']['INPC']
            goods_list_100 = goods_list / goods_list.sum()
            goods_w_contribution = goods_list_100 * goods_weight

            services_list = self.weights_raw[['INPC', 'Total_Servicios']]
            services_list = services_list[services_list['Total_Servicios'] == 'X']['INPC']
            services_list_100 = services_list / services_list.sum()
            services_w_contribution = services_list_100 * services_weight

            yoy_change_goods = self.final_data[goods_list_100.index].pct_change(12)
            goods_contribution = (yoy_change_goods.loc['2018-Jan':] * goods_w_contribution).sum(axis=1)

            yoy_change_services = self.final_data[services_list_100.index].pct_change(12)
            services_contribution = (yoy_change_services.loc['2018-Jan':] * services_w_contribution).sum(axis=1)

            total_core = pd.concat([goods_contribution, services_contribution], axis=1)
            total_core.columns = ['Goods', 'Services']

            return goods_contribution, services_contribution, total_core

        except Exception as e:
            logger.error(f"Error analyzing core inflation components: {e}")
            return None, None, None

    def save_core_components(self) -> None:
        """Save core inflation components plot."""
        _, _, total_core = self.analyze_core_inflation_components()

        if total_core is not None:
            plt.figure(figsize=(18, 6))
            (total_core * 100).plot.bar(stacked=True, figsize=(18, 6),
                                       color=['#004481', '#2DCCCD'])
            plt.title('Contributions - Goods & Services to Core Inflation', size=16)
            plt.legend(bbox_to_anchor=(1.1, 0.85))
            plt.xticks(size=12)
            plt.yticks(size=14)

            filename = "core_components.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved core components plot: {filename}")

    def analyze_goods_inflation_components(self) -> pd.DataFrame:
        """Analyzes goods inflation components (food vs non-food)."""
        try:
            goods_total = self.weights_raw['Total_Mercancias_Subyacente']['IndiceGeneral']
            food = self.weights_raw['AlimentosBebidasTabaco_Mercancias_Subyacente']['IndiceGeneral']
            nonfood = self.weights_raw['NoAlimenticias_Mercacias_Subyacente']['IndiceGeneral']

            food_weight = food / goods_total
            nonfood_weight = nonfood / goods_total

            food_list = self.weights_raw[self.weights_raw['AlimentosBebidasTabaco_Mercancias_Subyacente'] == 'X']['INPC']
            food_list_100 = food_list / food_list.sum()
            food_w_contribution = food_list_100 * food_weight

            nonfood_list = self.weights_raw[self.weights_raw['NoAlimenticias_Mercacias_Subyacente'] == 'X']['INPC']
            nonfood_list_100 = nonfood_list / nonfood_list.sum()
            nonfood_w_contribution = nonfood_list_100 * nonfood_weight

            yoy_change_food = self.final_data[food_list_100.index].pct_change(12)
            food_contribution_goods = (yoy_change_food.loc['2018-Jan':] * food_w_contribution).sum(axis=1)

            yoy_change_nonfood = self.final_data[nonfood_list_100.index].pct_change(12)
            nonfood_contribution_goods = (yoy_change_nonfood.loc['2018-Jan':] * nonfood_w_contribution).sum(axis=1)

            total_goods = pd.concat([food_contribution_goods, nonfood_contribution_goods], axis=1)
            total_goods.columns = ['Food', 'NonFood']
            return total_goods

        except Exception as e:
            logger.error(f"Error analyzing goods inflation components: {e}")
            return pd.DataFrame()

    def save_goods_components(self) -> None:
        """Save goods inflation components plot."""
        total_goods = self.analyze_goods_inflation_components()
        if not total_goods.empty:
            plt.figure(figsize=(18, 6))
            (total_goods * 100).plot.bar(stacked=True, figsize=(18, 6), color=['#004481', '#2DCCCD'])
            plt.title('Contributions - Food & Non-Food to Goods Inflation', size=16)
            plt.legend(bbox_to_anchor=(1.1, 0.85))
            plt.xticks(size=12)
            plt.yticks(size=14)

            filename = "goods_components.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved goods components plot: {filename}")

    def analyze_services_inflation_components(self) -> pd.DataFrame:
        """Analyzes services inflation components."""
        try:
            services_total = self.weights_raw['Total_Servicios']['IndiceGeneral']
            education = self.weights_raw['Educacion_Servicios_Subyacente']['IndiceGeneral']
            housing = self.weights_raw['Vivienda_Servicios_Subyacente']['IndiceGeneral']
            others = self.weights_raw['Otros_Servicios_Subyacente']['IndiceGeneral']

            education_weight = education / services_total
            housing_weight = housing / services_total
            others_weight = others / services_total

            education_list = self.weights_raw[self.weights_raw['Educacion_Servicios_Subyacente'] == 'X']['INPC']
            education_w_contribution = (education_list / education_list.sum()) * education_weight

            housing_list = self.weights_raw[self.weights_raw['Vivienda_Servicios_Subyacente'] == 'X']['INPC']
            housing_w_contribution = (housing_list / housing_list.sum()) * housing_weight

            others_list = self.weights_raw[self.weights_raw['Otros_Servicios_Subyacente'] == 'X']['INPC']
            others_w_contribution = (others_list / others_list.sum()) * others_weight

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
            logger.error(f"Error analyzing services inflation components: {e}")
            return pd.DataFrame()

    def save_services_components(self) -> None:
        """Save services inflation components plot."""
        total_services = self.analyze_services_inflation_components()
        if not total_services.empty:
            plt.figure(figsize=(18, 6))
            (total_services * 100).plot.bar(stacked=True, figsize=(18, 6), color=['#004481', '#2DCCCD', '#D8BE75'])
            plt.title('Contributions - Services Components to Services Inflation', size=16)
            plt.legend(bbox_to_anchor=(1.1, 0.85))
            plt.xticks(size=12)
            plt.yticks(size=14)

            filename = "services_components.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved services components plot: {filename}")

    def save_combined_core_components(self) -> None:
        """Save combined core inflation sub-components plot."""
        total_goods = self.analyze_goods_inflation_components()
        total_services = self.analyze_services_inflation_components()

        if not total_goods.empty and not total_services.empty:
            total = pd.concat([total_services, total_goods], axis=1)
            plt.figure(figsize=(18, 6))
            (total * 100).plot.bar(stacked=True, figsize=(18, 6), color=['#004481', '#2DCCCD', '#1973B8', '#A5A5A5', '#D8BE75'])
            plt.title('Contributions - Core Inflation Components', size=16)
            plt.legend(bbox_to_anchor=(1.1, 0.85))
            plt.xticks(size=12)
            plt.yticks(size=14)

            filename = "combined_core_components.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved combined core components plot: {filename}")

    def identify_outliers(self, df: pd.DataFrame, window: int = 36,
                         threshold: float = 2) -> pd.DataFrame:
        """Identify outliers using expanding window approach."""
        outliers_dict = {}

        for column in df.columns:
            if df[column].isna().any():
                continue

            outliers = pd.Series(index=df.index, dtype=bool, name=column)
            outliers[:] = False

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

            outliers_dict[column] = outliers[-window:]

        if outliers_dict:
            outliers_df = pd.DataFrame(outliers_dict)
            return outliers_df.dropna()
        else:
            return pd.DataFrame()

    def save_outlier_analysis(self) -> None:
        """Analyze and save outliers plot."""
        logger.info("Analyzing outliers...")
        if not self.final_data.empty:
            yoy_changes = self.final_data.pct_change(12).loc['2018-Jan':]
            outliers = self.identify_outliers(yoy_changes)
            if not outliers.empty:
                plt.figure(figsize=(16, 6))
                outlier_percentage = (outliers.sum(1) / len(outliers.columns) * 100)
                outlier_percentage.plot(kind='bar', figsize=(16, 6),
                                      title='Percentage of Outliers in Mexican Inflation Components')

                filename = "outlier_analysis.png"
                filepath = os.path.join(self.plots_dir, filename)
                plt.savefig(filepath, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved outlier analysis plot: {filename}")

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

    def save_inflation_distribution_area_charts(self) -> None:
        """Save area charts showing the distribution of inflation components."""
        logger.info("Generating inflation distribution area charts...")

        colors = ['#004481', '#2DCCCD', '#1973B8', '#A5A5A5', '#D8BE75']

        # Skip the first key (usually IndiceGeneral) as per original notebook logic
        keys_to_plot = list(self.weights_100.keys())[1:]

        for key in keys_to_plot:
            items = self.weights_100[key].index.to_list()
            available_items = [item for item in items if item in self.final_data.columns]

            if not available_items:
                continue

            yoy_changes = self.final_data[available_items].pct_change(12).loc['2018-Jan':]

            if yoy_changes.empty:
                continue

            results = self._calculate_pct_proportion(yoy_changes)

            if results.empty or results.shape[1] == 0:
                continue

            try:
                fig, ax = plt.subplots(figsize=(14, 8))

                plot_data = (results.T * 100).fillna(0)

                if hasattr(plot_data.index, 'strftime'):
                    plot_data.index = pd.to_datetime(plot_data.index).strftime('%Y-%m')
                else:
                    try:
                        plot_data.index = pd.to_datetime(plot_data.index).strftime('%Y-%m')
                    except:
                        pass

                plot_data.plot.area(ax=ax, stacked=True, color=colors[:len(plot_data.columns)], alpha=0.8)

                ax.set_title(f'% of Inflation Components among YoY ranges - {key}', fontsize=16, pad=20)
                ax.set_xlabel('Date', fontsize=14)
                ax.set_ylabel('Percentage of Components', fontsize=14)

                ax.legend(title='YoY Rate Ranges', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

                ax.tick_params(axis='x', rotation=45, labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()

                filename = f"distribution_area_{key}.png"
                filepath = os.path.join(self.plots_dir, filename)
                plt.savefig(filepath, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved distribution area chart: {filename}")

            except Exception as e:
                logger.error(f"Error plotting {key}: {e}")
                continue

    def save_data(self, full_file: str = 'FullInflation.xlsx',
                  relevant_file: str = 'RelevantInflation.xlsx') -> None:
        """Save processed data to Excel files."""
        try:
            full_path = os.path.join(self.output_dir, full_file)
            relevant_path = os.path.join(self.output_dir, relevant_file)

            if not self.inflation_data.empty:
                self.inflation_data.to_excel(full_path)
                logger.info(f"Full inflation data saved to {full_path}")

            if not self.final_data.empty:
                self.final_data.to_excel(relevant_path)
                logger.info(f"Relevant inflation data saved to {relevant_path}")

        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def run_full_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        logger.info("Starting full inflation analysis...")

        self.load_series_data()
        self.load_weights()
        self.prepare_final_data()
        self.calculate_contributions()

        self.save_data()

        logger.info("Generating visualizations...")
        self.save_contribution_heatmap('IndiceGeneral', title_suffix="YoY Contributions")

        for category in self.yoy_contributions.keys():
            self.save_percentile_heatmap(category)

        self.save_top_contributors()
        self.save_core_components()
        self.save_goods_components()
        self.save_services_components()
        self.save_combined_core_components()
        self.save_inflation_distribution_area_charts()
        self.save_outlier_analysis()

        logger.info("Analysis completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Mexican Inflation Analysis Tool')
    parser.add_argument('--api_key', type=str, help='INEGI API Key',
                        default=os.environ.get('INEGI_API_KEY', '4f988b8a-9fe0-8498-a864-7d45e96af34f'))
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--series_file', type=str, default='SeriesInflation_ids.xlsx', help='Series IDs Excel file')
    parser.add_argument('--weights_file', type=str, default='ponderadores.xlsx', help='Weights Excel file')

    args = parser.parse_args()

    if not os.path.exists(args.series_file):
        logger.error(f"Series file not found: {args.series_file}")
        return

    if not os.path.exists(args.weights_file):
        logger.error(f"Weights file not found: {args.weights_file}")
        return

    analyzer = InflationAnalyzer(
        api_key=args.api_key,
        output_dir=args.output_dir,
        series_file=args.series_file,
        weights_file=args.weights_file
    )

    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
