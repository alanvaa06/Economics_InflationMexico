"""
Mexican Inflation Analysis Module

This module provides a comprehensive analysis of Mexican inflation data using INEGI's API.
It includes data collection, processing, visualization, and statistical analysis of inflation components.

Author: Refactored from Jupyter Notebook
Date: 2024
"""

import function_module as fm

#%%
def main():
    """Main function to demonstrate usage."""
    # Example usage
    API_KEY = '4f988b8a-9fe0-8498-a864-7d45e96af34f'
    
    # Initialize analyzer
    analyzer = fm.InflationAnalyzer(API_KEY)
    
    # Run full analysis
    analyzer.run_full_analysis()
    
    # Example of additional analysis - this is now included in generate_all_visualizations
    # print("\nAdditional analysis examples:")
    # # Analyze outliers
    # if not analyzer.final_data.empty:
    #     yoy_changes = analyzer.final_data.pct_change(12).loc['2018-Jan':]
    #     outliers = analyzer.identify_outliers(yoy_changes)
        
    #     if not outliers.empty:
    #         outlier_percentage = (outliers.sum(1) / len(outliers.columns) * 100)
    #         outlier_percentage.plot(kind='bar', figsize=(16, 6), 
    #                               title='Percentage of outliers in Mexican inflation components')
    #         plt.show()


if __name__ == "__main__":
    main()
#%%

analyzer = fm.InflationAnalyzer('4f988b8a-9fe0-8498-a864-7d45e96af34f')
analyzer.plot_inflation_distribution_area_charts()