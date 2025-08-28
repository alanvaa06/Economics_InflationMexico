"""
Mexican Inflation Analysis Dashboard

Interactive dashboard for analyzing Mexican inflation data using Streamlit.
Provides comprehensive visualizations and statistics for inflation components.

Author: Alan
Date: 2024

Usage:
    streamlit run dashboard.py
"""

import sys
import os

def main():
    """Main function to run the dashboard."""
    
    # Import modules for Streamlit
    import function_module as fm
    
    # Configure page
    fm.st.set_page_config(
        page_title="Mexican Inflation Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Suppress warnings
    fm.warnings.filterwarnings('ignore')
    
    # Run dashboard
    dashboard = fm.InflationDashboard()
    dashboard.run_dashboard()


# Handle different execution methods
if __name__ == "__main__":
    # This runs when executed directly with python
    print("❌ Error: This is a Streamlit app!")
    print()
    print("📋 To run the dashboard, use one of these commands:")
    print("   streamlit run dashboard.py")
    print("   python -m streamlit run dashboard.py")
    print()
    print("💡 If you want to run the analysis without dashboard, use:")
    print("   python InflationAnalysis.py")
    print()
else:
    # This runs when imported by Streamlit
    main()