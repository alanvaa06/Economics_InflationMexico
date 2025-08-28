"""
Debug version of the dashboard to identify the exact issue
"""

import streamlit as st

st.title("🔍 Debug Dashboard")

try:
    st.write("Step 1: Importing function_module...")
    import function_module as fm
    st.success("✅ function_module imported successfully")
    
    st.write("Step 2: Creating InflationDashboard...")
    dashboard = fm.InflationDashboard()
    st.success("✅ InflationDashboard created successfully")
    
    st.write("Step 3: Testing basic Streamlit elements...")
    st.sidebar.header("Test Sidebar")
    st.success("✅ Sidebar working")
    
    st.write("Step 4: Testing analyzer creation...")
    analyzer = fm.InflationAnalyzer('test-key')
    st.success("✅ InflationAnalyzer created successfully")
    
    st.write("Step 5: Testing run_dashboard method...")
    
    # This is where the issue likely occurs
    dashboard.run_dashboard()
    
except Exception as e:
    st.error(f"❌ Error at step: {e}")
    st.exception(e)
    
    # Let's see the exact traceback
    import traceback
    st.text("Full traceback:")
    st.code(traceback.format_exc())

