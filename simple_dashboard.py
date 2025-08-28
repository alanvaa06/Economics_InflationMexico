"""
Simple test dashboard to verify basic functionality
"""

import streamlit as st

# Configure page
st.set_page_config(
    page_title="Test Dashboard",
    page_icon="🧪",
    layout="wide"
)

# Main content
st.title("🇲🇽 Mexican Inflation Analysis Dashboard")
st.markdown("Testing basic dashboard functionality...")

# Test sidebar
with st.sidebar:
    st.header("Test Controls")
    test_option = st.selectbox("Test Selection", ["Option 1", "Option 2", "Option 3"])
    
if st.button("Test Dashboard Loading"):
    st.success("✅ Dashboard is working!")
    st.info("📊 Ready to load inflation analysis")
    
    # Test importing our module
    try:
        import function_module as fm
        st.success("✅ Function module imported successfully")
        
        # Test creating analyzer (without API call)
        st.info("Testing InflationAnalyzer creation...")
        analyzer = fm.InflationAnalyzer('test-key')
        st.success("✅ InflationAnalyzer created successfully")
        
        # Test creating dashboard class
        st.info("Testing InflationDashboard creation...")
        dashboard = fm.InflationDashboard()
        st.success("✅ InflationDashboard created successfully")
        
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)

st.markdown("---")
st.markdown("If you can see this page, Streamlit is working correctly!")

