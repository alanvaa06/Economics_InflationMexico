"""
Minimal working dashboard - bypassing problematic areas
"""

import streamlit as st
import function_module as fm

# Configure page
st.set_page_config(
    page_title="Mexican Inflation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("🇲🇽 Mexican Inflation Analysis Dashboard")
st.markdown("Interactive analysis of Mexican inflation components using INEGI data")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    api_key = st.text_input(
        "INEGI API Key",
        value="4f988b8a-9fe0-8498-a864-7d45e96af34f",
        type="password"
    )
    
    load_data = st.button("Load Data", type="primary")

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Load data if button pressed
if load_data:
    with st.spinner('Loading inflation data...'):
        try:
            analyzer = fm.InflationAnalyzer(api_key)
            analyzer.load_series_data()
            analyzer.load_weights()
            analyzer.prepare_final_data()
            analyzer.calculate_contributions()
            
            st.session_state.analyzer = analyzer
            st.session_state.data_loaded = True
            st.success("✅ Data loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.exception(e)

# Show content based on data availability
if not st.session_state.data_loaded:
    st.info("👈 Please load data using the sidebar to begin analysis")
else:
    analyzer = st.session_state.analyzer
    
    # Show data summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Series", len(analyzer.final_data.columns))
    
    with col2:
        st.metric("Date Range", f"{analyzer.final_data.index[0]} to {analyzer.final_data.index[-1]}")
    
    with col3:
        st.metric("Total Observations", len(analyzer.final_data))
    
    with col4:
        st.metric("Weight Categories", len(analyzer.weights_100.keys()))
    
    # Simple time series selection
    st.subheader("📈 Time Series Analysis")
    
    available_series = list(analyzer.final_data.columns)
    selected_series = st.selectbox("Select Time Series", available_series)
    
    if selected_series:
        data = analyzer.final_data[selected_series]
        st.line_chart(data)
        
        st.subheader("📊 Statistics")
        st.write(f"**Latest Value:** {data.iloc[-1]:.2f}")
        st.write(f"**Mean:** {data.mean():.2f}")
        st.write(f"**Standard Deviation:** {data.std():.2f}")

st.markdown("---")
st.markdown("🎉 If you see this interface working, the issue was in the complex dashboard logic!")

