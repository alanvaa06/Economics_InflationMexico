# 🇲🇽 Mexican Inflation Dashboard

An interactive web dashboard for analyzing Mexican inflation data using INEGI's API.

## 🚀 Features

### **📊 Main Components:**
- **Time Series Analysis**: Interactive plots with YoY changes and distributions
- **Statistical Analysis**: Comprehensive statistics for each inflation component
- **Contribution Analysis**: Visual breakdown of inflation contributors by category
- **Correlation Analysis**: Correlation matrix for multiple time series
- **Data Export**: Download data as CSV files

### **🎛️ Interactive Controls:**
- **Dropdown Menus**: Select specific time series for analysis
- **Checkboxes**: Toggle different visualizations on/off
- **Multi-select**: Choose multiple series for correlation analysis
- **Export Options**: Download individual series data

## 📋 Requirements

Install the required dependencies:

```bash
pip install streamlit plotly pandas numpy matplotlib seaborn requests scipy openpyxl
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## 🏃 How to Run

### **Method 1: Direct Streamlit Command**
```bash
streamlit run dashboard.py
```

### **Method 2: Python Command**
```bash
python -m streamlit run dashboard.py
```

### **Method 3: After Installation**
If you've installed the project:
```bash
inflation-dashboard
```

## 🎯 Dashboard Usage

### **1. Initial Setup**
1. Open the dashboard in your browser (usually `http://localhost:8501`)
2. Enter your INEGI API key in the sidebar
3. Click "Load Data" to fetch the latest inflation data

### **2. Navigation**
The dashboard has 5 main tabs:

#### **📈 Time Series Analysis**
- Select any inflation component from the dropdown
- View interactive time series plots
- Toggle YoY changes and distribution histograms
- Zoom and pan on charts for detailed analysis

#### **📊 Statistics**
- Choose a series to view comprehensive statistics
- See measures like mean, median, volatility
- Latest values and YoY changes
- Quartile information

#### **🔄 Contributions**
- Select inflation categories (Core, Non-Core, etc.)
- View contribution breakdown charts
- Identify top/bottom contributors
- Analyze component impacts

#### **🔗 Correlations**
- Select multiple time series
- Generate correlation matrix heatmap
- Identify relationships between components
- Interactive correlation visualization

#### **📁 Data Export**
- Choose specific series to export
- Download data as CSV files
- Export statistical summaries
- Save for external analysis

## 🎨 Visualization Types

### **Time Series Plots**
- Interactive line charts with zoom/pan
- Date range selection
- Hover tooltips with exact values
- YoY change overlays

### **Statistical Charts**
- Histograms showing data distribution
- Box plots for quartile analysis
- Trend analysis with moving averages

### **Contribution Analysis**
- Horizontal bar charts for latest contributions
- Stacked area charts for historical contributions
- Color-coded by inflation category

### **Correlation Matrix**
- Interactive heatmap
- Hover for exact correlation values
- Color-coded correlation strength

## 🔧 Technical Features

### **Performance**
- **Caching**: Data is cached for faster subsequent loads
- **Lazy Loading**: Charts load only when needed
- **Responsive**: Works on desktop and mobile devices

### **Interactivity**
- **Plotly Integration**: Fully interactive charts
- **Real-time Updates**: Instant response to user selections
- **Export Functionality**: Download charts and data

### **Error Handling**
- Graceful handling of API failures
- Data validation and cleaning
- User-friendly error messages

## 📊 Data Sources

- **INEGI API**: Official Mexican inflation data
- **Series Coverage**: All major inflation components
- **Frequency**: Monthly data
- **Historical Range**: From 2000 onwards

## 🎛️ Customization

### **Adding New Visualizations**
1. Create new methods in the `InflationDashboard` class
2. Add new tabs in the `run_dashboard` method
3. Integrate with existing data structures

### **Styling**
- Modify Plotly templates for different themes
- Adjust color schemes in chart functions
- Customize Streamlit layout and styling

## 🐛 Troubleshooting

### **Common Issues**

**Dashboard won't start:**
```bash
# Check if Streamlit is installed
pip show streamlit

# Reinstall if needed
pip install streamlit --upgrade
```

**Data loading fails:**
- Check your INEGI API key
- Verify internet connection
- Check API rate limits

**Charts not displaying:**
- Ensure Plotly is properly installed
- Check browser compatibility
- Clear browser cache

### **Performance Issues**
- Use data caching (`@st.cache_data`)
- Limit number of series in correlation analysis
- Consider data sampling for large datasets

## 📈 Future Enhancements

- **Real-time Data**: Automatic data updates
- **More Visualizations**: Additional chart types
- **Advanced Analytics**: Forecasting and modeling
- **User Authentication**: Multi-user support
- **Database Integration**: Persistent data storage

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add new dashboard features
4. Test thoroughly
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details
