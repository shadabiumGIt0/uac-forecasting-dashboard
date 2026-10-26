# 2. Write the app code directly to a file named app.py
%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="UAC System Dashboard", layout="wide")
st.title("📊 UAC Program Predictive Forecasting System Dashboard")
st.markdown("Running live inside Google Colab via localtunnel!")

# Create simple mock data for immediate validation
dates = pd.date_range(start="2026-01-01", periods=30, freq="D")
df = pd.DataFrame({
    "Date": dates,
    "Children in HHS Care": np.random.randint(9000, 11000, size=30)
}).set_index("Date")

st.line_chart(df)

# 3. Extract your public IP address (You will need to copy this paste value!)
import urllib
print("Password/Tunnel IP Requirement:", urllib.request.urlopen('https://icanhazip.com').read().decode('utf8').strip())

# 4. Boot the Streamlit app background server and create a public browser tunnel link
!npx localtunnel --port 8501 & streamlit run app.py &>/dev/null:
