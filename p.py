import numpy as np
import pandas as pd
import folium
import streamlit as st
import time
import joblib
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh

# Load or train ML model for leak detection
try:
    model, scaler = joblib.load("leak_detection_model.pkl")
except:
    model = IsolationForest(contamination=0.01, random_state=42)  # Lower contamination to reduce false positives
    scaler = StandardScaler()
    # Generate initial training data
    training_data = np.random.normal(loc=[50, 100, 30], scale=[5, 10, 3], size=(1000, 3))
    training_data = scaler.fit_transform(training_data)  # Normalize data
    model.fit(training_data)
    joblib.dump((model, scaler), "leak_detection_model.pkl")

# Define pipeline segments (coordinates)
pipeline_segments = [
    {'name': 'Guwahati → Dimapur', 'start': (26.1445, 91.7362), 'end': (25.9063, 93.7276)},
    {'name': 'Dimapur → Kohima', 'start': (25.9063, 93.7276), 'end': (25.6747, 94.1086)},
    {'name': 'Kohima → Imphal', 'start': (25.6747, 94.1086), 'end': (24.817, 93.9368)}
]

# Generate intermediate pipeline points
def interpolate_points(start, end, num_points=5):
    lat_points = np.linspace(start[0], end[0], num_points)
    lon_points = np.linspace(start[1], end[1], num_points)
    return list(zip(lat_points, lon_points))

# Simulate sensor data at multiple pipeline points
def generate_sensor_data():
    data = []
    for segment in pipeline_segments:
        points = interpolate_points(segment['start'], segment['end'])
        for point in points:
            data.append({
                'segment': segment['name'],
                'location': point,
                'pressure': np.random.normal(50, 5),
                'flow': np.random.normal(100, 10),
                'acoustic': np.random.normal(30, 3),
                'time': time.time()
            })
    return pd.DataFrame(data)

# Leak detection using ML model
def detect_leak(df):
    required_columns = ['pressure', 'flow', 'acoustic']
    if not all(col in df.columns for col in required_columns):
        st.error("Missing necessary columns in the sensor data.")
        return pd.DataFrame()  # Return empty DataFrame if necessary columns are missing

    df[['pressure_z', 'flow_z', 'acoustic_z']] = scaler.transform(df[['pressure', 'flow', 'acoustic']])
    df['anomaly'] = model.predict(df[['pressure_z', 'flow_z', 'acoustic_z']]) == -1
    
    # Add timestamp for detected anomalies
    df['timestamp'] = datetime.now()
    
    return df[df['anomaly']]

# Save sensor data to a CSV file
def save_sensor_data_to_csv(sensor_data):
    file_path = "sensor_data.csv"
    if not os.path.exists(file_path):
        sensor_data.to_csv(file_path, index=False, mode='w', header=True)
    else:
        sensor_data.to_csv(file_path, index=False, mode='a', header=False)

# Save anomalies to a CSV file
def save_anomalies_to_csv(anomalies):
    file_path = "detected_leaks.csv"
    if not os.path.exists(file_path):
        anomalies.to_csv(file_path, index=False, mode='w', header=True)
    else:
        anomalies.to_csv(file_path, index=False, mode='a', header=False)

# Filter anomalies by date (daily, weekly, monthly)
def filter_anomalies_by_date(period='daily'):
    anomalies = pd.read_csv("detected_leaks.csv")
    anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'])
    current_date = datetime.now()
    
    if period == 'daily':
        filtered_anomalies = anomalies[anomalies['timestamp'].dt.date == current_date.date()]
    elif period == 'weekly':
        start_of_week = current_date - timedelta(days=current_date.weekday())
        filtered_anomalies = anomalies[anomalies['timestamp'] >= start_of_week]
    elif period == 'monthly':
        filtered_anomalies = anomalies[anomalies['timestamp'].dt.month == current_date.month]
    
    return filtered_anomalies

# Streamlit UI improvements
st.set_page_config(page_title="Pipeline Leak Detection", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f5f5f5;}
        h1 {color: #2c3e50; text-align: center; font-size: 2.5rem; font-family: 'Arial', sans-serif;}
        .stButton>button {background-color: #3498db; color: white; border-radius: 10px; padding: 10px;}
        .stCheckbox>div>label {font-size: 16px; color: #34495e; font-family: 'Arial', sans-serif;}
        .stDataFrame {width: 100%;}
        .map-container {height: 400px; width: 100%;}
        .stBlockquote {font-size: 1.1em; font-style: italic;}
        .stSidebar {background-color: #ecf0f1; padding: 15px;}
        .stSidebar .sidebar-content {margin-top: 50px;}
    </style>
""", unsafe_allow_html=True)

# Project Overview Section
st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2>🚀 Real-Time Pipeline Leak Detection</h2>
        <p>Monitor pipeline conditions and detect leaks in real time. This system uses machine learning to analyze sensor data (pressure, flow, and acoustic measurements) and identifies anomalies that could indicate a potential leak. Stay informed with up-to-date reports and alerts.</p>
    </div>
""", unsafe_allow_html=True)

# Manual refresh and auto-refresh options
refresh = st.button("🔄 Refresh Data")
auto_refresh = st.checkbox("Auto Refresh Every 30s", value=True)

if auto_refresh:
    st_autorefresh(interval=30000, key="auto_refresh")

# Store the last updated time and data
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = pd.DataFrame()
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = pd.DataFrame()  # To store detected leaks

# Check if enough time has passed to update the data
time_diff = time.time() - st.session_state.last_update_time
if time_diff > 30:  # Update data every 30 seconds
    st.session_state.sensor_data = generate_sensor_data()  # Generate new sensor data
    save_sensor_data_to_csv(st.session_state.sensor_data)  # Save sensor data to CSV file
    st.session_state.last_update_time = time.time()  # Update the last update time

# Get the latest sensor data
sensor_data = st.session_state.sensor_data

# Detect leaks based on the latest sensor data
anomalies = detect_leak(sensor_data)

# Store anomalies in session state
if not anomalies.empty:
    st.session_state.anomalies = anomalies  # Store detected leaks in session state
    save_anomalies_to_csv(anomalies)  # Save anomalies to CSV file

# Create static map visualization with pipeline routes
pipeline_map = folium.Map(location=[25.5, 93.0], zoom_start=7)  # Adjusted zoom for Northeast India

# Plot pipeline routes (Static)
for segment in pipeline_segments:
    points = interpolate_points(segment['start'], segment['end'])
    folium.PolyLine(points, color='blue', weight=5).add_to(pipeline_map)

# Add markers for leak points
for _, row in st.session_state.anomalies.iterrows():
    folium.Marker(
        location=row['location'], 
        icon=folium.Icon(color='red'), 
        popup=f"Leak at {row['location']}"
    ).add_to(pipeline_map)

# Layout with columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Live Sensor Data")
    if not sensor_data.empty:
        st.dataframe(sensor_data[['segment', 'location', 'pressure', 'flow', 'acoustic']])
    else:
        st.warning("No sensor data available yet.")
    
    if not st.session_state.anomalies.empty:
        st.warning("⚠️ Leak detected at the following locations:")
        st.dataframe(st.session_state.anomalies[['segment', 'location', 'pressure', 'flow', 'acoustic']])
        st.error("🚨 ALERT! Possible Leak Detected!")
    else:
        st.success("✅ No leaks detected.")

with col2:
    st.subheader("🗺️ Pipeline Map")
    # Display the smaller map with possible leak markers
    st_folium(pipeline_map, width=700, height=400, key="pipeline_map_static")

# Report Generation Section
st.sidebar.subheader("Generate Leak Detection Report")
report_period = st.sidebar.radio("Select Report Period", ["Daily", "Weekly", "Monthly"])

if st.sidebar.button("Generate Report"):
    # Filter anomalies based on the selected report period
    filtered_anomalies = filter_anomalies_by_date(period=report_period.lower())
    if not filtered_anomalies.empty:
        st.subheader(f"Detected Leaks for {report_period} Report")
        st.dataframe(filtered_anomalies[['segment', 'location', 'pressure', 'flow', 'acoustic', 'timestamp']])
    else:
        st.warning(f"No leaks detected for the {report_period.lower()} period.")

