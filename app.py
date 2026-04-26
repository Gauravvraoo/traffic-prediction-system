import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.title("🚦 Route Traffic Prediction System")

# Sample dataset (route-based)
data = pd.DataFrame({
    'Source': ['A', 'A', 'B', 'B', 'C', 'C', 'A'],
    'Destination': ['B', 'C', 'C', 'A', 'A', 'B', 'B'],
    'Time': [8, 10, 14, 18, 20, 9, 17],
    'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    'Weather': ['Clear', 'Rain', 'Clear', 'Cloudy', 'Rain', 'Clear', 'Cloudy'],
    'Distance': [5, 10, 7, 6, 12, 4, 8],
    'Traffic': ['High', 'Medium', 'Low', 'High', 'High', 'Medium', 'Low']
})

# Encoding
le_source = LabelEncoder()
le_dest = LabelEncoder()
le_day = LabelEncoder()
le_weather = LabelEncoder()
le_traffic = LabelEncoder()

data['Source'] = le_source.fit_transform(data['Source'])
data['Destination'] = le_dest.fit_transform(data['Destination'])
data['Day'] = le_day.fit_transform(data['Day'])
data['Weather'] = le_weather.fit_transform(data['Weather'])
data['Traffic'] = le_traffic.fit_transform(data['Traffic'])

X = data[['Source', 'Destination', 'Time', 'Day', 'Weather', 'Distance']]
y = data['Traffic']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

st.header("Enter Route Details")

source = st.selectbox("Select Source", ['A', 'B', 'C'])
destination = st.selectbox("Select Destination", ['A', 'B', 'C'])

time = st.slider("Time (Hour)", 0, 23, 8)

day = st.selectbox("Day", [
    'Monday','Tuesday','Wednesday',
    'Thursday','Friday','Saturday','Sunday'
])

weather = st.selectbox("Weather", ['Clear', 'Rain', 'Cloudy'])

distance = st.slider("Distance (km)", 1, 20, 5)

# Encode input
src_enc = le_source.transform([source])[0]
dest_enc = le_dest.transform([destination])[0]
day_enc = le_day.transform([day])[0]
weather_enc = le_weather.transform([weather])[0]

if st.button("Predict Traffic"):
    prediction = model.predict([[src_enc, dest_enc, time, day_enc, weather_enc, distance]])
    result = le_traffic.inverse_transform(prediction)
    
    st.success(f"🚗 Traffic from {source} → {destination}: {result[0]}")
