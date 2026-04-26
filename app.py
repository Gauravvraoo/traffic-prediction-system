import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.title("🚦 Traffic Prediction System")

# Sample dataset (you can replace with CSV later)
data = pd.DataFrame({
    'Time': [8, 9, 14, 18, 20],
    'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
    'Weather': ['Clear', 'Rain', 'Clear', 'Cloudy', 'Rain'],
    'Vehicles': [120, 80, 60, 150, 200],
    'Traffic': ['High', 'Medium', 'Low', 'High', 'High']
})

# Encoding
le_day = LabelEncoder()
le_weather = LabelEncoder()
le_traffic = LabelEncoder()

data['Day'] = le_day.fit_transform(data['Day'])
data['Weather'] = le_weather.fit_transform(data['Weather'])
data['Traffic'] = le_traffic.fit_transform(data['Traffic'])

X = data[['Time', 'Day', 'Weather', 'Vehicles']]
y = data['Traffic']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

st.header("Enter Traffic Details")

time = st.slider("Time (Hour)", 0, 23, 8)
day = st.selectbox("Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
weather = st.selectbox("Weather", ['Clear', 'Rain', 'Cloudy'])
vehicles = st.number_input("Number of Vehicles", 0, 500, 100)

# Convert input
day_encoded = le_day.transform([day])[0]
weather_encoded = le_weather.transform([weather])[0]

if st.button("Predict Traffic"):
    prediction = model.predict([[time, day_encoded, weather_encoded, vehicles]])
    result = le_traffic.inverse_transform(prediction)
    
    st.success(f"Predicted Traffic Level: {result[0]}")
