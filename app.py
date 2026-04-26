import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Traffic Predictor", layout="centered")

st.title("🚦 Smart Traffic & Travel Time Predictor")

# ---------------------------
# Predefined Routes (Real Data)
# ---------------------------
routes = {
    ("Gurugram", "Rishikesh"): 250,
    ("Delhi", "Agra"): 230,
    ("Delhi", "Jaipur"): 280,
    ("Gurugram", "Manali"): 540
}

# Sample dataset
data = pd.DataFrame({
    'Source': ['Gurugram','Delhi','Delhi','Gurugram'],
    'Destination': ['Rishikesh','Agra','Jaipur','Manali'],
    'Time': [8, 14, 18, 20],
    'Day': ['Monday','Wednesday','Friday','Sunday'],
    'Weather': ['Clear','Rain','Cloudy','Clear'],
    'Distance': [250, 230, 280, 540],
    'Traffic': ['High','Medium','High','Low']
})

# ---------------------------
# Encoding
# ---------------------------
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

X = data[['Source','Destination','Time','Day','Weather','Distance']]
y = data['Traffic']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# ---------------------------
# User Input
# ---------------------------
st.header("📍 Enter Route Details")

source = st.selectbox("Select Source", list(set([i[0] for i in routes])))
destination = st.selectbox("Select Destination", list(set([i[1] for i in routes])))

time = st.slider("Time (Hour)", 0, 23, 8)

day = st.selectbox("Day", [
    'Monday','Tuesday','Wednesday',
    'Thursday','Friday','Saturday','Sunday'
])

weather = st.selectbox("Weather", ['Clear','Rain','Cloudy'])

# Get distance
distance = routes.get((source, destination), 200)

st.write(f"📏 Distance: {distance} km")

# Encode input
src_enc = le_source.transform([source])[0]
dest_enc = le_dest.transform([destination])[0]
day_enc = le_day.transform([day])[0]
weather_enc = le_weather.transform([weather])[0]

# ---------------------------
# Prediction
# ---------------------------
if st.button("🚀 Predict Traffic"):
    prediction = model.predict([[src_enc, dest_enc, time, day_enc, weather_enc, distance]])
    result = le_traffic.inverse_transform(prediction)

    # Travel time logic
    if result[0] == "Low":
        travel_time = f"{distance//60} hours"
    elif result[0] == "Medium":
        travel_time = f"{distance//50} hours"
    else:
        travel_time = f"{distance//40} hours"

    st.success(f"🚗 Traffic Level: {result[0]}")
    st.info(f"⏱️ Estimated Travel Time: {travel_time}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Made by You 😎 | Traffic Prediction Project")
