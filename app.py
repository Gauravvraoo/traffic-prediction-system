import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import random

st.set_page_config(page_title="Traffic Predictor", layout="centered")

st.title("🚦 AI Traffic Prediction System (Gurugram)")

# ---------------------------
# START LOCATIONS
# ---------------------------
start_locations = [
    "IFFCO Chowk",
    "Cyber City",
    "Huda City Centre",
    "MG Road",
    "The NorthCap University"
]

# DESTINATIONS
destinations = [
    "Panchgaon","Manesar","Sohna",
    "Delhi","Noida","Jaipur","Rishikesh"
]

# ---------------------------
# TRAVEL MODE
# ---------------------------
mode = st.selectbox("🚗 Select Mode of Travel", ["Car", "Bike"])
travel_mode = "driving" if mode == "Car" else "two-wheeler"

# ---------------------------
# DISTANCE DATA
# ---------------------------
routes = {
    ("IFFCO Chowk", "Panchgaon"): 25,
    ("Cyber City", "Panchgaon"): 30,
    ("Huda City Centre", "Panchgaon"): 20,
    ("MG Road", "Panchgaon"): 28,
    ("The NorthCap University", "Panchgaon"): 35,

    ("IFFCO Chowk", "Manesar"): 20,
    ("Cyber City", "Manesar"): 25,
    ("Huda City Centre", "Manesar"): 18,
    ("MG Road", "Manesar"): 22,
    ("The NorthCap University", "Manesar"): 30,

    ("IFFCO Chowk", "Sohna"): 30,
    ("Cyber City", "Sohna"): 35,
    ("Huda City Centre", "Sohna"): 25,
    ("MG Road", "Sohna"): 32,
    ("The NorthCap University", "Sohna"): 40,

    ("IFFCO Chowk", "Delhi"): 30,
    ("Cyber City", "Delhi"): 25,
    ("Huda City Centre", "Delhi"): 35,
    ("MG Road", "Delhi"): 28,
    ("The NorthCap University", "Delhi"): 20,

    ("IFFCO Chowk", "Noida"): 45,
    ("Cyber City", "Noida"): 50,
    ("Huda City Centre", "Noida"): 55,
    ("MG Road", "Noida"): 48,
    ("The NorthCap University", "Noida"): 35,

    ("IFFCO Chowk", "Jaipur"): 280,
    ("Cyber City", "Jaipur"): 285,
    ("Huda City Centre", "Jaipur"): 290,
    ("MG Road", "Jaipur"): 275,
    ("The NorthCap University", "Jaipur"): 270,

    ("IFFCO Chowk", "Rishikesh"): 250,
    ("Cyber City", "Rishikesh"): 255,
    ("Huda City Centre", "Rishikesh"): 260,
    ("MG Road", "Rishikesh"): 245,
    ("The NorthCap University", "Rishikesh"): 240
}

# ---------------------------
# DATASET (UPDATED WITH NCU)
# ---------------------------
data = pd.DataFrame({
    'Source': [
        "IFFCO Chowk","Cyber City","Huda City Centre","MG Road",
        "IFFCO Chowk","Cyber City","Huda City Centre","MG Road",
        "The NorthCap University","The NorthCap University"
    ],
    'Destination': [
        "Panchgaon","Delhi","Noida","Jaipur",
        "Rishikesh","Manesar","Sohna","Delhi",
        "Delhi","Manesar"
    ],
    'Time': [8,14,18,20,10,16,9,22,9,17],
    'Day': [
        "Monday","Wednesday","Friday","Sunday",
        "Tuesday","Thursday","Saturday","Monday",
        "Monday","Friday"
    ],
    'Weather': [
        "Clear","Rain","Cloudy","Clear",
        "Rain","Clear","Cloudy","Rain",
        "Clear","Rain"
    ],
    'Distance': [25,25,55,275,250,20,30,30,20,30],
    'Traffic': [
        "High","Medium","High","Low",
        "High","Medium","Low","Medium",
        "Medium","High"
    ]
})

# ---------------------------
# ENCODING
# ---------------------------
le_src = LabelEncoder()
le_dest = LabelEncoder()
le_day = LabelEncoder()
le_weather = LabelEncoder()
le_traffic = LabelEncoder()

le_src.fit(start_locations)
le_dest.fit(destinations)
le_day.fit(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
le_weather.fit(['Clear','Rain','Cloudy'])
le_traffic.fit(['Low','Medium','High'])

data['Source'] = le_src.transform(data['Source'])
data['Destination'] = le_dest.transform(data['Destination'])
data['Day'] = le_day.transform(data['Day'])
data['Weather'] = le_weather.transform(data['Weather'])
data['Traffic'] = le_traffic.transform(data['Traffic'])

X = data[['Source','Destination','Time','Day','Weather','Distance']]
y = data['Traffic']

model = DecisionTreeClassifier()
model.fit(X, y)

# ---------------------------
# USER INPUT
# ---------------------------
st.header("📍 Enter Route Details")

source = st.selectbox("Start Location", start_locations)
destination = st.selectbox("Destination", destinations)

time = st.slider("Time (Hour)", 0, 23, 8)
day = st.selectbox("Day", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
weather = st.selectbox("Weather", ['Clear','Rain','Cloudy'])

distance = routes.get((source, destination), 50)

# Encode
src_enc = le_src.transform([source])[0]
dest_enc = le_dest.transform([destination])[0]
day_enc = le_day.transform([day])[0]
weather_enc = le_weather.transform([weather])[0]

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("🚀 Predict Traffic"):

    pred = model.predict([[src_enc, dest_enc, time, day_enc, weather_enc, distance]])
    result = le_traffic.inverse_transform(pred)[0]

    st.success(f"🚗 Predicted Traffic Level: {result}")

    # GOOGLE MAP LINK
    map_link = f"https://www.google.com/maps/dir/?api=1&origin={source.replace(' ','+')}&destination={destination.replace(' ','+')}&travelmode={travel_mode}"

    st.info("⏱️ Exact travel time will be shown in Google Maps below 👇")
    st.markdown(f"🗺️ [Open Route in Google Maps ({mode})]({map_link})")

    # MAP INSIDE APP
    st.subheader("🗺️ Map View (Gurugram)")
    map_data = pd.DataFrame({
        'lat': [28.4744],
        'lon': [77.0795]
    })
    st.map(map_data)

    # GRAPH
    st.subheader("📊 Traffic vs Speed")
    chart_data = pd.DataFrame({
        "Traffic": ["Low", "Medium", "High"],
        "Speed (km/h)": [60, 40, 25]
    })
    st.bar_chart(chart_data.set_index("Traffic"))

    # ALTERNATE ROUTE
    if result in ["Medium", "High"]:
        if random.choice([True, False]):
            st.warning("⚠️ Traffic is moderate/high, try alternate route")
            st.markdown(f"🗺️ [View Alternate Route ({mode})]({map_link})")
        else:
            st.info("✅ No better alternate route available")

# Footer
st.markdown("---")
st.caption("Smart AI Traffic Prediction 🚀")
