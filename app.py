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
    "MG Road"
]

# ---------------------------
# DESTINATIONS
# ---------------------------
destinations = [
    "Panchgaon",
    "Manesar",
    "Sohna",
    "Delhi",
    "Noida",
    "Jaipur",
    "Rishikesh"
]

# ---------------------------
# DISTANCE DATA
# ---------------------------
routes = {
    ("IFFCO Chowk", "Panchgaon"): 25,
    ("Cyber City", "Panchgaon"): 30,
    ("Huda City Centre", "Panchgaon"): 20,
    ("MG Road", "Panchgaon"): 28,

    ("IFFCO Chowk", "Manesar"): 20,
    ("Cyber City", "Manesar"): 25,
    ("Huda City Centre", "Manesar"): 18,
    ("MG Road", "Manesar"): 22,

    ("IFFCO Chowk", "Sohna"): 30,
    ("Cyber City", "Sohna"): 35,
    ("Huda City Centre", "Sohna"): 25,
    ("MG Road", "Sohna"): 32,

    ("IFFCO Chowk", "Delhi"): 30,
    ("Cyber City", "Delhi"): 25,
    ("Huda City Centre", "Delhi"): 35,
    ("MG Road", "Delhi"): 28,

    ("IFFCO Chowk", "Noida"): 45,
    ("Cyber City", "Noida"): 50,
    ("Huda City Centre", "Noida"): 55,
    ("MG Road", "Noida"): 48,

    ("IFFCO Chowk", "Jaipur"): 280,
    ("Cyber City", "Jaipur"): 285,
    ("Huda City Centre", "Jaipur"): 290,
    ("MG Road", "Jaipur"): 275,

    ("IFFCO Chowk", "Rishikesh"): 250,
    ("Cyber City", "Rishikesh"): 255,
    ("Huda City Centre", "Rishikesh"): 260,
    ("MG Road", "Rishikesh"): 245
}

# ---------------------------
# ALTERNATIVE ROUTES
# ---------------------------
alt_routes = {
    ("IFFCO Chowk", "Panchgaon"): "Via NH48 Service Road",
    ("IFFCO Chowk", "Manesar"): "Via KMP Expressway",
    ("IFFCO Chowk", "Sohna"): "Via Sohna Road",
    ("IFFCO Chowk", "Delhi"): "Via MG Road",
    ("IFFCO Chowk", "Noida"): "Via DND Flyway",
    ("IFFCO Chowk", "Jaipur"): "Via NH48",
    ("IFFCO Chowk", "Rishikesh"): "Via Meerut Expressway"
}

# ---------------------------
# DATASET (FIXED)
# ---------------------------
data = pd.DataFrame({
    'Source': [
        "IFFCO Chowk","Cyber City","Huda City Centre","MG Road",
        "IFFCO Chowk","Cyber City","Huda City Centre","MG Road"
    ],
    'Destination': [
        "Panchgaon","Delhi","Noida","Jaipur",
        "Rishikesh","Manesar","Sohna","Delhi"
    ],
    'Time': [8,14,18,20,10,16,9,22],
    'Day': [
        "Monday","Wednesday","Friday","Sunday",
        "Tuesday","Thursday","Saturday","Monday"
    ],
    'Weather': [
        "Clear","Rain","Cloudy","Clear",
        "Rain","Clear","Cloudy","Rain"
    ],
    'Distance': [25,25,55,275,250,20,30,30],
    'Traffic': [
        "High","Medium","High","Low",
        "High","Medium","Low","Medium"
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

source = st.selectbox("Select Start Location", start_locations)
destination = st.selectbox("Select Destination", destinations)

time = st.slider("Time (Hour)", 0, 23, 8)

day = st.selectbox("Day", [
    'Monday','Tuesday','Wednesday',
    'Thursday','Friday','Saturday','Sunday'
])

weather = st.selectbox("Weather", ['Clear','Rain','Cloudy'])

distance = routes.get((source, destination), 50)
st.write(f"📏 Distance: {distance} km")

# Encode
src_enc = le_src.transform([source])[0]
dest_enc = le_dest.transform([destination])[0]
day_enc = le_day.transform([day])[0]
weather_enc = le_weather.transform([weather])[0]

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("🚀 Predict Traffic"):

    prediction = model.predict([[src_enc, dest_enc, time, day_enc, weather_enc, distance]])
    result = le_traffic.inverse_transform(prediction)

    # Time calculation
    if result[0] == "Low":
        speed = 50
    elif result[0] == "Medium":
        speed = 40
    else:
        speed = 30

    time_taken = distance / speed
    hours = int(time_taken)
    minutes = int((time_taken - hours) * 60)

    if hours == 0:
        travel_time = f"{minutes} minutes"
    else:
        travel_time = f"{hours} hr {minutes} min"

    st.success(f"🚗 Traffic Level: {result[0]}")
    st.info(f"⏱️ Estimated Travel Time: {travel_time}")

    # ---------------------------
    # FINAL ALTERNATE ROUTE LOGIC
    # ---------------------------
    if result[0] in ["Medium", "High"]:
        if random.choice([True, False]):
            alt = alt_routes.get((source, destination), "Use alternate road")
            st.warning(f"⚠️ Suggested Alternative Route: {alt}")
        else:
            st.info("✅ No better alternate route available")

# Footer
st.markdown("---")
st.caption("Smart AI Traffic Prediction 🚀")
