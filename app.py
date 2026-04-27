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

# DESTINATIONS
destinations = [
    "Panchgaon","Manesar","Sohna",
    "Delhi","Noida","Jaipur","Rishikesh"
]

# DISTANCES
routes = {
    ("IFFCO Chowk", "Panchgaon"): 25,
    ("IFFCO Chowk", "Delhi"): 30,
    ("IFFCO Chowk", "Noida"): 45,
    ("IFFCO Chowk", "Jaipur"): 280,
    ("IFFCO Chowk", "Rishikesh"): 250
}

# ---------------------------
# DATASET
# ---------------------------
data = pd.DataFrame({
    'Source': ["IFFCO Chowk","Cyber City","Huda City Centre","MG Road",
               "IFFCO Chowk","Cyber City","Huda City Centre","MG Road"],
    'Destination': ["Panchgaon","Delhi","Noida","Jaipur",
                    "Rishikesh","Manesar","Sohna","Delhi"],
    'Time': [8,14,18,20,10,16,9,22],
    'Day': ["Monday","Wednesday","Friday","Sunday",
            "Tuesday","Thursday","Saturday","Monday"],
    'Weather': ["Clear","Rain","Cloudy","Clear",
                "Rain","Clear","Cloudy","Rain"],
    'Distance': [25,25,55,275,250,20,30,30],
    'Traffic': ["High","Medium","High","Low",
                "High","Medium","Low","Medium"]
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
st.write(f"📏 Distance: {distance} km")

# Encode
src_enc = le_src.transform([source])[0]
dest_enc = le_dest.transform([destination])[0]
day_enc = le_day.transform([day])[0]
weather_enc = le_weather.transform([weather])[0]

# ---------------------------
# PREDICT
# ---------------------------
if st.button("🚀 Predict Traffic"):

    pred = model.predict([[src_enc, dest_enc, time, day_enc, weather_enc, distance]])
    result = le_traffic.inverse_transform(pred)[0]

    # Time calc
    speed = 50 if result=="Low" else 40 if result=="Medium" else 30
    t = distance / speed
    h = int(t)
    m = int((t-h)*60)

    travel = f"{m} min" if h==0 else f"{h} hr {m} min"

    st.success(f"🚗 Traffic: {result}")
    st.info(f"⏱️ Time: {travel}")

    # ---------------------------
    # GOOGLE MAP LINK
    # ---------------------------
    map_link = f"https://www.google.com/maps/dir/?api=1&origin={source.replace(' ','+')}&destination={destination.replace(' ','+')}"
    st.markdown(f"🗺️ [Open in Google Maps]({map_link})")

    # ---------------------------
    # MAP INSIDE APP
    # ---------------------------
    st.subheader("🗺️ Map View")
    map_data = pd.DataFrame({
        'lat': [28.4744],   # Gurugram approx
        'lon': [77.0795]
    })
    st.map(map_data)

    # ---------------------------
    # GRAPH (Traffic Comparison)
    # ---------------------------
    st.subheader("📊 Traffic Comparison")

    chart_data = pd.DataFrame({
        "Traffic Level": ["Low","Medium","High"],
        "Speed (km/h)": [50,40,30]
    })

    st.bar_chart(chart_data.set_index("Traffic Level"))

    # ---------------------------
    # ALTERNATE ROUTE
    # ---------------------------
    if result in ["Medium","High"]:
        if random.choice([True, False]):
            st.warning("⚠️ Try alternate route via highways")
        else:
            st.info("✅ No better route available")

st.markdown("---")
st.caption("Smart AI Traffic Prediction 🚀")
