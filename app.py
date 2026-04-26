import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.title("🚦 AI Traffic Prediction System (Gurugram)")

# ---------------------------
# REAL LOCATIONS
# ---------------------------
locations = ["IFFCO Chowk", "Cyber City", "Rajiv Chowk", "Huda City Centre"]

destinations = ["Rishikesh", "Delhi", "Noida", "Jaipur"]

# Distance mapping (approx real values)
routes = {
    ("IFFCO Chowk", "Rishikesh"): 250,
    ("IFFCO Chowk", "Delhi"): 30,
    ("IFFCO Chowk", "Noida"): 45,
    ("IFFCO Chowk", "Jaipur"): 280,

    ("Cyber City", "Rishikesh"): 255,
    ("Cyber City", "Delhi"): 25,
    ("Cyber City", "Noida"): 50,
    ("Cyber City", "Jaipur"): 285,

    ("Rajiv Chowk", "Rishikesh"): 240,
    ("Rajiv Chowk", "Delhi"): 5,
    ("Rajiv Chowk", "Noida"): 20,
    ("Rajiv Chowk", "Jaipur"): 270,

    ("Huda City Centre", "Rishikesh"): 260,
    ("Huda City Centre", "Delhi"): 35,
    ("Huda City Centre", "Noida"): 55,
    ("Huda City Centre", "Jaipur"): 290
}

# Alternative routes
alt_routes = {
    ("IFFCO Chowk", "Rishikesh"): "Via Meerut Highway (NH334)",
    ("Cyber City", "Rishikesh"): "Via NH334 & Haridwar Road",
    ("Rajiv Chowk", "Rishikesh"): "Via Ghaziabad - Meerut Expressway",
    ("Huda City Centre", "Rishikesh"): "Via NH9",

    ("IFFCO Chowk", "Delhi"): "Via MG Road",
    ("Cyber City", "Delhi"): "Via NH48",
    ("Rajiv Chowk", "Delhi"): "Inner city roads",
    ("Huda City Centre", "Delhi"): "Via Sohna Road",

    ("IFFCO Chowk", "Noida"): "Via DND Flyway",
    ("Cyber City", "Noida"): "Via NH48 + DND",
    ("Rajiv Chowk", "Noida"): "Via Akshardham route",
    ("Huda City Centre", "Noida"): "Via NH9",

    ("IFFCO Chowk", "Jaipur"): "Via NH48 Expressway",
    ("Cyber City", "Jaipur"): "Via NH48",
    ("Rajiv Chowk", "Jaipur"): "Via Gurgaon Expressway",
    ("Huda City Centre", "Jaipur"): "Via Sohna Road + NH48"
}

# ---------------------------
# SAMPLE DATASET
# ---------------------------
data = pd.DataFrame({
    'Source': ['IFFCO Chowk','Cyber City','Rajiv Chowk','Huda City Centre'],
    'Destination': ['Rishikesh','Delhi','Noida','Jaipur'],
    'Time': [8, 14, 18, 20],
    'Day': ['Monday','Wednesday','Friday','Sunday'],
    'Weather': ['Clear','Rain','Cloudy','Clear'],
    'Distance': [250, 25, 20, 290],
    'Traffic': ['High','Medium','High','Low']
})

# ---------------------------
# ENCODING
# ---------------------------
le_src = LabelEncoder()
le_dest = LabelEncoder()
le_day = LabelEncoder()
le_weather = LabelEncoder()
le_traffic = LabelEncoder()

data['Source'] = le_src.fit_transform(data['Source'])
data['Destination'] = le_dest.fit_transform(data['Destination'])
data['Day'] = le_day.fit_transform(data['Day'])
data['Weather'] = le_weather.fit_transform(data['Weather'])
data['Traffic'] = le_traffic.fit_transform(data['Traffic'])

X = data[['Source','Destination','Time','Day','Weather','Distance']]
y = data['Traffic']

model = DecisionTreeClassifier()
model.fit(X, y)

# ---------------------------
# USER INPUT
# ---------------------------
st.header("📍 Enter Route Details")

source = st.selectbox("Select Source", locations)
destination = st.selectbox("Select Destination", destinations)

time = st.slider("Time (Hour)", 0, 23, 8)

day = st.selectbox("Day", [
    'Monday','Tuesday','Wednesday',
    'Thursday','Friday','Saturday','Sunday'
])

weather = st.selectbox("Weather", ['Clear','Rain','Cloudy'])

# Distance fetch
distance = routes.get((source, destination), 100)
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

    # Travel time
    if result[0] == "Low":
        travel_time = f"{distance//60} hours"
    elif result[0] == "Medium":
        travel_time = f"{distance//50} hours"
    else:
        travel_time = f"{distance//40} hours"

    st.success(f"🚗 Traffic Level: {result[0]}")
    st.info(f"⏱️ Estimated Travel Time: {travel_time}")

    # Alternative route
    if result[0] in ["Medium", "High"]:
        alt = alt_routes.get((source, destination), "Use alternate highway")
        st.warning(f"⚠️ Alternative Route: {alt}")

# Footer
st.markdown("---")
st.caption("Smart AI Traffic Prediction 🚀")
