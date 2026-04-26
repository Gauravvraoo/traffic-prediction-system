import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.title("🚦 AI Traffic Prediction (Gurugram Routes)")

# ---------------------------
# Real Routes with Distance
# ---------------------------
routes = {
    ("IFFCO Chowk", "Rishikesh"): 250,
    ("IFFCO Chowk", "Delhi"): 30,
    ("IFFCO Chowk", "Noida"): 45,
    ("IFFCO Chowk", "Jaipur"): 280
}

# Alternative routes (for demo)
alt_routes = {
    ("IFFCO Chowk", "Rishikesh"): "Via Meerut Highway (NH334)",
    ("IFFCO Chowk", "Delhi"): "Via MG Road",
    ("IFFCO Chowk", "Noida"): "Via DND Flyway",
    ("IFFCO Chowk", "Jaipur"): "Via NH48 Expressway"
}

# ---------------------------
# Dataset (training sample)
# ---------------------------
data = pd.DataFrame({
    'Source': ['IFFCO Chowk']*4,
    'Destination': ['Rishikesh','Delhi','Noida','Jaipur'],
    'Time': [8, 14, 18, 20],
    'Day': ['Monday','Wednesday','Friday','Sunday'],
    'Weather': ['Clear','Rain','Cloudy','Clear'],
    'Distance': [250, 30, 45, 280],
    'Traffic': ['High','Medium','High','Low']
})

# ---------------------------
# Encoding
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

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# ---------------------------
# USER INPUT
# ---------------------------
st.header("📍 Enter Route Details")

source = st.selectbox("Source", ["IFFCO Chowk"])
destination = st.selectbox("Destination", ["Rishikesh","Delhi","Noida","Jaipur"])

time = st.slider("Time (Hour)", 0, 23, 8)

day = st.selectbox("Day", [
    'Monday','Tuesday','Wednesday',
    'Thursday','Friday','Saturday','Sunday'
])

weather = st.selectbox("Weather", ['Clear','Rain','Cloudy'])

# Distance auto
distance = routes[(source, destination)]
st.write(f"📏 Distance: {distance} km")

# Encode input
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

    # Travel time logic
    if result[0] == "Low":
        travel_time = f"{distance//60} hours"
    elif result[0] == "Medium":
        travel_time = f"{distance//50} hours"
    else:
        travel_time = f"{distance//40} hours"

    st.success(f"🚗 Traffic Level: {result[0]}")
    st.info(f"⏱️ Estimated Travel Time: {travel_time}")

    # Alternative route if traffic high/medium
    if result[0] in ["Medium", "High"]:
        alt = alt_routes[(source, destination)]
        st.warning(f"⚠️ Suggested Alternative Route: {alt}")

# Footer
st.markdown("---")
st.caption("AI Traffic Prediction Project 🚀")
