import streamlit as st
import joblib

# Load model files
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
target_encoder = joblib.load("target_encoder.joblib")

st.title("Introvert vs Extrovert Prediction System")

# Numerical inputs
time_spent_alone = st.number_input("Time Spent Alone")
social_event_attendance = st.number_input("Social Event Attendance")
going_outside = st.number_input("Going Outside")
friends_circle_size = st.number_input("Friends Circle Size")
post_frequency = st.number_input("Post Frequency")

# Categorical inputs
stage_fear = st.selectbox("Stage Fear", [0, 1])
drained_after_socializing = st.selectbox("Drained After Socializing", [0, 1])

# Predict button
if st.button("Predict Personality"):

    data = [[
        time_spent_alone,
        stage_fear,
        social_event_attendance,
        going_outside,
        drained_after_socializing,
        friends_circle_size,
        post_frequency
    ]]

    data = scaler.transform(data)

    prediction = model.predict(data)

    result = target_encoder.inverse_transform(prediction)

    st.success(f"Predicted Personality: {result[0]}")