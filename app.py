import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="🏏 BPL Match Win Predictor", layout="centered")
st.title("🏏 BPL Match Win Predictor")

# Load the trained pipeline
pipe = None
try:
    if os.path.exists('pipe.pkl'):
        with open('pipe.pkl', 'rb') as f:
            pipe = pickle.load(f)
    else:
        st.error("⚠️ Model file `pipe.pkl` not found. Please upload it to the app directory.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Failed to load model: {e}")
    st.stop()

# Team and city options
teams = [
    'Khulna Tigers', 
    'Rangpur Riders', 
    'Barishal Bulls',
    'Sylhet Strikers', 
    'Rajshahi Kings', 
    'Dhaka Dominators',
    'Comilla Victorians', 
    'Chattogram Challengers'
]

cities = ['Dhaka', 'Chattogram', 'Sylhet', 'Khulna', 'Barishal', 'Rajshahi']

# User Inputs
col1, col2 = st.columns(2)

with col1:
    selected_batting_team = st.selectbox("Select Batting Team", sorted(teams))
with col2:
    selected_bowling_team = st.selectbox("Select Bowling Team", sorted([t for t in teams if t != selected_batting_team]))

selected_city = st.selectbox("Select City", sorted(cities))

target = st.number_input("🎯 Target Score", min_value=1)
current_score = st.number_input("🏏 Current Score", min_value=0, max_value=target)
overs_completed = st.number_input("⏱️ Overs Completed", min_value=0.0, max_value=20.0, step=0.1, format="%.1f")
wickets = st.slider("🧍‍♂️ Wickets Lost", 0, 9)

# Prediction Button
if st.button("🔮 Predict Win Probability"):
    try:
        runs_left = target - current_score
        balls_bowled = int(overs_completed * 6)
        balls_left = 120 - balls_bowled

        if balls_left <= 0:
            st.error("All overs completed. Can't predict further.")
        else:
            crr = (current_score * 6) / balls_bowled if balls_bowled > 0 else 0
            rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

            input_df = pd.DataFrame({
                'batting_team': [selected_batting_team],
                'bowling_team': [selected_bowling_team],
                'city': [selected_city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets': [10 - wickets],
                'total_runs_x': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            result = pipe.predict_proba(input_df)
            loss_prob = np.round(result[0][0]*100, 2)
            win_prob = np.round(result[0][1]*100, 2)

            st.markdown(f"### 🟢 {selected_batting_team} Win Probability: `{win_prob}%`")
            st.markdown(f"### 🔴 {selected_bowling_team} Win Probability: `{loss_prob}%`")
            st.progress(int(win_prob))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
