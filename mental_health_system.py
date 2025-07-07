# mental_health_system.py
pip install scikit-learn
pip install matplotlib

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Generate Sample Dataset
# -----------------------------------
def generate_mock_data():
    data = {
        'sleep_hours': np.random.randint(4, 9, 100),
        'screen_time': np.random.randint(4, 12, 100),
        'work_hours': np.random.randint(6, 12, 100),
        'breaks_taken': np.random.randint(0, 4, 100),
        'mood_score': np.random.randint(1, 10, 100),
        'social_activity': np.random.randint(0, 3, 100),
        'stress_level': np.random.choice(['Low', 'Medium', 'High'], 100)
    }
    return pd.DataFrame(data)

# -----------------------------------
# 2. Train Model
# -----------------------------------
def train_model(df):
    X = df[['sleep_hours', 'screen_time', 'work_hours', 'breaks_taken', 'mood_score', 'social_activity']]
    y = df['stress_level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# -----------------------------------
# 3. Predict Stress Level
# -----------------------------------
def predict_stress(model, input_data):
    df = pd.DataFrame([input_data])
    return model.predict(df)[0]

# -----------------------------------
# 4. Recommendation Logic
# -----------------------------------
def get_recommendation(stress_level):
    if stress_level == "Low":
        return "✅ You're doing great! Keep it up!"
    elif stress_level == "Medium":
        return "🟡 Consider taking breaks, reducing screen time, or getting more sleep."
    else:
        return "🔴 Please consider speaking with a friend or mental health professional."

# -----------------------------------
# 5. Streamlit App
# -----------------------------------
def main():
    st.set_page_config(page_title="Mental Health System", layout="centered")
    st.title("🧠 Mental Health and Stress Prevention System")
    st.write("Enter your daily routine to assess stress level:")

    sleep_hours = st.slider("🛌 Sleep Hours", 0, 12, 6)
    screen_time = st.slider("📱 Screen Time (in hours)", 0, 15, 8)
    work_hours = st.slider("💼 Work Hours", 0, 12, 8)
    breaks_taken = st.slider("☕ Breaks Taken", 0, 5, 2)
    mood_score = st.slider("😊 Mood Score (1 - worst, 10 - best)", 1, 10, 5)
    social_activity = st.slider("👥 Social Activity (0=Low, 1=Medium, 2=High)", 0, 2, 1)

    if st.button("📊 Analyze Stress Level"):
        # Step 1: Create mock data and train model
        data = generate_mock_data()
        model = train_model(data)

        # Step 2: Gather user input
        user_input = {
            'sleep_hours': sleep_hours,
            'screen_time': screen_time,
            'work_hours': work_hours,
            'breaks_taken': breaks_taken,
            'mood_score': mood_score,
            'social_activity': social_activity
        }

        # Step 3: Predict stress level
        stress = predict_stress(model, user_input)
        recommendation = get_recommendation(stress)

        # Output Results
        st.subheader(f"🧾 Predicted Stress Level: `{stress}`")
        st.success(recommendation)

        # Visual Summary
        st.write("### 📊 Your Input Overview")
        fig, ax = plt.subplots()
        ax.barh(list(user_input.keys()), list(user_input.values()), color='skyblue')
        ax.set_xlabel("Values")
        ax.set_title("Your Daily Routine Summary")
        st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()

