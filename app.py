import streamlit as st
import numpy as np
import pickle
import pandas as pd
from eda import show_eda

# Page config
st.set_page_config(
    page_title="Mental Health Risk Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load cleaned and raw data
@st.cache_data
def load_data():
    return pd.read_csv("data/survey.csv")

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Predict Risk", "Explore Data", "About Project"])

# Header
st.markdown("""
    <h1 style='text-align: center;'>Mental Health Risk Predictor</h1>
    <p style='text-align: center; color: grey;'>
        Use this interactive tool to assess potential mental health risk<br>
        based on workplace-related survey data.
    </p>
""", unsafe_allow_html=True)

# -----------------------
# Page 1: Predictor
# -----------------------
if page == "Predict Risk":
    st.markdown("### Please fill in the following details:")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 60, 25)
        self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
    with col2:
        family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
        work_interfere = st.selectbox("How often does your mental health interfere with work?", 
                                      ["Never", "Rarely", "Sometimes", "Often"])

    mapping = {"Yes": 1, "No": 0, "Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
    features = np.array([
        age,
        mapping[self_employed],
        mapping[family_history],
        mapping[work_interfere]
    ]).reshape(1, -1)

    model = pickle.load(open("model/model.pkl", "rb"))

    if st.button("Predict Mental Health Risk"):
        prediction = model.predict(features)[0]
        if prediction == 1:
            st.error("You may be at high risk for mental health issues. Consider seeking help.")
        else:
            st.success("You seem to be at low risk for mental health issues.")

# -----------------------
# Page 2: EDA
# -----------------------
elif page == "Explore Data":
    st.markdown("### Data Explorer")
    show_eda(df)

# -----------------------
# Page 3: About
# -----------------------
elif page == "About Project":
    st.markdown("## About This Project")

    st.markdown("""
    This tool is built to explore and predict mental health treatment needs based on a publicly available mental health survey dataset.

    - Data Source: [Kaggle Mental Health in Tech Survey 2014](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
    - Dataset contains responses from employees in the tech industry about mental health awareness, treatment, and workplace conditions.
    - Model: Random Forest Classifier trained on selected features.

    ---
    ### Original Raw Data (Preview)
    """)

    raw_df = pd.read_csv("data/survey.csv")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.markdown("""
    ---
    ### How to Use
    - Go to **"Predict Risk"** to check your mental health risk.
    - Go to **"Explore Data"** for visual insights from the dataset.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Made by <b>Abhishek Gupta</b></p>",
    unsafe_allow_html=True
)
