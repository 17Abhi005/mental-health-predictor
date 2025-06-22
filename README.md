# Mental Health Predictor

An interactive web app built with **Streamlit** that predicts whether an individual is likely to seek mental health treatment using a survey dataset from **Kaggle**. Includes real-time predictions, insightful EDA, and a trained machine learning model.

---

## Features

- **Real-time Prediction** from user survey input  
- **Interactive Visualizations** using Plotly and Seaborn  
- **Exploratory Data Analysis** (Age, Gender, Family History, etc.)  
- **Correlation Heatmap** of features  
- **Downloadable and Deployable App**

---

## Model Details

- **Random Forest Classifier** trained on cleaned & encoded survey data  
- **Label Encoders** saved for consistent predictions  
- **Class Balance** handled with `class_weight='balanced'`  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score

---

## Installation & Run Locally

### 1. Clone the Repo
```bash
git clone https://github.com/17Abhi005/mental-health-predictor.git
cd mental-health-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Run the App
```bash
streamlit run app.py
or
Python -m streamlit run app.py
```

---
