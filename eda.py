import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def show_eda(df):
    st.subheader("Exploratory Data Analysis")

    # ------------------------------
    # Filter & Clean Data
    # ------------------------------
    df = df.copy()

    # Remove unrealistic ages
    df = df[df['Age'].between(18, 60)]

    # Normalize gender
    valid_genders = ['Male', 'Female', 'Other']
    df = df[df['Gender'].isin(valid_genders)]

    # Normalize family_history
    df = df[df['family_history'].isin(['Yes', 'No'])]

    # ------------------------------
    # Age Distribution
    # ------------------------------
    st.markdown("### Age Distribution")
    if 'Age' in df.columns:
        fig_age = px.histogram(df, x='Age', nbins=30, title="Distribution of Age (18–60)")
        st.plotly_chart(fig_age, use_container_width=True)
    else:
        st.warning("Age column not found in dataset.")

    # ------------------------------
    # Gender Distribution
    # ------------------------------
    st.markdown("### Gender Distribution")
    if 'Gender' in df.columns:
        fig_gender = px.pie(df, names='Gender', title="Gender Breakdown")
        st.plotly_chart(fig_gender, use_container_width=True)
    else:
        st.warning("Gender column not found in dataset.")

    # ------------------------------
    # Family History of Mental Illness
    # ------------------------------
    st.markdown("### Family History")
    if 'family_history' in df.columns:
        family_df = df['family_history'].value_counts().reset_index()
        family_df.columns = ['Response', 'Count']
        fig_family = px.bar(family_df, x='Response', y='Count', title="Family History of Mental Illness")
        st.plotly_chart(fig_family, use_container_width=True)
    else:
        st.warning("family_history column not found in dataset.")

    # ------------------------------
    # Correlation Heatmap
    # ------------------------------
        # ------------------------------
    # Correlation Heatmap
    # ------------------------------
    st.markdown("### Feature Correlation (Numeric Only)")
    numeric_df = df.select_dtypes(include='number')

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, square=True, cbar_kws={"shrink": 0.75})
        ax.set_title("Correlation Heatmap", fontsize=14, pad=12)
        st.pyplot(fig)
    elif numeric_df.shape[1] == 1:
        st.info("Only one numeric column found — correlation heatmap requires at least two.")
    else:
        st.info("No numeric columns available for correlation heatmap.")

