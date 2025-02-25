
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:33:30 2025

@author: KF8447
"""

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set a colorful CSS style
st.markdown("""
    <style>
    body {
        background-color: #1c1c1c;
        color: #f0f0f0;
    }
    h1 {
        color: #00ffff;
        text-align: center;
    }
    h2, h3, h4 {
        color: #ff69b4;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e2e2e, #4b4b4b);
        color: #f0f0f0;
    }
    </style>
    """, unsafe_allow_html=True)
    


# App Header with a decorative banner image using a fallback image URL
col1, col2 = st.columns([5, 3])
with col1:


    st.markdown("""
    <div style="text-align: center; background-color: #003366; padding: 20px; border-radius: 10px; margin-bottom: 15px;">   
    <h1 style="color: #FFD700; font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;; font-size: 20px; font-weight: 600;">
            Transforming Insights into Action: Telco Customer Churn Analysis
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    
with col2:

    st.image("Title.jpeg", width=200)
# Load dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv('Customer_Churn.csv')
    return df

df = load_data()

# Preprocessing function
def preprocess_data(data):
    data = data.copy()
    # Drop customerID column if it exists
    if 'customerID' in data.columns:
        data = data.drop(columns=['customerID'])
    
    # Replace inconsistent service strings
    data.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)
    
    # Convert TotalCharges to numeric and fill missing values with the median
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
    
    # Encode binary categorical variables
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        data[col] = data[col].apply(lambda x: 1 if x in ['Male', 'Yes'] else 0)
    
    # Function to encode three-option services
    def encode_service(x):
        if x == "Yes":
            return 2
        elif x == "No":
            return 0
        else:
            return 1  # Covers any non-standard response
    
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        data[col] = data[col].apply(encode_service)
    
    # Encode Contract: Month-to-month -> 0, One year -> 1, Two year -> 2
    data['Contract'] = data['Contract'].apply(lambda x: 0 if x == "Month-to-month" else 1 if x == "One year" else 2)
    
    # Encode PaymentMethod: Electronic check -> 0, Mailed check -> 1, Bank transfer (automatic) -> 2, Credit card (automatic) -> 3
    data['PaymentMethod'] = data['PaymentMethod'].apply(
        lambda x: 0 if x == "Electronic check" else 1 if x == "Mailed check" 
        else 2 if x == "Bank transfer (automatic)" else 3
    )
    
    # Ensure target variable (Churn) is numeric
    if data['Churn'].dtype == 'object':
        data['Churn'] = data['Churn'].apply(lambda x: 1 if x == "Yes" else 0)
    
    return data

df_processed = preprocess_data(df)

# Train-Test Split and Model Training
def train_model(data):
    features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    X = data[features]
    y = data['Churn']
    
    # Split the data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Get feature importance
    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    
    return model, acc, cm, feature_importance, X_train, X_test, y_train, y_test

model, acc, cm, feature_importance, X_train, X_test, y_train, y_test = train_model(df_processed)

# Sidebar options for navigation
option = st.sidebar.selectbox("🔍 Choose Analysis", 
                              ["📊 Dataset Overview", "📈 Visualizations", "🤖 Churn Prediction", "📈 Model Evaluation"])
# Global Filters - Positioned Below Page Selection
st.sidebar.markdown("### 🌍 Global Filters")

gender_filter = st.sidebar.multiselect("Filter by Gender", 
                                       options=[0, 1], 
                                       default=[0, 1], 
                                       format_func=lambda x: "Male" if x == 1 else "Female")

# Filter for contract type (0 = Month-to-month, 1 = One year, 2 = Two year)
contract_filter = st.sidebar.multiselect("Filter by Contract Type", 
                                         options=[0, 1, 2], 
                                         default=[0, 1, 2], 
                                         format_func=lambda x: ["Month-to-month", "One year", "Two year"][x])


# Filter for churn status (0 = No, 1 = Yes)
churn_filter = st.sidebar.multiselect("Filter by Churn", 
                                      options=[0, 1], 
                                      default=[0, 1], 
                                      format_func=lambda x: "Yes" if x == 1 else "No")

# Filter for Senior Citizen (0 = No, 1 = Yes)
seniorcitizen_filter = st.sidebar.multiselect("Filter by Senior Citizen", 
                                      options=[0, 1], 
                                      default=[0, 1], 
                                      format_func=lambda x: "Yes" if x == 1 else "No")
# Filter for Dependents (0 = No, 1 = Yes)
dependents_filter = st.sidebar.multiselect("Filter by Dependents", 
                                      options=[0, 1], 
                                      default=[0, 1], 
                                      format_func=lambda x: "Yes" if x == 1 else "No")

# Apply filters to the dataset
filtered_df = df_processed[
    (df_processed["gender"].isin(gender_filter)) &  # Gender filter
    (df_processed["Contract"].isin(contract_filter)) &  # Contract filter
    (df_processed["Churn"].isin(churn_filter)) &  # Churn filter
    (df_processed["SeniorCitizen"].isin(seniorcitizen_filter)) &  # Senior Citizen filter
    (df_processed["Dependents"].isin(dependents_filter))  # Corrected column name
]



# Page 1: Dataset Overview
if option == "📊 Dataset Overview":


    st.write("## 📝 Filtered Dataset Overview")
    # Convert customer_churn DataFrame to CSV
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    if filtered_df.empty:
        st.warning("⚠️ No data available based on the selected filters. Try modifying your selection.")
    else:
        # Convert numerical columns back to readable text for display
        display_df = filtered_df.copy()
        display_df["gender"] = display_df["gender"].map({0: "Female", 1: "Male"})
        display_df["Contract"] = display_df["Contract"].map({0: "Month-to-month", 1: "One year", 2: "Two year"})
        display_df["Churn"] = display_df["Churn"].map({0: "No", 1: "Yes"})
        display_df["SeniorCitizen"] = display_df["SeniorCitizen"].map({0: "No", 1: "Yes"})
        display_df["Dependents"] = display_df["Dependents"].map({0: "No", 1: "Yes"})
        
        st.write(display_df.head())  # Show first few rows with labels
        st.write("### 📌 Filtered Data Summary")
        st.write(display_df.describe())
        
        # Add a Download Button for CSV
        st.download_button(
        label="📥 Download Customer Churn Data as CSV",
        data=csv_data,
        file_name="customer_churn.csv",
        mime="text/csv",
)

    # Shareable Link (Replace with actual deployed URL)
    dashboard_url = "http://localhost:8501/"
    
    st.markdown(
        f"""
        📤 **Share this Dashboard:**  
        Copy & Share this link 👉 [Customer Churn Dashboard]({dashboard_url})
        """,
        unsafe_allow_html=True
    )
    
    # Copy Link Button
    st.text_input("🔗 Shareable Link", dashboard_url, disabled=True)
    st.button("📋 Copy Link")

# Page 2: Visualizations
elif option == "📈 Visualizations":
    # Section 1: Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Increase width of col1 (1:1 ratio)
    col1, col2, col3, col4 = st.columns([1, 1,1,2])  
    
    with col1:
        st.metric("Total Customers", len(filtered_df),help="Total number of customers in dataset")
    
    # Convert 'Churn' column from strings to numeric (1 for Yes, 0 for No)
    if filtered_df['Churn'].dtype == 'object':
        filtered_df['Churn'] = filtered_df['Churn'].map({'Yes': 1, 'No': 0})
    
    with col2:
        churn_rate = filtered_df['Churn'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
    with col3:
        st.metric("Avg Monthly Charges", f"${filtered_df['MonthlyCharges'].mean():.2f}")
        
    with col4:
        st.metric("Avg Tenure", f"{filtered_df['tenure'].mean():.1f} months",help="Average Customer Lifetime Value")

    # Increase width of col1 (1:1 ratio)
    col1, col2 = st.columns([1, 1])  

        
    with col1:
        # Create a copy and map gender values to labels for display
        pie_df = filtered_df.copy()
        pie_df["gender"] = pie_df["gender"].map({0: "Female", 1: "Male"})  # Convert 0/1 to labels
        
        # Calculate gender distribution percentages
        gender_counts = pie_df["gender"].value_counts(normalize=True) * 100
        male_percentage = gender_counts.get("Male", 0)
        female_percentage = gender_counts.get("Female", 0)
        
        # Display dynamic help text
        st.caption(f"📊 Gender Distribution: Males - {male_percentage:.1f}%, Females - {female_percentage:.1f}%")
        
        # Generate pie chart with readable labels
        fig = px.pie(pie_df, names='gender', title='Gender Distribution',
                     color_discrete_sequence=px.colors.sequential.Aggrnyl)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(width=400, height=500)
    
        st.plotly_chart(fig, use_container_width=True)

       
            
    with col2:
        fig = px.histogram(filtered_df, x='tenure', nbins=20, title='Tenure Distribution',
                           color_discrete_sequence=['#00F3FF'])
        fig.update_layout(bargap=0.1)
        fig.update_layout(width=1000, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    

    # Funnel Chart: Customer Retention Flow
    st.markdown("""
        <h3 style="color:#FF1493;">📊 Customer Retention Funnel</h3>
    """, unsafe_allow_html=True)

    # Group data to calculate churned vs retained customers
    funnel_data = filtered_df.groupby('Churn').size().reset_index(name='count')
    funnel_data['Churn'] = funnel_data['Churn'].map({0: "Retained", 1: "Churned"})

    # Create funnel chart
    fig = px.funnel(
        funnel_data, 
        x='count', 
        y='Churn', 
        title="Customer Retention Funnel"
    )
    
    # Display chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


    # Sunburst Chart: Customer Breakdown by Contract, Payment Method, and Churn

    st.markdown("""
        <h3 style="color:#32CD32;">📌 Customer Breakdown: Contract + Payment + Churn</h3>
    """, unsafe_allow_html=True)
    
    # Create a copy for display and map values for better readability
    sunburst_df = filtered_df.copy()
    sunburst_df["Contract"] = sunburst_df["Contract"].map({0: "Month-to-month", 1: "One year", 2: "Two year"})
    sunburst_df["Churn"] = sunburst_df["Churn"].map({0: "No", 1: "Yes"})
    sunburst_df["PaymentMethod"] = sunburst_df["PaymentMethod"].map({
        0: "Electronic Check", 1: "Mailed Check", 2: "Bank Transfer (Auto)", 3: "Credit Card (Auto)"
    })
    
    # Group the data to get counts for each segment
    grouped_sunburst = sunburst_df.groupby(["Contract", "PaymentMethod", "Churn"]).size().reset_index(name='Count')
    
    # Calculate churn rates dynamically
    churn_rates = sunburst_df.groupby(["Contract", "PaymentMethod"])["Churn"].value_counts(normalize=True).unstack().fillna(0)
    churn_rates["Churn Rate"] = churn_rates["Yes"] * 100  # Convert to percentage
    
    # Identify highest and lowest churn segments
    highest_churn = churn_rates["Churn Rate"].idxmax()
    lowest_churn = churn_rates["Churn Rate"].idxmin()
    
    highest_churn_text = f"{highest_churn[0]} contracts with {highest_churn[1]} have the highest churn rate."
    lowest_churn_text = f"{lowest_churn[0]} contracts with {lowest_churn[1]} have the lowest churn rate."
    
    # Display dynamic churn insights
    st.markdown(f"📊 **{highest_churn_text}**")
    st.markdown(f"✅ **{lowest_churn_text}**")
    
    # Create Sunburst chart with correct values and labels
    fig = px.sunburst(
        grouped_sunburst,
        path=['Contract', 'PaymentMethod', 'Churn'],
        values='Count',
        title="Customer Segmentation",
        color='Churn', 
        color_discrete_map={"No": "#008000", "Yes": "#FF0000"}  # Green for No Churn, Red for Churn
    )
    
    # Show labels explicitly with percentages
    fig.update_traces(textinfo="label+percent parent")
    
    # Display chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


    st.header("🔌 Service Usage Patterns")
    service_cols = ['PhoneService', 'InternetService', 'StreamingTV', 'TechSupport']
    selected_service = st.selectbox("Select Service", service_cols)

    fig = px.bar(filtered_df, x=selected_service, color='Churn', barmode='group',
                 title=f'{selected_service} vs Churn',
                 color_discrete_sequence=['#FF00FF', '#00FF00'])

    st.plotly_chart(fig, use_container_width=True)

# Page 3: Churn Prediction
elif option == "🤖 Churn Prediction":
    st.write("## 🔮 Churn Prediction Model")

    # User input for features
    tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=1500.0)
    
    # Categorical inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    # Encode inputs consistently with training
    gender_encoded = 1 if gender == "Male" else 0
    partner_encoded = 1 if partner == "Yes" else 0
    dependents_encoded = 1 if dependents == "Yes" else 0
    phone_service_encoded = 1 if phone_service == "Yes" else 0
    
    def encode_service_input(x):
        if x == "Yes":
            return 2
        elif x == "No":
            return 0
        else:
            return 1
    
    multiple_lines_encoded = encode_service_input(multiple_lines)
    online_security_encoded = encode_service_input(online_security)
    online_backup_encoded = encode_service_input(online_backup)
    device_protection_encoded = encode_service_input(device_protection)
    tech_support_encoded = encode_service_input(tech_support)
    streaming_tv_encoded = encode_service_input(streaming_tv)
    streaming_movies_encoded = encode_service_input(streaming_movies)
    
    contract_encoded = 0 if contract=="Month-to-month" else 1 if contract=="One year" else 2
    paperless_billing_encoded = 1 if paperless_billing=="Yes" else 0
    payment_method_encoded = (0 if payment_method=="Electronic check" 
                              else 1 if payment_method=="Mailed check" 
                              else 2 if payment_method=="Bank transfer (automatic)" else 3)
    
    # Create input data DataFrame
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'gender': [gender_encoded],
        'Partner': [partner_encoded],
        'Dependents': [dependents_encoded],
        'PhoneService': [phone_service_encoded],
        'MultipleLines': [multiple_lines_encoded],
        'OnlineSecurity': [online_security_encoded],
        'OnlineBackup': [online_backup_encoded],
        'DeviceProtection': [device_protection_encoded],
        'TechSupport': [tech_support_encoded],
        'StreamingTV': [streaming_tv_encoded],
        'StreamingMovies': [streaming_movies_encoded],
        'Contract': [contract_encoded],
        'PaperlessBilling': [paperless_billing_encoded],
        'PaymentMethod': [payment_method_encoded]
    })
    
    if st.button("Predict Churn"):
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.write("**Prediction: This customer is likely to churn.**")
        else:
            st.write("**Prediction: This customer is likely to stay.**")

# Page 4: Model Evaluation
elif option == "📈 Model Evaluation":
    st.write("## 📊 Model Evaluation Metrics")

    # Apply colorful, bold, and large font size to Model Accuracy
    st.markdown(f"""
        <div style="text-align: center; 
                    font-size: 24px; 
                    font-weight: bold; 
                    color: #FF4500; 
                    background-color: #1E1E1E; 
                    padding: 10px; 
                    border-radius: 10px; 
                    box-shadow: 2px 2px 10px rgba(255, 69, 0, 0.5);">
            🔥 Model Accuracy: {acc:.2f} 🔥
        </div>
    """, unsafe_allow_html=True)

    # Add more space
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Confusion Matrix
    st.markdown("""
        <h3 style="color:#FF6347; font-size:24px; font-weight: bold;">📊 Confusion Matrix</h3>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Add space between sections
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Feature Importance
    st.markdown("""
        <h3 style="color:#FF6347; font-size:24px; font-weight: bold;">🚀 Feature Importance</h3>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots()
    feature_importance.plot(kind='bar', ax=ax, color='navy')
    st.pyplot(fig)

    # Add more space
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Correlation Matrix
    st.markdown("""
        <h3 style="color:#FF6347; font-size:24px; font-weight: bold;">📈 Correlation Matrix</h3>
    """, unsafe_allow_html=True)

    # Compute the correlation matrix
    numeric_df = df_processed.select_dtypes(include=[np.number])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)

    st.pyplot(fig)

    # Add more space
    st.markdown("<br><br>", unsafe_allow_html=True)



    


