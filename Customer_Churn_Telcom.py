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

st.markdown("""
    <div style="text-align: center; background-color: #003366; padding: 20px; border-radius: 10px; margin-bottom: 15px;">   
    <h1 style="color: #FFD700; font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;; font-size: 25px; font-weight: 600;">
            Transforming Insights into Action: Telco Customer Churn Analysis
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    
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
                              ["📊 Dataset Exploration","📈 Visualizations", "🤖 Churn Prediction", "📈 Model Validation & Analysis", "🎯 Customer Profile Analysis"])
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
if option == "📊 Dataset Exploration":

    
    st.markdown("""
        <div style="text-align: center; 
                    font-size: 25px; 
                    font-weight: bold; 
                    color: #000000;"> 
            📝 Filtered Dataset Overview
        </div>
      """, unsafe_allow_html=True)
      

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
        
        
        st.markdown("""
            <div style="text-align: center; 
                        font-size: 25px; 
                        font-weight: bold; 
                        color: #000000;"> 
                📝 Filtered Data Summary
            </div>
          """, unsafe_allow_html=True)
          
        st.write(display_df.describe())
        
        # Add a Download Button for CSV
        st.download_button(
        label="📥 Download Customer Churn Data as CSV",
        data=csv_data,
        file_name="customer_churn.csv",
        mime="text/csv",
)

    # Shareable Link (Replace with actual deployed URL)
    dashboard_url = "https://telcochurnmodel-eemv76bq5znp2ypijyfk3b.streamlit.app/"
    
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
    # Increase width of col1 (1:1 ratio)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    # Apply Custom CSS for Grey Boxes and Yellow Text
    st.markdown("""
        <style>
            .metric-box {
                background-color: #444; /* Grey background */
                padding: 8px;
                border-radius: 6px;
                text-align: center;
                color: #FFD700 !important; /* Yellow font */
                font-size: 14px;
                margin: 5px;
            }
            .metric-title {
                font-size: 12px;
                font-weight: bold;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Function to create a custom metric box
    def custom_metric(title, value):
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">{title}</div>
                <div>{value}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col1:
        custom_metric("Total Customers", len(filtered_df))
    
    # Convert 'Churn' column from strings to numeric (1 for Yes, 0 for No)
    if filtered_df['Churn'].dtype == 'object':
        filtered_df['Churn'] = filtered_df['Churn'].map({'Yes': 1, 'No': 0})
    
    with col2:
        churn_rate = filtered_df['Churn'].mean() * 100
        custom_metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col3:
        custom_metric("Avg Monthly Charges", f"${filtered_df['MonthlyCharges'].mean():.2f}")
    
    with col4:
        custom_metric("Avg Tenure", f"{filtered_df['tenure'].mean():.1f} months")
    
             
       # 📌 Apply Custom CSS for Better Styling
    st.markdown("""<style>
        body {
            background-color: #f7f7f7;  /* Light Background */
            color: #FFD700 !important;  /* Yellow Font */
        }
        .box-container {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .chart-selector {
            font-size: 22px !important;
            text-align: center;
            font-weight: bold !important;
            color: #FFD700 !important;  /* Yellow Font */
        }
        .stRadio > label {
            font-size: 30px !important;
            text-align: center;
            font-weight: bold !important;
            color: #FFD700 !important;  /* Yellow Font */
        }
        .stRadio div[role="radiogroup"] {
            justify-content: center;
            display: flex;
            background: #FFF2CC;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.1);
        }
        .stRadio div[role="radiogroup"] label {
            color: #000000 !important; /* Black text for buttons *
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 25px !important;
        }
        h1, h2, h3 {
            color: #FFD700 !important;  /* Yellow Font */
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    
        # Center-align and set medium font size for "Select Chart Type"
    st.markdown("""
            <div style="text-align: center; 
                        font-size: 25px; 
                        font-weight: bold; 
                        color: #000000;"> 
                📊 Select Chart Type
            </div>
        """, unsafe_allow_html=True)
        
        # Radio Button for Chart Selection (Still Functional)

    chart_type = st.radio("", ["Pie Chart", "Histogram", "Sunburst Chart", "Funnel Chart", "Service Usage Patterns"], horizontal=True)
        
            
    st.markdown("""
              <div style="text-align: center; 
                          font-size: 25px; 
                          font-weight: bold; 
                          color: #000000;"> 
                  📊 Smart Churn Analytics
              </div>
          """, unsafe_allow_html=True)
          
   # Add more space
    st.markdown("<br><br>", unsafe_allow_html=True)
       
    
# Pie Chart: Churn Distribution
    if chart_type == "Pie Chart":
        st.markdown('<h3 style="color:#000000;">📊 Gender Pie Chart</h3>', unsafe_allow_html=True)

        # Create a copy and map gender values to labels for display
        pie_df = filtered_df.copy()
        pie_df["gender"] = pie_df["gender"].map({0: "Female", 1: "Male"})  # Convert 0/1 to labels
        
        # Calculate gender distribution percentages
        gender_counts = pie_df["gender"].value_counts(normalize=True) * 100
        male_percentage = gender_counts.get("Male", 0)
        female_percentage = gender_counts.get("Female", 0)
    
        # Display dynamic help text
        st.markdown(f"""
        <div style=" 
                    font-size: 20px; 
                    font-weight: bold; 
                    text-align: center;">
            Gender Distribution: Males - {male_percentage:.1f}%, Females - {female_percentage:.1f}%
        </div>
        """, unsafe_allow_html=True)
    
        # Generate pie chart with readable labels
        fig = px.pie(pie_df, names='gender', title='Gender Distribution',
                     color_discrete_sequence=px.colors.sequential.Aggrnyl)
    
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(width=400, height=500)
    
        st.plotly_chart(fig, use_container_width=True)


    elif chart_type == "Histogram":
        st.markdown('<h3 style="color:#000000;">📊 Tenure Histogram</h3>', unsafe_allow_html=True)
        fig = px.histogram(filtered_df, x='tenure', nbins=20, title='Tenure Distribution',
                           color_discrete_sequence=['#00F3FF'])
        fig.update_layout(bargap=0.1)
        fig.update_layout(width=1000, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Funnel Chart":
        # Funnel Chart: Customer Retention Flow
        st.markdown("""
            <h3 style="color:#00000;">📊 Customer Retention Funnel</h3>
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

    elif chart_type == "Sunburst Chart":
        # Sunburst Chart: Customer Breakdown by Contract, Payment Method, and Churn
    
        st.markdown("""
                <h3 style="color:#00000;">📌 Sunburst Chart </h3>
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

    elif chart_type == "Service Usage Patterns":
        st.markdown("""
            <h3 style="color:#00000;">📊 Service Usage Patterns</h3>
        """, unsafe_allow_html=True)
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
elif option == "📈 Model Validation & Analysis":
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

    st.markdown("""
    <style>
        div.stButton > button {
            font-weight: bold;
            font-size: 16px;
            background-color: #FFA500 !important;  /* Orange Button */
            color: white !important;  /* White Text */
            border-radius: 8px;
            padding: 10px 15px;
            border: none;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #FF6347 !important;  /* Tomato Red on Hover */
        }
    </style>
    
""", unsafe_allow_html=True)

# 📌 Heading Above Buttons
    st.markdown("""
    <h3 style="text-align: center; color: #000000; font-weight: bold; margin-bottom: 10px;">
        Select Evaluation Metrics
    </h3>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)


    # Create a session state variable to store selected button
    if "selected_metric" not in st.session_state:
        st.session_state.selected_metric = "conf_matrix"  # Default to Confusion Matrix
    
    # Define button actions
    with col1:
        if st.button("🔍 Confusion Matrix", key="conf_matrix"):
            st.session_state.selected_metric = "conf_matrix"
    
    with col2:
        if st.button("📊 Feature Importance", key="feat_imp"):
            st.session_state.selected_metric = "feat_imp"
    
    with col3:
        if st.button("🔗 Correlation Matrix", key="corr_matrix"):
            st.session_state.selected_metric = "corr_matrix"
    
    st.markdown("</div>", unsafe_allow_html=True)  # Closing the box container
    
    
    # 📌 Display Confusion Matrix
    if st.session_state.selected_metric == "conf_matrix":
        st.markdown('<h3 style="text-align: center; color: #000000;">📉 Confusion Matrix</h3>', unsafe_allow_html=True)
    
        # Generate a Random Confusion Matrix for Demonstration (Replace with Actual Model Output)
        y_true = filtered_df["Churn"]
        y_pred = filtered_df["Churn"].sample(frac=1, random_state=42)  # Dummy shuffled values (Replace with model predictions)
        cm = confusion_matrix(y_true, y_pred)
    
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
    
    # 📌 Display Feature Importance
    elif st.session_state.selected_metric == "feat_imp":
        st.markdown('<h3 style="text-align: center; color: #000000;">📊 Feature Importance</h3>', unsafe_allow_html=True)
    
        fig, ax = plt.subplots(figsize=(5, 3))
    
        # Ensure bars are light yellow
        feature_importance.plot(kind='barh', ax=ax, color='#FFD700', edgecolor='black')  # Light Yellow Bars
        
        # Update axis labels with smaller font size
        ax.set_xlabel("Importance Score", fontsize=10, fontweight='bold')
        ax.set_ylabel("Features", fontsize=10, fontweight='bold')
    
        # Improve visual styling
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        st.pyplot(fig)
    
    # 📌 Display Correlation Matrix
    elif st.session_state.selected_metric == "corr_matrix":
        st.markdown('<h3 style="text-align: center; color: #000000;">🔗 Correlation Matrix</h3>', unsafe_allow_html=True)
    
        # Select only numeric columns for correlation
        numeric_df = filtered_df.select_dtypes(include=['number'])
    
        # Ensure there are numeric columns before plotting
        if numeric_df.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ No numeric columns available for correlation matrix.")
    
    
        # Add more space
        st.markdown("<br><br>", unsafe_allow_html=True)
    
# Customer Profile Analysis Page
if option == "🎯 Customer Profile Analysis":
    st.title("🎯 Customer Profile Analysis: Understanding Your Customers")
    
    st.markdown("""
    Gain deep insights into customer behaviors, preferences, and churn patterns.
    Use this data to develop targeted strategies and improve retention.
    """)
    
    # Gender and Churn Analysis
    st.subheader("🧑‍🤝‍🧑 Gender Distribution & Churn")
    filtered_df['gender'] = filtered_df['gender'].map({1: 'Male', 0: 'Female'})
    gender_df = filtered_df.groupby(['gender', 'Churn']).size().reset_index(name='count')
    fig = px.sunburst(gender_df, path=['gender', 'Churn'], values='count',
                      title="Gender Influence on Churn",
                      color='Churn', color_discrete_map={'Yes': '#1656AD', 'No': '#00B496'},color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)
    
    # Age and Tenure Influence
    st.subheader("⏳ Customer Tenure & Monthly Charges")
    fig = px.scatter(filtered_df, x='tenure', y='MonthlyCharges', color='Churn',
                     title="Tenure vs Monthly Charges: Who is at Risk?",
                     color_discrete_map={'Yes': '#E74C3C', 'No': '#2ECC71'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Contract Type vs. Churn
    st.subheader("📜 Contract Type & Churn")
    contract_df = filtered_df.groupby(['Contract', 'Churn']).size().reset_index(name='count')
    fig = px.bar(contract_df, x='Contract', y='count', color='Churn',
                 title="Which Contract Type is More Stable?",
                 barmode='group', color_discrete_map={'Yes': '#D35400', 'No': '#1ABC9C'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer Engagement Metrics
    st.subheader("💳 Payment Method & Engagement")
    filtered_df['PaymentMethod'] = filtered_df['PaymentMethod'].map({0: 'Electronic check', 1: 'Mailed check',2: 'Bank transfer', 3: 'Automatic'})
    fig = px.pie(filtered_df, names='PaymentMethod', title="Preferred Payment Methods",
                 color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📌 Takeaway Insights:")
    st.markdown("- Customers with **month-to-month contracts** are more likely to churn.")
    st.markdown("- **Electronic Check users** show higher churn rates – possibly due to dissatisfaction with billing.")
    st.markdown("- Customers with longer **tenure** tend to have higher loyalty.")
    st.markdown("- **High Monthly Charges** could be a red flag leading to increased churn risk.")
    
    st.markdown("🚀 Use these insights to improve retention strategies and reduce churn!")




    


