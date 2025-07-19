import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("model.pkl")
scaler = pickle.load(open("scaler.pkl","rb"))
label_encoders = pickle.load(open("label_encoders.pkl","rb"))

st.set_page_config(page_title="Employee Salary prediction", page_icon="👨‍💼",layout="centered")

st.title("👨‍💼 Employee Salary Prediction Application")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# Define dropdown options for each feature
workclass_options = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Others']
marital_status_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
education_options = ['7th-8th','9th','10th','11th','12th','HS-grad','Some-college','Assoc-voc','Assoc-acdm','Bachelors','Masters','Prof-school','Doctorate']
occupation_options = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                      'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
relationship_options = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
race_options = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
gender_options = ['Male', 'Female']
native_country_options = [
    'United-States', 'Mexico', 'Others', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada',
    'El-Salvador', 'India', 'Cuba', 'England', 'China', 'South', 'Jamaica', 'Italy',
    'Dominican-Republic', 'Japan', 'Guatemala', 'Poland', 'Vietnam', 'Columbia', 'Haiti',
    'Portugal', 'Taiwan', 'Iran', 'Nicaragua', 'Greece', 'Peru', 'Ecuador', 'France',
    'Ireland', 'Thailand', 'Hong', 'Cambodia', 'Trinadad&Tobago', 'Laos',
    'Outlying-US(Guam-USVI-etc)', 'Yugoslavia', 'Scotland', 'Honduras', 'Hungary',
    'Holand-Netherlands'
]

# Collect user input
age = st.sidebar.slider("Age",17,65,30,help="Age of the individual in years.")
education = st.sidebar.selectbox("Education", education_options)
experience = st.sidebar.slider("Experience",0,50,0)
workclass = st.sidebar.selectbox("Workclass", workclass_options)
marital_status = st.sidebar.selectbox("Marital Status", marital_status_options)
occupation = st.sidebar.selectbox("Occupation", occupation_options)
relationship = st.sidebar.selectbox("Relationship", relationship_options, help="Household relationship (e.g., Husband, Not-in-family).")
race = st.sidebar.selectbox("Race", race_options)
gender = st.sidebar.selectbox("Gender", gender_options)
capital_gain = st.sidebar.slider("Capital Gain",0,100000,0,1000, help="Investment income (excluding wages). Usually zero for most people.")
capital_loss = st.sidebar.slider("Capital Loss",0,5000,0,100,help="Investment loss. Often zero.")
hours_per_week = st.sidebar.slider("Hours per Week",1,100,50)
native_country = st.sidebar.selectbox("Native Country", native_country_options)

#
education_num = education_options.index(education) + 4

# Input Details
input_df = pd.DataFrame({
    'Age': [age],
    'Experience': [experience],
    'Workclass': [workclass],
    'Education': [education],
    'Marital_status': [marital_status],
    'Occupation': [occupation],
    'Relationship': [relationship],
    'Race': [race],
    'Gender': [gender],
    'Capital_gain': [capital_gain],
    'Capital_loss': [capital_loss],
    'Hours_per_week': [hours_per_week],
    'Native_country': [native_country]
})

st.write("## Input Provided")
st.write(input_df)

# Encoding categorical features 
def encode_inputs(age, experience, workclass, education_num, marital_status,
                  occupation, relationship, race, gender, capital_gain,
                  capital_loss, hours_per_week, native_country):
    
    # Label encode categorical fields
    workclass_enc = label_encoders['workclass'].transform([workclass])[0]
    marital_status_enc = label_encoders['marital-status'].transform([marital_status])[0]
    occupation_enc = label_encoders['occupation'].transform([occupation])[0]
    relationship_enc = label_encoders['relationship'].transform([relationship])[0]
    race_enc = label_encoders['race'].transform([race])[0]
    gender_enc = label_encoders['gender'].transform([gender])[0]
    native_country_enc = label_encoders['native-country'].transform([native_country])[0]

    # Feature vector
    input_vector = [[
        age, experience, workclass_enc, education_num, marital_status_enc,
        occupation_enc, relationship_enc, race_enc, gender_enc,
        capital_gain, capital_loss, hours_per_week, native_country_enc
    ]]

    # Scale using MinMaxScaler
    scaled_input = scaler.transform(input_vector)
    return scaled_input
if st.button("Predict Salary"):
    processed_input = encode_inputs(
        age, experience, workclass, education_num, marital_status,
        occupation, relationship, race, gender, capital_gain,
        capital_loss, hours_per_week, native_country
    )

    prediction = model.predict(processed_input)
    if prediction[0] == 0:
        st.success("✔ Predicted Salary Group: Below or Equal to $ 50K")
    else: 
        st.success("✔ Predicted Salary Group: Above $ 50K")

st.markdown("---")
st.write("## 📂 Batch Prediction")
st.markdown("It allows you to upload a CSV file containing data for multiple records (e.g., 13 columns for different employees), and get predictions for all of them in one go — displayed as a table or downloadable file.")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        expected_cols = ['age', 'experience', 'workclass', 'educational-num',
                         'marital-status', 'occupation', 'relationship', 'race', 'gender',
                         'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        
        if all(col in batch_data.columns for col in expected_cols):
            st.success("✅ CSV loaded successfully!")
            st.markdown("---")
            st.write("Uploaded data preview:", batch_data.head())

            batch_data['workclass'] = label_encoders['workclass'].transform(batch_data['workclass'])
            batch_data['marital-status'] = label_encoders['marital-status'].transform(batch_data['marital-status'])
            batch_data['occupation'] = label_encoders['occupation'].transform(batch_data['occupation'])
            batch_data['relationship'] = label_encoders['relationship'].transform(batch_data['relationship'])
            batch_data['race'] = label_encoders['race'].transform(batch_data['race'])
            batch_data['gender'] = label_encoders['gender'].transform(batch_data['gender'])
            batch_data['native-country'] = label_encoders['native-country'].transform(batch_data['native-country'])

            batch_data_sc = scaler.transform(batch_data)
            batch_preds = model.predict(batch_data_sc)
            batch_data_df = pd.DataFrame(batch_data_sc,columns= batch_data.columns)
            batch_data_df['PredictedClass'] = batch_preds
            st.write("✅ Predictions:")
            st.markdown("PredictedClass = 0 : employee earns ≤50K")
            st.markdown("PredictedClass = 1 : employee earns >50K")
            st.write(batch_data_df.head())
            csv = batch_data_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
        else:
            st.error("❌ Uploaded CSV is missing one or more required columns.")
    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")



# To run this code type in terminal: python -m streamlit run app.py