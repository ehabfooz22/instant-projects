
import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder


featuress = pd.read_csv("data.csv")
target = featuress.drop(columns=['stroke'])

st.set_page_config(page_title="Stroke Prediction", page_icon='images/istockphoto-1250205787-612x612.jpg')
st.title('Stroke Prediction', '\n')
st.image('images/istockphoto-1250205787-612x612.jpg')

st.subheader("Patient's Stroke Forcasting")

st.markdown("""
##### The goal of this application is to classify the patient has a stroke or not based on the characteristics or data of this person.
##### To predict whether the loan is accepted or not, just follow these steps:
##### 1. Enter information describing the patirnt's data.
##### 2. Press the "Predict" button and wait for the result.
# """)
st.image('images/pexels-anna-shvets-4226219.jpg')

expander = st.expander('Group Members For the Project')
expander.markdown("""
            **Eslam Ashraf Mohamed**\n
            **Ehab Tarek Ali**\n
            **Ahmed Abdallah Gamil**\n   
            """)
    
def user_input_features():

    st.sidebar.write('# Fill this form please..')

    gender = st.sidebar.radio("gender",
                                options=(gender for gender in featuress.gender.unique()))

    hypertension = st.sidebar.radio("Hypertension", 
                                options=(hypertension for hypertension in featuress.hypertension.unique()))
    
    heart_disease = st.sidebar.radio("Heart Disease", 
                                options=(heart_disease for heart_disease in featuress.heart_disease.unique()))

    ever_married = st.sidebar.radio("Ever Married", 
                                options=(ever_married for ever_married in featuress.ever_married.unique()))

    work_type = st.sidebar.radio("Work Type", 
                                options=(work_type for work_type in featuress.work_type.unique()))

    Residence_type = st.sidebar.selectbox("Residence Type",
                                    options=(Residence_type for Residence_type in featuress.Residence_type.unique()))

    smoking_status = st.sidebar.selectbox("Smoking Status", 
                                    (smoking_status for smoking_status in featuress.smoking_status.unique()))
    
    age = st.sidebar.slider('What is your Age ?', 0, 100)
    
    bmi = st.sidebar.slider('What is your BMI ?', 0, 100)

    avg_glucose_level = st.sidebar.slider('What is your avgerage glucose level ?', 0, 300)
    

    
    data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": Residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status

        }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

df = pd.concat([input_df,target],axis=0)
for col in df[['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']]:
    df[col] = LabelEncoder().fit_transform(df[col])


RF_MODEL_PATH = joblib.load("Models/model.h5")
RF_SCALER_PATH = joblib.load("Models/scaler.h5")

scaled_data = RF_SCALER_PATH.transform(df)
prediction_proba = RF_MODEL_PATH.predict_proba(scaled_data)

if st.sidebar.button('Predict'):
    st.sidebar.success(f'# The probability of the patient has a stroke is : {round(prediction_proba[0][1] * 100, 2)}%')
