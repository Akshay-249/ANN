import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import streamlit as st
import pickle


## load trained model
model = tf.keras.models.load_model('model.h5')

# load encoder and scaler pickel files
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encode_geography.pkl', 'rb') as file:
    onehot_encode_geography = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn prediction')

# user input
geography = st.selectbox('Geography', onehot_encode_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 99)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Salary')
tenure = st.slider('Tenure', 0, 10)
num_products = st.slider('Products', 1, 4)
has_cr_card = st.selectbox("Has credit card", [0, 1])
is_active = st.selectbox("is active member", [0, 1])

# prepare inpute data
input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active],
    'EstimatedSalary' : [estimated_salary]
})

# one-hote encode 'Geography'
geography_encoder = onehot_encode_geography.transform([[geography]]).toarray()
geography_encoder_df = pd.DataFrame(geography_encoder, columns=onehot_encode_geography.get_feature_names_out(['Geography']))

# combine one-hot encoded columns with input_data
input_data = pd.concat([input_data.reset_index(drop=True), geography_encoder_df], axis=1)

#sacle input_data
input_data_scaled = scaler.transform(input_data)

# Prediction of churning
predicition = model.predict(input_data_scaled)
predicition_prob = predicition[0][0]

st.write(f'churning probability : {predicition_prob}')

if predicition_prob <0.5:
    st.write ('The Customer is not likely to churn')
else:
    st.write("Customer likely to churn")
    

