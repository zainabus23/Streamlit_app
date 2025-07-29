import pandas as pd
import joblib
import streamlit as st


#bring in pur model 
#import all encoders
country_enc = joblib.load ("country_encoder.sav")
educ_enc = joblib.load("educ_encoder.sav")
gen_enc = joblib.load("genc_encoder.sav")
jobtype_enc = joblib.load("jobtype_encoder.sav")
loc_enc = joblib.load("jobtype_encoder.sav")
rel_enc = joblib.load("rel_head_encoder.sav")
cell_enc = joblib.load("cellacess_encoder.sav")
marit_enc = joblib.load("marit_encoder.sav")
year_enc = joblib.load("year_encoder.sav")
model = joblib.load("rf_project2_model.sav")



st.title("Supervised Machine Learning Project")
st.header("Customer Bank Account Usage prediction")

column_desc = pd.read_csv("VariableDefinitions (2).csv",
                          header=None,
                          skiprows=1,
                          names= ["country", "Country interviewee is in"])
column_desc = column_desc.iloc[1:14]
st.write("Dataset description: The dataset contains demographic information and what financial services are used by approximately 33,600 individuals across East Africa")
st.dataframe(column_desc)



with st.sidebar:
    year = st.number_input("Year survey was done in", 0,2)
    country_encoder = st.selectbox("country interviewr is in",[i for i in country_enc.classes_])
    educ_enc = st.selectbox("highest educational level", [i for i in educ_enc.classes_])
    marit_encoder = st.selectbox("marital status", [i for i in marit_enc.classes_])
    jobtype_encoder= st.selectbox("job type", [i for i in jobtype_enc.classes_])
    rel_head_encoder = st.selectbox("relationship status", [i for i in rel_enc.classes_])
    genc_encoder = st.selectbox("gender", [i for i in gen_enc.classes_])
    location_encoder= st.selectbox("location", [i for i in loc_enc.classes_])
    cellaccess_encoder = st.selectbox("cellphone access", [i for i in cell_enc.classes_])
    household_size = st.number_input("household size", 1,21)
    age_of_respondent= st.number_input("age of respondent", 16,100)



features = [year,
            country_encoder.transform([country]),
            educ_enc.transform([educational_level]),
            marit_enc.transform([marital_status]),
            jobtype_enc.transform([job_type]),
            rel_enc.transfrom([relationship_status]),
            gen_enc.transform([gender_of_respondent]),
            loc_enc.transform([location_type]),
            cell_enc.transform([cellphone_access]),
            household_size,
            age_of_respondent
            ]



user_data = pd.DataFrame([features], columns=['country', 'educational_level', 'marital_status', 'job_type',
        'relationship_status', 'gender_of_respondent', 'location_type', 'cellphone_access', 'household_size', 'age_of_respondent',
        ])



if st.sidebar.button('Predict'):
    st.dataframe(user_data)
    prediction = model.predict(user_data)
    if prediction == 0:
        st.write('Individual not likely to own a bank account')
    else:
        st.write('Inidividual likely to own a bank account')