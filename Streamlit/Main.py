import streamlit as st
import datetime
import pandas as pd 
import numpy as np 
import pickle

#Multiple page configurations
st.set_page_config(
    page_title="Credit Card Eligibility", #Main.py file as main page
    page_icon="💳",
)

#Background image (WIP)
# st.markdown("""
#     <style>
#         .stApp {
#         background: url("");
#         background-size: cover;
#         }
#     </style>""", unsafe_allow_html=True)

st.title("Prediction of Credit Card approval")

st.sidebar.info("You are now on the eligibility page ✅")

st.write("""
        ### Check if you are eligible in seconds!✅💳
""")

choices = {0: "Y",1: "N"}

def format_func(option):
        return choices[option]

#Get input from user

#gender
GENDER = st.selectbox("Select your Gender",("M","F"),index=None,placeholder="Select your option")

#Birthday_count
b_day = st.date_input("Your birthday date", min_value = datetime.date(1950,1,1))
Birthday_count = np.abs((b_day - datetime.date.today()))
Birthday_count = Birthday_count.days

#Marital_status
Marital_status = st.selectbox("Select your Marital status",("Single / not married","Married","Civil marriage","Separated","Widow"),index=None,placeholder="Select your option")

#Children
CHILDREN = st.slider("How many dependent children are currently under your care or support?",0,14)

#Family_Members
Family_Members = st.slider("How many number of family members?",0,15)


#EDUCATION
EDUCATION = st.selectbox("Select your Education level",("Lower secondary","Secondary / secondary special","Higher education","Academic degree","Incomplete higher"),index=None,placeholder="Select your option")

#Employed_days
employment_choice = st.selectbox('Choose your current employment status',('Employed','Unemployed'))
min_date = datetime.date(1950,1,1)
max_date = datetime.date.today()
if employment_choice == 'Employed':
        #Employed_days
        em = st.date_input("Select your most recent employment date", min_value = min_date,max_value = max_date)
        Employed_days = (em - datetime.date.today())
        Employed_days = Employed_days.days
if employment_choice == 'Unemployed':
        em = st.date_input("Select your the daterange of your unemployment", min_value = min_date,max_value=max_date)
        Employed_days = (datetime.date.today()- em)
        Employed_days = Employed_days.days

#Type_Occupation
Type_Occupation = st.selectbox("Select your Occupation type",("Laborers","Core staff","Managers","Sales staff","Drivers","High skill tech staff","Medicine staff","Accountants","Security staff","Cleaning staff","Cooking staff","Private service staff","Secretaries","Low-skill Laborers","Waiters/barmen staff","HR staff","IT staff","Realty agents"),index=None,placeholder="Select your option")

#Annual_income
Annual_income = st.slider("What is your total yearly earnings?",30000,1500000,step=5000)

#Type_Income
Type_Income = st.selectbox("Select your type of Income",("State servant","Pensioner","Commercial associate","Working"),index=None,placeholder="Select your option")

#Car_owner
Car_Owner = st.selectbox("Do you own ateast one car?",index=None,options = list(choices.keys()),format_func = format_func,placeholder="Select your option")

#Propert_Owner
Propert_Owner = st.selectbox("Do you own ateast one property?",index=None,options = list(choices.keys()),format_func = format_func,placeholder="Select your option")

#Housing_type
Housing_type = st.selectbox("Select your Housing type",("House / apartment","With parents","Rented apartment","Municipal apartment","Co-op apartment","Office apartment"),index=None,placeholder="Select your option")

#Mobile_phone
Mobile_phone = st.selectbox("Do you own a Mobile phone?",index=None,options = list(choices.keys()),format_func = format_func,placeholder="Select your option")


#Work_Phone
Work_Phone = st.selectbox("Do you own a Work phone?",index=None,options = list(choices.keys()),format_func = format_func,placeholder="Select your option")


#Phone
Phone = st.selectbox("Do you own atleast one phone number?",index=None,options = list(choices.keys()),format_func = format_func,placeholder="Select your option")


#EMAIL_ID
EMAIL_ID = st.selectbox("Do you have ateast one email ID created?",index=None,options = list(choices.keys()),format_func = format_func,placeholder="Select your option")


# days to year conversion
d = Birthday_count/365
em_conv = Employed_days/365

#Prediction button
if st.button("Check my approval"):

        with open('https://github.com/povembu/D590-ADS-Project/blob/main/Streamlit/rfc_model.pkl','rb') as fid:
                model = pickle.load(fid)
        
        pred = model.predict([[GENDER, Car_Owner, Propert_Owner, CHILDREN, Annual_income,
       Type_Income, EDUCATION, Marital_status, Housing_type,
       Birthday_count, Employed_days, Mobile_phone, Work_Phone,
       Phone, EMAIL_ID, Type_Occupation, Family_Members,d,
       em_conv]])
        if pred ==0:
                st.success("You have high chances of approval for Credit Card!")
        else:
                st.error("Sorry, your approval chances are low")

