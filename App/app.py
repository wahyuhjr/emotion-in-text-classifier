from secrets import choice
from unittest import result
import streamlit as st
import pandas as pd
import numpy as np
import joblib

pipe_lr = joblib.load(open("models/emotion.pkl","rb"))
#fxn
def predict_emotion(docx):
    result = pipe_lr.predict([docx])
    return result[0]

def get_prediction_proba(docx):
    result = pipe_lr.predict_proba([docx])
    return result

def main():
    st.title("Emotion Detection App")
    menu = ["Home", "Monitor","About"]
    pilih = st.sidebar.selectbox("Menu", menu)
    
    if pilih == "Home":
        st.subheader("Home-Emotion in Text")
        
        with st.form(key='myform_emotion_form'):
            raw_text = st.text_area("masukan kalimat")
            submit_text = st.form_submit_button(label='Submit')
            
        if submit_text:    
            col1,col2 = st.beta_columns(2)
            
            prediction = predict_emotion(raw_text)
            probability = get_prediction_proba(raw_text)
            
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                
                st.success("Predicted")
                st.write(prediction)

            with col2:
                st.success("Prediction Probability")
                st.write(probability)
        
    elif pilih == "Monitor":
        st.subheader("Mointor App")
        
    elif pilih == "About":
        st.subheader("About")

if __name__ == '__main__':
    main()
