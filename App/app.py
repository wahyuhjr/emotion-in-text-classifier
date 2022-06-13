from secrets import choice
from turtle import color
from unittest import result
import altair as alt
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


emotion_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
                       "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}


def main():
    st.title("Emotion Detection App")
    menu = ["Home", "Monitor","About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
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
                
                st.success("Prediction")
                emoji_icon = emotion_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))
                
            with col2:
                st.success("Prediction Probability")
                # st.write(probability)
                proba_df = pd.DataFrame(probability, columns = pipe_lr.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotion", "probability"]
                
                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotion', y='probability',color='emotion')
                st.altair_chart(fig, use_container_width=True)

    elif pilih == "Monitor":
        st.subheader("Mointor App")
        
    elif pilih == "About":
        st.subheader("About")

if __name__ == '__main__':
    main()
