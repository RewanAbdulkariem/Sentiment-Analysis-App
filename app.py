import streamlit as st
import helper
import pickle

import os
import pickle

# Use os.path.join for cross-platform compatibility
model_path = os.path.join('model', 'model.pkl')
model = pickle.load(open(model_path, 'rb'))
vectorized_path = os.path.join('model', 'vectorizer.pkl')

vectorized = pickle.load(open(vectorized_path, 'rb'))


st.title("Sentiment Analysis App using ML")
st.write("Analyze the sentiment of a given review as Positive or Negative.")


text = st.text_input("Please enter your review")
# Preprocess and predict
token = helper.preprocessing_step(text)
vectorized_text = vectorized.transform([token])
prediction = model.predict(vectorized_text)

if st.button("pedict"):
    if not text.strip():
        st.warning("Please enter a valid review.")
    else:
        if prediction[0] == 1:
            sentiment = "Positive"
            st.success(f"The sentiment of the review is: **{sentiment}**")
            st.balloons()
        else:
            sentiment = "Negative"
            st.error(f"The sentiment of the review is: **{sentiment}**")
