import streamlit as st
import helper
import pickle

# Load the model
model = pickle.load(open(r'model\model.pkl', 'rb'))
vectorized = pickle.load(open(r'model\vectorizer.pkl', 'rb'))


st.title("Sentiment Analysis App using ML")
st.write("Analyze the sentiment of a given review as Positive or Negative.")


text = st.text_input("Please enter your review")

if st.button("pedict"):
    if not text.strip():
        st.warning("Please enter a valid review.")
    else:
        # Preprocess and predict
        token = helper.preprocessing_step(text)
        vectorized_text = vectorized.transform([token])
        prediction = model.predict(vectorized_text)
        if prediction[0] == 1:
            sentiment = "Positive"
            st.success(f"The sentiment of the review is: **{sentiment}**")
            st.balloons()
        else:
            sentiment = "Negative"
            st.error(f"The sentiment of the review is: **{sentiment}**")
