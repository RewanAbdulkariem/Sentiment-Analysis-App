import streamlit as st
import helper
import pickle
import downloader
import os
import pickle

# Use os.path.join for cross-platform compatibility
model_path = os.path.join('model', 'model.pkl')
model = pickle.load(open(model_path, 'rb'))
vectorized_path = os.path.join('model', 'vectorizer.pkl')

vectorized = pickle.load(open(vectorized_path, 'rb'))


# App title and description
st.title("🎭 Sentiment Analysis App using Machine Learning")
st.text("Analyze the sentiment of provided reviews as Positive or Negative.")

text = st.text_input("Please enter your review")

if st.button("🔍 Analyze Sentiment"):
    if not text.strip():
        st.warning("Please enter a valid review.")
    else:
        with st.spinner('Analyzing the review...'):
            # Preprocess the text and make a prediction
            token = helper.preprocessing_step(text)
            vectorized_text = vectorized.transform([token])
            prediction = model.predict(vectorized_text)
            confidence = model.predict_proba(vectorized_text).max()

            if prediction[0] == 1:
                sentiment = "Positive 😄"
                st.success(f"Prediction Result: **{sentiment}**")
                st.write(f"🎯 Confidence Score: {confidence:.2f}")
                st.balloons()
            else:
                sentiment = "Negative 🙁"
                st.error(f"Prediction Result: **{sentiment}**")
                st.write(f"🎯 Confidence Score: {confidence:.2f}")

# Page footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Developed by <strong>Rewan Khaled</strong>.</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)