# Sentiment-Analysis-App using ML
This project is a simple web application built with Streamlit for performing sentiment analysis on user-provided reviews. By leveraging machine learning models, the app predicts whether the sentiment of the given text is positive or negative.

### Features
- User Input: Enter any text or review to analyze its sentiment.
- Prediction: The app predicts if the sentiment is positive or negative.
- Interactive UI: Built with Streamlit for a clean and interactive user experience.

# Clone the repository:

```BASH
git clone https://github.com/RewanAbdulkariem/Sentiment-Analysis-App.git
cd sentiment-analysis-app
Create a virtual environment (optional but recommended):
```
```BASH
python -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
Install the required packages
```
```BASH

pip install -r requirements.txt
```
Ensure your model and vectorizer files are in the right location:

The pre-trained model (model.pkl) and the vectorizer (vectorizer.pkl) should be placed inside the model directory.
Run the application:

```BASH
streamlit run app.py
```
### Usage
- Enter any review or piece of text in the provided input field.
- Click on the "Predict" button.
- The app will display the sentiment of the entered text as either Positive or Negative.


