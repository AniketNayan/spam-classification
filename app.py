from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Initialize stemmer
ps = PorterStemmer()

# Load TF-IDF vectorizer and model
cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


@app.route('/')
def home():
    return render_template('index.html', prediction_text=None, input_text="")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_sms = request.form.get('message', '')

        if input_sms.strip() == "":
            return render_template('index.html', prediction_text="⚠️ Please enter a message.", input_text=input_sms)

        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = cv.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display
        prediction = "🚫 Spam" if result == 1 else "✅ Not Spam"

        return render_template('index.html', prediction_text=prediction, input_text=input_sms)


if __name__ == '__main__':
    app.run(debug=True)
