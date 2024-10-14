from flask import Flask, request, render_template
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
try:
    with open(r'C:\Users\ajeethan\sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open(r'C:\Users\ajeethan\tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except (EOFError, FileNotFoundError) as e:
    print(f"Error loading model or vectorizer: {e}")

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the review from the form
        review = request.form['review']
        
        # Clean the review
        cleaned_review = review.lower()
        
        # Vectorize the review using the TF-IDF vectorizer
        review_tfidf = vectorizer.transform([cleaned_review])
        
        # Make a prediction
        prediction = model.predict(review_tfidf)[0]
        
        # Return the result
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        return render_template('index.html', prediction_text=f'This review is {sentiment}')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
