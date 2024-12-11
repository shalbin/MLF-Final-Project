from flask import Flask, request, render_template
from transformers import pipeline

# Initialize Flask app
app = Flask("__name__")

# Load Hugging Face sentiment analysis pipeline
sent_pipeline = pipeline("sentiment-analysis")

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def sentimentAnalysis():
    try:
        # Get user input from the form
        input_text = request.form['query1']
        
        # Perform sentiment analysis
        predictions = sent_pipeline(input_text)
        
        # Extract prediction details
        label = predictions[0]['label']
        score = predictions[0]['score']

        # Prepare output
        output_message = f"Sentiment: {label} (Confidence: {score:.2%})"
        
        # Return result to the web page
        return render_template('home.html', 
                               output1=output_message, 
                               query1=input_text)
    except Exception as e:
        return render_template('home.html', 
                               output1=f"Error: {str(e)}", 
                               query1="")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
