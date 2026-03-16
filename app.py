import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from summarizer import PaperSummarizer

app = Flask(__name__)
# Set uploading limits
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize our AI model
print("Loading AI Model (this might take a few moments the first time)...")
summarizer_model = PaperSummarizer()

@app.route('/', methods=['GET'])
def index():
    """Render the main user interface."""
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    """Handle PDF upload, extract text, and run the AI summary."""
    # Check if a file is in the request
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['pdf_file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.pdf'):
        # Secure the filename to prevent security vulnerabilities
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Save the file temporarily
            file.save(filepath)
            
            # Use our AI logic to process and summarize the PDF
            results = summarizer_model.process_pdf(filepath)
            
            # Clean up the file after processing
            os.remove(filepath)
            
            # Return success response with data
            return jsonify({
                'success': True,
                'abstract': results['abstract'],
                'findings': results['findings'],
                'simple_explanation': results['simple_explanation'],
                'context': results['context']
            })
            
        except Exception as e:
            # Clean up the file if there's an error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)})
            
    return jsonify({'error': 'Invalid file type. Please upload a PDF.'})

@app.route('/ask', methods=['POST'])
def ask():
    """Answer a specific question based on the paper's context."""
    data = request.json
    if not data or 'question' not in data or 'context' not in data:
        return jsonify({'error': 'Missing question or context'}), 400
        
    answer = summarizer_model.answer_question(data['question'], data['context'])
    return jsonify({'answer': answer})

@app.route('/suggest-questions', methods=['POST'])
def suggest_questions():
    """Generate suggested questions based on the paper's context."""
    data = request.json
    if not data or 'context' not in data:
        return jsonify({'error': 'Missing context'}), 400
        
    questions = summarizer_model.generate_suggested_questions(data['context'])
    return jsonify({'questions': questions})

if __name__ == '__main__':
    # Run the server on localhost port 5000
    print("Starting server... Go to http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
