import PyPDF2
from transformers import pipeline

class PaperSummarizer:
    def __init__(self):
        """
        Initialize the AI Model. 
        We use google/flan-t5-base because it's good at following instructions
        and answering specific questions about text (like 'summarize this' or 'what are the key findings').
        """
        # We load a text2text-generation pipeline. 
        # The first time you run this, it will download the model (takes ~1 GB of space).
        self.model = pipeline("text2text-generation", model="google/flan-t5-base")
        
        # We load a Question-Answering pipeline to extract answers from the text.
        self.qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

    def extract_text_from_pdf(self, pdf_path):
        """
        Helper method to read a PDF file and extract its text content.
        For a fast demo, we'll only extract the first 3 pages to save time/memory.
        """
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Read up to the first 3 pages
            num_pages = min(3, len(reader.pages))
            
            for i in range(num_pages):
                page = reader.pages[i]
                text += page.extract_text() + " "
                
        # Clean up the text by removing extra whitespace
        return " ".join(text.split())

    def process_pdf(self, pdf_path):
        """
        Main pipeline: Extract PDF text, then generate three different types of insights.
        """
        # 1. Extract the text
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        # In a real app, papers are very long (>10,000 words).
        # Transformer models have a context limit (e.g. 512 tokens).
        # For simplicity in this beginner project, we take the first 3500 characters
        # (which roughly equals ~1000 tokens, but T5-base cuts off context after 512).
        input_text = raw_text[:3500] 

        # 2. Generate the results by asking our T5 model specific questions
        abstract = self._generate_answer(
            f"Summarize the main idea of this research paper abstractly:\n\n{input_text}"
        )
        
        findings = self._generate_answer(
            f"What are the key findings or results of this paper? Provide in a brief paragraph:\n\n{input_text}"
        )
        
        simple_explanation = self._generate_answer(
            f"Explain this research paper like I'm a 5 year old:\n\n{input_text}"
        )

        return {
            "abstract": abstract,
            "findings": findings,
            "simple_explanation": simple_explanation,
            "context": input_text
        }

    def generate_suggested_questions(self, text):
        """
        Generate 5-8 suggested questions based on the paper's content.
        """
        prompt = f"Based on this research paper excerpt, generate 5 useful questions a reader might ask to learn more about this work. Format as a bulleted list:\n\n{text[:2000]}"
        
        try:
            result = self.model(
                prompt,
                max_length=150,
                min_length=20,
                do_sample=False
            )
            generated = result[0]['generated_text']
            
            # Simple parsing: try to split by common list indicators if it generated a block of text
            questions = [q.strip() for q in generated.replace('?', '?|').split('|') if len(q.strip()) > 10 and '?' in q]
            
            # Fallback if the AI didn't format it well
            if len(questions) < 3:
                return [
                    "What is the main contribution of this paper?",
                    "What problem is being solved?",
                    "What methodology is used?",
                    "What datasets were used for experiments?",
                    "What are the limitations of this work?",
                    "What future work is suggested?"
                ]
            
            # Limit to 5-8 questions
            return questions[:8]
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return [
                "What problem does this paper address?",
                "What method or model is proposed?",
                "What dataset was used?",
                "What are the key findings?"
            ]

    def answer_question(self, question, context):
        """
        Use the QA pipeline to answer a specific question using the paper's text as context.
        """
        try:
            result = self.qa_model(question=question, context=context)
            # The QA model returns a dict with 'score', 'start', 'end', and 'answer'
            # If confidence is very low, we might want to warn the user, but for now we just return it.
            return result['answer']
        except Exception as e:
            print(f"Error answering question: {e}")
            return "I'm sorry, I couldn't find a clear answer to that question in the paper's text."

    def _generate_answer(self, prompt):
        """
        Helper method to run a specific prompt through the Hugging Face transformer.
        We adjust max_length and min_length for reasonably sized outputs.
        """
        try:
            result = self.model(
                prompt,
                max_length=150,
                min_length=30,
                do_sample=False  # Deterministic output
            )
            # Extracted generated text
            return result[0]['generated_text']
        except Exception as e:
            print(f"Error during AI generation: {e}")
            return "Unable to generate summary. The text might be too complex or an error occurred."
