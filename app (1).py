import os
if not os.path.exists('static'):
    os.makedirs('static')
import gradio as gr    
import pdfplumber
import fitz  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import io
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
key = os.getenv('api_key')


# Initialize the models and embeddings
primary_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=key,
    temperature=0.2,
    convert_system_message_to_human=True
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=key
)

fallback_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=key,
    temperature=0.5,
    convert_system_message_to_human=True
)

def validate_pdf(file):
    """Validate if the uploaded file is a valid PDF."""
    try:
        file.seek(0)  # Ensure the file pointer is at the start
        pdf = pdfplumber.open(file)
        if pdf.pages:
            return True
    except Exception as e:
        print(f"Validation error: {e}")
    return False


def load_pdf(file):
    pages = []
    try:
        file.seek(0)  # Ensure file pointer is at the start
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
                else:
                    pages.append("No text found on this page.")
    except Exception as e:
        return f"Error reading the PDF file: {e}"
    return pages


def get_fallback_answer(question):
    """Get an answer using the fallback model if the PDF content is not sufficient."""
    try:
        prompt = (
            "Please provide a comprehensive answer to the following question based on general knowledge:\n\n"
            f"Question: {question}"
        )
        result = fallback_model.invoke(prompt)

        if hasattr(result, 'content'):
            return result.content
        else:
            return "No answer found."
    except Exception as e:
        return f"An error occurred with the fallback model: {e}"

def process_pdf_and_answer_question(pdf_file, question):
    try:
        # Convert bytes to a file-like object
        pdf_file = io.BytesIO(pdf_file)

        # Validate the PDF
        if not validate_pdf(pdf_file):
            return "The uploaded file is not a valid PDF."

        # Reset file pointer to the beginning after validation
        pdf_file.seek(0)

        pages = load_pdf(pdf_file)

        if isinstance(pages, str):
            return pages

        context = "\n\n".join(pages)  # Combine all page contents into a single string

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        texts = text_splitter.split_text(context)

        # Create vector index
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})

        # Initialize the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            primary_model,
            retriever=vector_index,
            return_source_documents=True
        )

        # Construct the prompt with additional context
        prompt = (
            "You are an expert in telling everything in detail even if the detail is not sufficient. "
            "Please provide a comprehensive answer based on the following PDF content and your own knowledge base.\n\n"
            f"PDF Content:\n{context}\n\n"
            f"Question: {question}"
        )

        # Perform the query
        result = qa_chain({"query": prompt})

        # Retrieve the answer from the PDF content
        pdf_answer = result.get("result", None)

        # If no answer is found from the PDF content, use the fallback model
        if not pdf_answer or pdf_answer == "No answer found.":
            pdf_answer = get_fallback_answer(question)

        # Clean the answer by removing asterisks and hashes
        clean_answer = pdf_answer.replace('*', '').replace('#', '').strip()

        # Return the cleaned answer
        return clean_answer if clean_answer else "No answer could be generated."

    except Exception as e:
        return f"An error occurred: {e}"

def process_chatbot_answer(question):
    """Process the user's question for the chatbot."""
    answer = get_fallback_answer(question)
    # Clean the answer by removing asterisks and hashes
    clean_answer = answer.replace('*', '').replace('#', '').strip()
    return clean_answer if clean_answer else "No answer could be generated."

# Define Gradio interface
iface = gr.Interface(
    fn=lambda choice, pdf_file, question: process_pdf_and_answer_question(pdf_file, question) if choice == "PDF Q&A" else process_chatbot_answer(question),
    inputs=[
        gr.Radio(label="Choose an Option", choices=["PDF Q&A", "Chatbot"], type="value"),
        gr.File(label="Upload PDF", type="binary"),  # 'binary' type for direct bytes handling
        gr.Textbox(label="Ask a Question", lines=2)
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Aura-AI RAG Application",
    description="Choose between a PDF Q&A system and a general-purpose chatbot. Upload a PDF to ask questions about its content or ask general questions to the chatbot."
)

# Launch the Gradio app
iface.launch()
