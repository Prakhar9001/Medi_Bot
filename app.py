from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import os
from PIL import Image
import torch
from torchvision import transforms
from PIL import Image
import io
import sys
import os
from Lung_Disease_Detection_CNN_Model import predict_lung_disease

# Define paths
DB_FAISS_PATH = os.path.join(os.path.dirname(__file__), "vectorstores", "db_faiss")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "llama-2-7b-chat.ggmlv3.q8_0-002.bin")
uploads_dir = "edumit/llama2-PDF-Chatbot/uploads"
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)
# Custom prompt template
custom_prompt_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Provide detailed answers that are easy to understand.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:"""
def predict_fracture(image):
    # Convert to RGB if grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    # Get results
    has_fracture = bool(prediction.item())
    confidence_score = confidence.item()
    
    # Create result message
    if has_fracture:
        result = f"Fracture detected with {confidence_score:.2%} confidence"
    else:
        result = f"No fracture detected ({confidence_score:.2%} confidence)"
    
    return result

def set_custom_prompt():
    """Create a prompt template for QA"""
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )
    return prompt

def load_llm():
    """Load the Llama 2 model"""
    # Ensure model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def setup_qa_chain():
    """Set up the QA chain with embeddings, vector store, and LLM"""
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Use 'cuda' if you have GPU support
        )
        
        # Load the vector store with allow_dangerous_deserialization
        if not os.path.exists(DB_FAISS_PATH):
            raise FileNotFoundError(
                f"Vector store not found at {DB_FAISS_PATH}. Please run ingest.py first."
            )
        
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings,
            allow_dangerous_deserialization=True  # Added this flag
        )
        
        # Load the LLM
        llm = load_llm()
        
        # Create the prompt
        qa_prompt = set_custom_prompt()
        
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': qa_prompt}
        )
        
        return qa_chain
        
    except Exception as e:
        print(f"Error setting up QA chain: {str(e)}")
        raise

# ...existing code...

@cl.on_chat_start
async def start():
    """Start the chat and initialize the QA chain"""
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load the FAISS index with allow_dangerous_deserialization=True
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Initialize the language model
        llm = CTransformers(
            model=MODEL_PATH,
            model_type="llama",
            config={"max_new_tokens": 512, "temperature": 0.7, "context_length": 2048}
        )
        
        # Create a prompt template
        prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        # Store the chain in the user session
        cl.user_session.set("chain", qa_chain)
        
        # Welcome message
        await cl.Message(
            content="""Hello I am MediBot!
            How can I help you today?"""
        ).send()
        
        # Add an upload image button with action
        upload_msg = await cl.Message(
            content="📤Upload Image",
            actions=[
                cl.Action(
                    name="upload_image", 
                    label="Upload X-ray Image", 
                    description="Upload an X-ray image for lung disease detection",
                    payload={"accept": ["image/jpeg", "image/png", "image/gif"]}
                )
            ]
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"⚠️ Error initializing the chat bot: {str(e)}"
        ).send()

# ...existing code...
# Handle image upload action
@cl.action_callback("upload_image")
async def on_image_upload(action):
    """Handle image upload action"""
    try:
        files = await cl.AskFileMessage(
            content="Please upload your X-ray or lung image for disease detection",
            accept=["image/jpeg", "image/png", "image/gif"],
            max_size_mb=5,
            raise_on_timeout=False,
        ).send()
        if not files:
            await cl.Message(content="No file was uploaded. Please try again.").send()
            return
        file = files[0]
        # Robustly get file content
        if hasattr(file, "content"):
            file_content = file.content
        elif hasattr(file, "read"):
            file_content = await file.read()
        elif hasattr(file, "path"):
            with open(file.path, "rb") as f:
                file_content = f.read()
        else:
            raise Exception("Could not access file content for uploaded file.")
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(file_content))
        # Predict using lung disease model
        pred_class = predict_lung_disease(image)
        await cl.Message(content=f"Image '{file.name}' received. Predicted disease: {pred_class}").send()
    except Exception as e:
        await cl.Message(content=f"Error processing the image: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    try:
        chain = cl.user_session.get("chain")
        if chain is None:
            raise ValueError("QA chain not initialized. Please restart the chat.")
        
        # Process the message content
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True
        
        response = await chain.ainvoke(
            {"query": message.content},
            callbacks=[cb]
        )
        
        
        answer = response["result"]
        #sources = response["source_documents"]
        
        #if sources:
        #    answer += "\n\nSources:\n"
        #    for i, source in enumerate(sources, 1):
        #        answer += f"\n{i}. {source.page_content[:200]}..."
        
        # Send the answer as a simple message
        await cl.Message(content=answer).send()
        
    except Exception as e:
        await cl.Message(
            content=f"Error processing your question: {str(e)}"
        ).send()
if __name__ == "__main__":
    cl.run_sync(start)