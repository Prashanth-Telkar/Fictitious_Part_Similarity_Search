import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained Sentence-BERT model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the pre-trained GPT-2 model for general chatbot conversation
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
llm_model = GPT2LMHeadModel.from_pretrained("gpt2")

# FAISS index for storing part descriptions and their embeddings
def create_faiss_index(descriptions):
    """
    Create a FAISS index to perform similarity search on part descriptions.

    Args:
        descriptions (list): List of part descriptions as strings.

    Returns:
        tuple: FAISS index and embeddings for the descriptions.
    """
    try:
        # Convert descriptions to embeddings
        embeddings = embedding_model.encode(descriptions, show_progress_bar=True)

        # Create FAISS index
        dim = embeddings.shape[1]  # The dimension of the embeddings
        index = faiss.IndexFlatL2(dim)  # L2 distance metric
        index.add(np.array(embeddings).astype(np.float32))
        
        return index, embeddings
    
    except Exception as e:
        st.error(f"Error while creating FAISS index: {e}")
        return None, None

# Load parts data from CSV file
def load_parts_data():
    """
    Load part descriptions data from a CSV file and clean the data.

    Returns:
        tuple: Dataframe with part information and list of descriptions.
    """
    try:
        # Read CSV file into pandas DataFrame
        df = pd.read_csv('Task_3/parts_data/Parts.csv', sep=';')
        
        # Drop rows with NaN values in 'DESCRIPTION' column
        df = df.dropna(subset=['DESCRIPTION'])
        
        # Convert the 'DESCRIPTION' column to a list of strings
        descriptions = df['DESCRIPTION'].tolist()
        
        return df, descriptions
    
    except Exception as e:
        st.error(f"Error while loading parts data: {e}")
        return None, None

# Setup Streamlit app
st.title("Fictitious Parts Similarity Search Chat-bot")
st.write("This application allows users to find similar parts based on their descriptions and also chat normally.")
st.write("This application is powered by GPT2 along with all-MiniLM-L6-v2 sentence transformer.")

# Load the data and create FAISS index
df, descriptions = load_parts_data()
if df is None or descriptions is None:
    st.stop()

index, embeddings = create_faiss_index(descriptions)
if index is None or embeddings is None:
    st.stop()

# Function to search for similar parts
def search_similar_parts(query, k=5):
    """
    Search for similar parts based on the user's query using FAISS.

    Args:
        query (str): The user's query (part description).
        k (int): The number of top similar parts to return.

    Returns:
        tuple: Indices of the top k similar parts and their distances.
    """
    try:
        # Convert query to embedding
        query_embedding = embedding_model.encode([query])
        
        # Perform the search using FAISS
        distances, indices = index.search(np.array(query_embedding).astype(np.float32), k)
        
        # Convert the distances to similarity scores in percentage (0-100%)
        similarity_scores = (1 - distances / np.max(distances)) * 100  # Normalize to percentage
        return indices, similarity_scores
    
    except Exception as e:
        st.error(f"Error while searching for similar parts: {e}")
        return None, None

# Function for generating general chatbot response using LLM (GPT-2)
def generate_chat_response(query):
    """
    Generate a chatbot response using GPT-2.

    Args:
        query (str): The user's query.

    Returns:
        str: The generated response from GPT-2.
    """
    try:
        inputs = tokenizer.encode(query, return_tensors="pt")
        outputs = llm_model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    except Exception as e:
        st.error(f"Error while generating chatbot response: {e}")
        return "Sorry, I couldn't understand that."

# Function to handle the chatbot response for similar parts
def get_part_similarity_response(query, indices, similarity_scores, df):
    """
    Generate a response displaying the top similar parts based on a user's query.

    Args:
        query (str): The user's query (part description).
        indices (ndarray): Indices of the top similar parts.
        similarity_scores (ndarray): Similarity scores of the top similar parts.
        df (DataFrame): Dataframe containing part data.

    Returns:
        str: The response showing similar parts with their details and similarity scores.
    """
    try:
        response = f"Here are the top {len(indices[0])} similar parts for your query:\n '{query}':\n"
        
        for i, idx in enumerate(indices[0]):
            part_info = df.iloc[idx]
            response += f"\nPart ID: {part_info['ID']}\n" 
            response += f"\nDescription: {part_info['DESCRIPTION']}\n"
            response += f"\nSimilarity Score: {similarity_scores[0][i]:.2f}%\n"
        
        return response
    
    except Exception as e:
        st.error(f"Error while generating similarity response: {e}")
        return "Sorry, I couldn't retrieve the similar parts."

# Function to detect intent using LLM
def detect_intent_and_respond(query):
    """
    Detect the intent of the user's query using GPT-2.

    Args:
        query (str): The user's query.

    Returns:
        str: The detected intent ('similar parts' or 'general question').
    """
    try:
        # Ask LLM to detect if the query is about part similarity or general conversation
        prompt = (
            f"User: {query}\n\n"
            "Assistant: Please classify the user's query into one of the following categories:\n"
            "1. 'general question' - The user is engaging in normal conversation, unrelated to finding parts.\n"
            "2. 'similar parts search' - The user is asking to find similar parts based on a part description.\n\n"
            "Your response should be one of these two categories: 'similar parts search' or 'general question'."
        )
         
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = llm_model.generate(inputs, max_length=300, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check if the response is related to finding similar parts or general conversation
        if "similar parts" in response.lower():
            return 'similar parts'
        else:
            return 'general question'
    
    except Exception as e:
        st.error(f"Error while detecting intent: {e}")
        return 'general question'

# Streamlit input
user_input = st.text_input("Enter part description or query:")

if user_input:
    # Detect the intent of the user query using LLM
    intent = detect_intent_and_respond(user_input)
    
    if intent == 'similar parts':
        # Find similar parts using the FAISS index
        indices, similarity_scores = search_similar_parts(user_input)
        
        if indices is not None and similarity_scores is not None:
            # Get similarity response
            response = get_part_similarity_response(user_input, indices, similarity_scores, df)
    
    else:
        # General conversation with LLM (GPT-2)
        response = generate_chat_response(user_input)
    
    # Display response in Streamlit app
    st.write(response)

else:
    st.warning("Please enter a query to find similar parts or chat.")
