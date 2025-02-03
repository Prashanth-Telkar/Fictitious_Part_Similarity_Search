import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import pandas as pd

# Load pre-trained Word2Vec embeddings (Google News Word2Vec model)
model = api.load("word2vec-google-news-300")

def get_embedding(text, model):
    """
    Given a text description, this function returns the Word2Vec embedding by averaging the word embeddings
    of the words in the description.

    Args:
        text (str): The part description text.
        model: Pre-trained Word2Vec model.

    Returns:
        np.ndarray: The averaged embedding vector for the description.
    """
    words = text.split()
    embeddings = []
    for word in words:
        if word in model:
            embeddings.append(model[word])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

def find_similar_parts_word2vec(df):
    """
    Find the top 5 similar parts based on Word2Vec embeddings and cosine similarity.

    This function takes a DataFrame with part descriptions, converts each description to its Word2Vec embedding,
    computes cosine similarities, and finds the top 5 most similar parts for each part description.

    Args:
        df (pd.DataFrame): A DataFrame containing 'ID' and 'DESCRIPTION' columns, with part descriptions.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - 'Part ID': The ID of the part being compared.
            - 'Part Description': The description of the part being compared.
            - 'Similar Part ID': The ID of a similar part.
            - 'Similar Part Description': The description of the similar part.
            - 'Similarity Percentage': The similarity percentage between the parts (0-100%).

    Raises:
        ValueError: If the input DataFrame does not contain 'ID' and 'DESCRIPTION' columns.
        Exception: For any unexpected errors during the execution of the function.
    """
    try:
        # Ensure the DataFrame contains the required columns
        if not all(col in df.columns for col in ['ID', 'DESCRIPTION']):
            raise ValueError("DataFrame must contain 'ID' and 'DESCRIPTION' columns")

        # Extract descriptions and IDs
        descriptions = df['DESCRIPTION'].tolist()
        ids = df['ID'].tolist()

        # Step 1: Convert each part description to its Word2Vec embedding
        embeddings = [get_embedding(part, model) for part in descriptions]

        # Step 2: Compute cosine similarity matrix
        cos_sim = cosine_similarity(embeddings)

        # Step 3: Find the top 5 similar parts for each part
        top_5_similar_parts = []
        for idx, part in enumerate(descriptions):
            # Get the most similar parts (including self for sorting, but we exclude self from the final list)
            similar_parts_indices = np.argsort(cos_sim[idx])[::-1][1:6]  # Sorting in descending order and excluding the self comparison
            
            for i in similar_parts_indices:
                similar_id = ids[i]
                similar_description = descriptions[i]
                similarity_score = cos_sim[idx][i] * 100  # Converting to percentage
                similarity_percentage = round(similarity_score, 2)  # Round to 2 decimal places
                
                top_5_similar_parts.append({
                    'Part ID': ids[idx],
                    'Part Description': descriptions[idx],
                    'Similar Part ID': similar_id,
                    'Similar Part Description': similar_description,
                    'Similarity Percentage': similarity_percentage
                })

        # Convert the result into a DataFrame for better visualization
        result_df = pd.DataFrame(top_5_similar_parts)
        return result_df
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise 
    except Exception as e:
        print(f"An error occurred: {e}")
        raise 
