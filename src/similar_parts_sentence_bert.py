import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd

def find_similar_parts_sentence_bert(df):
    """
    Finds the top 5 similar parts based on Sentence-BERT embeddings and cosine similarity.

    This function takes a DataFrame with part descriptions, computes embeddings using Sentence-BERT,
    calculates cosine similarities, and finds the top 5 most similar parts for each part description.

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

        # Load the Sentence-BERT model (you can use other models as well)
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Step 1: Convert each part description to its Sentence-BERT embedding
        embeddings = model.encode(descriptions)

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
        raise  # Re-raise the exception after logging
    except Exception as e:
        print(f"An error occurred: {e}")
        raise  # Re-raise the exception after logging

