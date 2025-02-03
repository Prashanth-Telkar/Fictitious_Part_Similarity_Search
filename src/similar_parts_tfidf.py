import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_parts_tfidf(df):
    """
    Find the top 5 similar parts based on the 'DESCRIPTION' column in the provided DataFrame.

    This function computes the cosine similarity between descriptions of parts and finds the top 5 most similar parts
    for each part in the dataset. It also includes the similarity percentage (between 0 and 100) for each similar part.

    Args:
        df (pd.DataFrame): A DataFrame containing two columns: 'ID' and 'DESCRIPTION'.
            - 'ID' (int or str): Unique identifier for each part.
            - 'DESCRIPTION' (str): Descriptive text for each part.

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
        
        # Step 1: Convert text data into TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        # Step 2: Compute cosine similarity matrix
        cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Step 3: For each part, find the top 5 most similar parts
        top_5_similar_parts = []
        
        for idx, part in enumerate(descriptions):
            # Get the most similar parts (including self for sorting, but we'll exclude self from final list)
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
