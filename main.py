import pandas as pd
from src.preprocessing import preprocess_parts_data
from src.similar_parts_tfidf import find_similar_parts_tfidf
from src.similar_parts_sentence_bert import find_similar_parts_sentence_bert
from src.similar_parts_word2vec import find_similar_parts_word2vec  # Import the Word2Vec function

def main(path_parts_data, path_similar_parts, method="tfidf"):
    """
    Main function to process the parts data and save the similar parts based on the selected method 
    (TF-IDF, Word2Vec similarity or Sentence-BERT).

    Args:
        path_parts_data (str): The path to the CSV file containing parts data, including 'ID' and 'DESCRIPTION' columns.
        path_similar_parts (str): The directory path where the similar parts CSV file should be saved.
        method (str): The method to use for similarity calculation. Can be either "tfidf", "sentence-bert", or "word2vec".

    Returns:
        None

    Raises:
        FileNotFoundError: If any of the provided file paths do not exist.
        ValueError: If the CSV does not contain the expected 'ID' and 'DESCRIPTION' columns.
        Exception: For any unexpected errors during processing or file saving.

    Example:
        >>> main("path/to/parts_data.csv", "path/to/save/", method="sentence-bert")
    """
    try:
        # Step 1: Preprocess parts data
        parts_data = preprocess_parts_data(file_path=path_parts_data)

        # Step 2: Choose the similarity method based on the user's choice
        if method == "tfidf":
            similar_parts = find_similar_parts_tfidf(parts_data)
        elif method == "sentence-bert":
            similar_parts = find_similar_parts_sentence_bert(parts_data)
        elif method == "word2vec":
            similar_parts = find_similar_parts_word2vec(parts_data)  # Call the Word2Vec function
        else:
            raise ValueError("Invalid method specified. Please choose either 'tfidf', 'sentence-bert', or 'word2vec'.")

        # Step 3: Save the results to CSV
        similar_parts.to_csv(path_similar_parts + f'similar_parts_{method}.csv', index=False)
        print(f"Similar parts have been saved to {path_similar_parts}similar_parts_{method}.csv")

    except FileNotFoundError as fnf_error:
        print(f"FileNotFoundError: {fnf_error}")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":

    # Define file paths
    path_parts_data = 'parts_data/Parts.csv'
    path_similar_parts = 'result_similar_parts/'

    # Choose similarity method: 'tfidf', 'word2vec', or 'sentence-bert'
    similarity_method = "word2vec"  # Change to 'tfidf' or 'sentence-bert' as needed

    # Call the main function with the chosen method
    main(path_parts_data, path_similar_parts, method=similarity_method)
