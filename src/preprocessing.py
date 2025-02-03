import pandas as pd

def preprocess_parts_data(file_path):
    """
    Preprocess the parts data from the given CSV file.

    This function reads the CSV file, drops rows where the 'DESCRIPTION' column has NaN values,
    and returns a DataFrame with only the 'ID' and 'DESCRIPTION' columns.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing only the 'ID' and 'DESCRIPTION' columns,
                      with rows containing NaN in 'DESCRIPTION' removed.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
        ValueError: If the CSV does not contain the expected columns 'ID' and 'DESCRIPTION'.
        Exception: For any unexpected errors during file reading or processing.
    """
    try:
        # Read the CSV file with the specified separator
        df = pd.read_csv(file_path, sep=';')

        # Check if 'ID' and 'DESCRIPTION' columns are present
        if 'ID' not in df.columns or 'DESCRIPTION' not in df.columns:
            raise ValueError("CSV file must contain 'ID' and 'DESCRIPTION' columns")

        # Drop rows where 'DESCRIPTION' is NaN
        df_cleaned = df.dropna(subset=['DESCRIPTION'])

        # Return only the 'ID' and 'DESCRIPTION' columns
        df_cleaned = df_cleaned[['ID', 'DESCRIPTION']]

        return df_cleaned

    except FileNotFoundError as fnf_error:
        print(f"FileNotFoundError: {fnf_error}")
        raise
    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
