## Fictitious Parts Similarity Search

## Problem Statement

The objective of this project is to find 5 alternative parts to each provided fictitious part in the dataset based on their similarity. This involves finding similar parts using the following methods:

1. **Simple Method: Cosine Similarity with TF-IDF**
   - The simplest method for finding similar parts is by using **TF-IDF (Term Frequency - Inverse Document Frequency)**. The cosine similarity between parts is calculated based on their TF-IDF representations. This method is effective for capturing keyword-based similarities between parts.
   
2. **Intermediate Method: Word Embeddings (Word2Vec)**
   - Instead of relying on TF-IDF, the **Word2Vec** model is used. Word2Vec represents words in a continuous vector space, where semantically similar words (and therefore parts) are closer together. This method enables a better understanding of the semantic relationship between parts.

3. **Advanced Method: Transformer-based Embeddings (e.g., BERT, Sentence-BERT)**
   - The **Sentence-BERT** model **all-MiniLM-L6-v2** is used in this method. It represents parts as dense vector embeddings in a high-dimensional space using a pre-trained transformer model. This advanced method takes into account the context of each part description and provides more accurate results when comparing parts.

After comparing the results of each method, the best performing similarity method is selected.

Additionally, a Streamlit-powered **chatbot** is built that allows users to find similar parts based on their descriptions and engage in normal conversation. It utilizes a combination of the **Sentence-BERT model (all-MiniLM-L6-v2)** for part description embeddings and **GPT-2** for natural language conversation. Additionally, it uses the **FAISS** library for fast similarity searches in large datasets.

## Project Structure
```
.
├── parts_data/
│   └── Parts.csv                        # Input data containing part descriptions
├── result_similar_parts/                # Folder to store the results (similar parts)
├── chat_bot/                            # Folder containing chatbot-related code
│   └── part_similarity_chatbot.py       # Chatbot script for part similarity queries
├── src/                                 
│   ├── preprocessing.py                 # Data preprocessing steps
│   ├── similar_parts_tfidf.py           # Computes similarity using TF-IDF
│   ├── similar_parts_word2vec.py        # Computes similarity using Word2Vec
│   ├── similar_parts_sentence_bert.py   # Computes similarity using Sentence-BERT
├── main.py                              # Main script to process and compute similar parts
├── docs/                                # Documentation folder
└── README.md                            # Project documentation
```

### Folder Descriptions

- **parts_data/**: Contains the `Parts.csv` file, which holds the part descriptions to be used for similarity calculation.
- **result_similar_parts/**: Stores the results of similar parts calculated using different methods (TF-IDF, Word2Vec, Sentence-BERT).
- **chat_bot/**: Contains the script `part_similarity_chatbot.py`, which enables a conversational interface for querying similar parts and general inquiries.
- **src/**: Contains the source code for different methods of similarity calculation:
  - `preprocessing.py`: Preprocesses the part data.
  - `similar_parts_tfidf.py`: Implements similarity computation using TF-IDF and cosine similarity.
  - `similar_parts_word2vec.py`: Implements similarity computation using Word2Vec.
  - `similar_parts_sentence_bert.py`: Implements similarity computation using Sentence-BERT.
- **main.py**: The main script that integrates different methods for similarity calculation and saves results in the `result_similar_parts/` folder.
- **docs/**: Documentation folder to store additional resources and information related to the project.

## Methods for Finding Similar Parts

### 1. Cosine Similarity with TF-IDF
- **TF-IDF** is calculated for the "DESCRIPTION" column in the dataset. The cosine similarity is then computed between part descriptions to find the 5 most similar parts.

### 2. Word Embeddings (Word2Vec)
- **Word2Vec** embeddings are generated for each part description. The cosine similarity between parts is computed in the vector space to determine similarity based on semantic meaning.

### 3. Transformer-based Embeddings (Sentence-BERT)
- **Sentence-BERT** embeddings are generated for each part description using the pre-trained **all-MiniLM-L6-v2** model. The similarity between parts is calculated using cosine similarity in the embedding space, providing more context-aware and accurate similarity results.

## Chatbot Integration

A **chatbot** is developed to interact with users. The chatbot has two main functionalities:
1. **Find Similar Parts**: Users can input a description of a part, and the chatbot will return the 5 most similar parts based on the selected similarity method.
2. **General Conversation**: The chatbot can also hold general conversations with the user using a transformer-based model (GPT-2) for natural language processing.

# Features

The chatbot uses the part descriptions and their computed similarities to provide relevant and useful responses. It has following features.
- FAISS Integration: Efficient similarity search using FAISS index to retrieve top similar parts.
- Intent Detection: The chatbot intelligently distinguishes between similarity searches and general conversation using prompt 
  engineering.

## How to Run the Project

### Step 1: Clone the repository
Clone this repository in your local machine. 
```
git clone <repository_url>
```

### Step 2: Install Dependencies
Make sure you have Python 3.x installed, and then install the required dependencies using pip:
```
pip install -r requirements.txt
```

### Step 3: Prepare the Dataset
Place your Parts.csv file in the parts_data/ folder.
The file should contain at least DESCRIPTION and ID column and some other columns like RATED_VOLTAGE, RATED_CURRENT, etc.

### Step 4: Running the Main Script
Run the main script to calculate similar parts using your preferred method (TF-IDF, Word2Vec, or Sentence-BERT):
```
python main.py
```
You can see the results in the results_similar_parts folder. This contains the list of similar parts for each part in the given Parts.csv

### Step 5: Run the App : Fictitious Parts Similarity Search Chat-bot
Change the directory to /chat_bot folder.
Run the Streamlit app with the following command:
```
streamlit run part_similarity_chatbot.py
```

This will start the Streamlit application, which you can access in your browser at http://localhost:8501.
