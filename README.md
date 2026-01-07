# Semantic Search Engine – EPL 2023–24

This project explores semantic search over full-match commentary from the English Premier League 2023–24 season using NLP techniques.

The goal is to retrieve relevant matches based on natural language queries, focusing on meaning rather than exact keyword matching.

## Problem Statement

Match commentaries contain rich contextual information about player performances, momentum shifts, and decisive moments.  
However, searching these commentaries using keyword-based methods fails to capture semantic intent.

This project aims to answer questions like:
- *“Late goals by Haaland”*
- *“Matches where Arsenal struggled defensively”*
- *“High-pressure performances by Manchester City”*

even when the exact wording does not appear in the text.


## Dataset

- English Premier League 2023–24 season
- Each row contains **full-match commentary text**
- Additional metadata includes teams and match-level information

Each match commentary is split into **overlapping text chunks** to improve retrieval granularity.
Link - [English Premier League 23/24](https://www.kaggle.com/datasets/pranavkarnani/english-premier-league-match-commentary)


## Approach (v1)

### 1. Data Preprocessing
- Removed irrelevant columns
- Normalized text (lowercasing, cleaning)
- Inferred home and away teams from commentary, since the original dataset had inconsistent home and away columns.
- Assigned stable match IDs

### 2. Text Chunking
- Match commentary is split into overlapping word-based chunks
- This preserves context while improving semantic retrieval

### 3. Vectorization
- TF-IDF vectorization using `scikit-learn`
- Stopword removal and n-gram support
- Each text chunk is converted into a sparse vector representation

### 4. Semantic Search
- User query is vectorized using the same TF-IDF space
- Cosine similarity is computed between the query and all chunks
- Results are ranked and filtered using a similarity threshold


## Project Structure

semantic-search-epl/
│
├── data/
│ └── 23_24_match_details.csv
│
├── notebooks/
│ └── semantic.ipynb
│
├── src/
│ ├── load_data.py
│ ├── preprocess.py
│ ├── chunking.py
│ ├── vectorizer.py
│ ├── search.py
│ └── main.py
│
├── requirements.txt
└── README.md

## How to Run

From the project Root:
```bash
$ python src/main.py
```
You can then enter natural language queries in the terminal
Type "exit" to quit

### Example Query
```csharp
Enter search query:
late goals by haaland
```
### Example output
```yaml
Match ID: 128 | Manchester City vs Arsenal | Score: 0.214
Haaland finishes a late move after sustained pressure...
```

## Version 1 Scope
### Included
- Tf-Idf based semantic search
- Cosine similarity ranking
- Text chunking 
- CLI- based querying

## Tech Stack
- Python
- Jupyter Notebook
- Pandas, Numpy
- Sci-kit learn (TF-IDF, Cosine Similarity)

## Author
Lavanya Dharmadhikari