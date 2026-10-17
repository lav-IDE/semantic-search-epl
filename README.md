# EPL Match Commentary Search

This project contains a Streamlit app for searching the 2023–24 English Premier League match commentary dataset. It currently prepares commentary chunks and ranks them with two retrieval methods: TF-IDF with cosine similarity, and sentence embeddings with cosine similarity.

## Data

The app reads `data/23_24_match_details.csv`. The CSV is included in this repository. Rows without an `events` value are removed. The app lowercases the commentary, drops unused columns when present, and assigns each remaining source row an ID based on its original CSV row index. It guesses the home and away teams by looking for team names in the first 40 words of the commentary; if fewer than two names are found, both teams are marked `unknown`.

Commentary is split into overlapping chunks of 50 words with a 10-word overlap. Each chunk keeps its match ID and inferred team names.

## Search methods

- **TF-IDF:** fits a scikit-learn `TfidfVectorizer` over the chunks, removes English stop words, and ranks chunks by cosine similarity. The app shows up to five results and filters out scores below `0.09`.
- **Embeddings:** encodes chunks and queries with `sentence-transformers/all-MiniLM-L6-v2`, using normalized embeddings and cosine similarity. The app shows up to ten results.

The Streamlit page displays the two result sets in separate tabs. Each result shows the match ID, inferred teams, similarity score, and the first 300 characters of the chunk.

## Run the app

Use Python with the packages in `requirements.txt` and install the embedding package as well, since it is imported by the app but is not currently listed in that file:

```bash
pip install -r requirements.txt
pip install sentence-transformers
streamlit run src/main.py
```

Run the command from the project root. On first use, the Sentence Transformers model may need to be downloaded. Enter a query in the app to see results.

## Project files

```text
data/23_24_match_details.csv  Dataset used by the app
notebooks/semantic.ipynb      Notebook with the TF-IDF exploration
src/load_data.py              CSV loading
src/preprocess.py             Column selection, team inference, chunk records
src/chunking.py               Word-based chunking
src/vectorizer.py             TF-IDF vectorizer and query transform
src/search.py                 TF-IDF and embedding ranking
src/embeddings.py             Sentence Transformers model and encoding
src/main.py                   Streamlit app
requirements.txt              Declared Python dependencies
```

## Current limitations

- Home and away teams are inferred from commentary text and can be wrong or unknown.
- The app builds the TF-IDF matrix and embedding vectors at runtime; there is no persisted search index.
- Embedding search requires `sentence-transformers`, which is not yet declared in `requirements.txt`.
- Search is limited to the included 2023–24 commentary CSV.

## Author

Lavanya Dharmadhikari
