from load_data import load_csv_path
from preprocess import select_relevant_columns, correcting_home_away_teams, build_chunk_dataframe
from chunking import chunk_text
from vectorizer import build_vectorizer, vectorize_query
from search import semantic_search
import streamlit as st

csv_path = "data/23_24_match_details.csv"

teams = ["Manchester City", "Liverpool", "Burnley", "Brighton", "Wolves", "Tottenham", "Arsenal", "Fulham", "Aston Villa", "Newcastle United",
         "Manchester United", "Nottingham Forest", "Brentford", "Chelsea", "Crystal Palace", "Luton Town", "Bournemouth", "Everton", "Sheffield United", "West Ham"]


@st.cache_data
def load_and_prepare_chunks(path, team_names):
    df = load_csv_path(path)
    df = select_relevant_columns(df)
    df = correcting_home_away_teams(df, team_names)
    return build_chunk_dataframe(df, chunk_text)


@st.cache_resource
def build_search_index(chunks_df):
    return build_vectorizer(chunks_df["chunk_text"])

st.title("Semantic search engine")
st.header("Searching user queries across the 23/24 PL commentary dataset")
st.write("Returning the top 5 most similar results")

try:
    df_chunks = load_and_prepare_chunks(csv_path, teams)
    vectorizer, doc_matrix = build_search_index(df_chunks)

    query = st.text_input("Enter search query", placeholder="Type here...").strip()

    if not query:
        st.info("Enter a query to see the top 5 similar results.")
    else:
        query_vec = vectorize_query(vectorizer, query)

        results = semantic_search(
            query_vec=query_vec,
            doc_matrix=doc_matrix,
            df_chunks=df_chunks,
            top_k=5,
            min_similarity=0.09,
        )

        if results.empty:
            st.write("No relevant results found.")
        else:
            for _, row in results.iterrows():
                label = (
                    f"Match {row['match_id']}: {row['home']} vs {row['away']} "
                    f"(Score: {row['similarity_score']:.3f})"
                )
                with st.expander(label):
                    st.write(row["chunk_text"][:300] + "...")
                    st.caption(f"Full Match ID: {row['match_id']}")

            
            
except Exception as e:
    st.error(f"Error: {e}")
            