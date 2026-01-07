from load_data import load_csv_path
from preprocess import select_relevant_columns, correcting_home_away_teams, build_chunk_dataframe
from chunking import chunk_text
from vectorizer import build_vectorizer, vectorize_query
from search import semantic_search

csv_path = "data/23_24_match_details.csv"

teams = ["Manchester City", "Liverpool", "Burnley", "Brighton", "Wolves", "Tottenham", "Arsenal", "Fulham", "Aston Villa", "Newcastle United",
         "Manchester United", "Nottingham Forest", "Brentford", "Chelsea", "Crystal Palace", "Luton Town", "Bournemouth", "Everton", "Sheffield United", "West Ham"]

def run():
    
    df = load_csv_path(csv_path)
    
    df = select_relevant_columns(df)
    df = correcting_home_away_teams(df, teams)
    
    df_chunks = build_chunk_dataframe(df, chunk_text)
    
    vectorizer, doc_matrix = build_vectorizer(df_chunks["chunk_text"])
    
    while True:
        query = input("\nEnter search query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        
        query_vec = vectorize_query(vectorizer, query)

        results = semantic_search(
            query_vec=query_vec,
            doc_matrix=doc_matrix,
            df_chunks=df_chunks,
            top_k=5,
            min_similarity=0.09
        )
        
        if results.empty:
            print("No relevant results found.")
            continue

        for _, row in results.iterrows():
            print(
                f"\nMatch ID: {row['match_id']} | "
                f"{row['home']} vs {row['away']} | "
                f"Score: {row['similarity_score']:.3f}"
            )
            print(row["chunk_text"][:300] + "...")
            print("_"*80)
            
            
if __name__ == "__main__":
    run()
    