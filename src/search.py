import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_search(query_vec, doc_matrix, df_chunks, top_k=5, min_similarity=0.09):
    similarities = cosine_similarity(query_vec, doc_matrix).flatten()

    valid_indices = np.where(similarities >= min_similarity)[0]

    sorted_indices = valid_indices[
        np.argsort(similarities[valid_indices])[::-1]
    ]

    top_indices = sorted_indices[:top_k]

    results = df_chunks.iloc[top_indices].copy()
    results["similarity_score"] = similarities[top_indices]

    return results[
        ["match_id", "home", "away", "chunk_text", "similarity_score"]
    ]

def embedding_search(query_vector, chunk_vectors, df_chunks, top_k=5):
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = df_chunks.iloc[top_indices].copy()
    results["similarity_score"] = similarities[top_indices]

    return results[
        ["match_id", "home", "away", "chunk_text", "similarity_score"]
    ]