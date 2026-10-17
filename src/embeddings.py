from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(model, chunks):
    return model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def embed_query(model, query):
    return model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )