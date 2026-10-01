import chromadb
from . import config

client = chromadb.PersistentClient(path=config.CHROMA_DIR)

collection = client.get_or_create_collection(
    name="documind",
    metadata={"hnsw:space": "cosine"}
)

def get_collection():
    return collection

def list_stored_documents():
    """Returns a list of dicts: [{'doc_id': ..., 'filename': ..., 'chunks': ...}]"""
    col = get_collection()
    res = col.get(include=["metadatas"])
    metadatas = res.get("metadatas", [])
    
    docs_map = {}
    if metadatas:
        for meta in metadatas:
            if not meta or "doc_id" not in meta:
                continue
            did = meta["doc_id"]
            fname = meta.get("filename", "Unknown PDF")
            if did not in docs_map:
                docs_map[did] = {"doc_id": did, "filename": fname, "chunks": 0}
            docs_map[did]["chunks"] += 1

    return list(docs_map.values())

def delete_stored_document(doc_id: str):
    """Deletes all chunks belonging to doc_id."""
    col = get_collection()
    col.delete(where={"doc_id": doc_id})

def clear_all_documents():
    """Deletes all documents from collection."""
    global collection
    client.delete_collection("documind")
    collection = client.get_or_create_collection(
        name="documind",
        metadata={"hnsw:space": "cosine"}
    )

