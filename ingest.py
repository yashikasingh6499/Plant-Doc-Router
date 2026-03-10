import os
from tqdm import tqdm

from src.chunking import semantic_chunk_text
from src.config import SOURCE_CONFIG, INDEX_DIR, MAX_CHARS_PER_CHUNK, CHUNK_OVERLAP
from src.embeddings import get_embeddings
from src.hybrid_store import HybridStore
from src.loaders import load_source_document
from src.utils import ensure_dir


def build_index_for_source(source_key: str, source_cfg: dict):
    print(f"\nBuilding index for: {source_cfg['label']}")
    docs = load_source_document(source_cfg["file_path"])

    if not docs:
        print(f"No document found for {source_cfg['file_path']}")
        return

    all_texts = []
    all_metadata = []

    for doc in docs:
        chunks = semantic_chunk_text(
            doc["text"],
            max_chars=MAX_CHARS_PER_CHUNK,
            overlap=CHUNK_OVERLAP
        )

        for idx, chunk in enumerate(chunks):
            all_texts.append(chunk["chunk_text"])
            all_metadata.append({
                "source_key": source_key,
                "source_label": source_cfg["label"],
                "file_name": doc["file_name"],
                "file_path": doc["file_path"],
                "chunk_id": idx,
                "heading": chunk["heading"],
                "chunk_text": chunk["chunk_text"]
            })

    print(f"Total chunks: {len(all_texts)}")

    embeddings = []
    batch_size = 64
    for i in tqdm(range(0, len(all_texts), batch_size)):
        batch = all_texts[i:i + batch_size]
        embeddings.extend(get_embeddings(batch))

    if not embeddings:
        print("No embeddings created.")
        return

    dim = len(embeddings[0])
    store = HybridStore(dim=dim)
    store.add(embeddings, all_metadata)

    save_path = os.path.join(INDEX_DIR, source_key)
    ensure_dir(save_path)
    store.save(save_path)

    print(f"Saved index to: {save_path}")


def main():
    ensure_dir(INDEX_DIR)
    for source_key, source_cfg in SOURCE_CONFIG.items():
        build_index_for_source(source_key, source_cfg)
    print("\nAll indexes built successfully.")


if __name__ == "__main__":
    main()