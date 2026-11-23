"""Semantic card search: embed every card in the snapshot into ChromaDB and query it.

Adapted from the class semantic-search activity (sentence-transformers + ChromaDB). Each
result carries the attributes the price model needs, so a front end can go straight from
"charizard ex 151" to a fair value.

Usage:
    python -m pipeline.search build                         # embed the configured snapshot
    python -m pipeline.search query "fiery flying dragon"   # try a search
"""

import argparse
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import load_snapshot
from .features import RAW_INPUT_COLUMNS

# Card fields copied into each search result (and Chroma metadata).
CARD_FIELDS = [c for c in RAW_INPUT_COLUMNS if c != "printing"] + [
    "product_id", "set_name", "image_url", "url",
]


def card_documents(df: pd.DataFrame) -> tuple[list[str], list[str], list[dict]]:
    """One document per product. Printings (normal/holo/...) are kept as a price map."""
    printings = (
        df.groupby("product_id")[["printing", "market_price"]]
        .apply(lambda g: json.dumps(dict(zip(g["printing"], g["market_price"].round(2)))))
        .rename("printings")
    )
    products = df.drop_duplicates("product_id").set_index("product_id").join(printings).reset_index()

    ids, docs, metas = [], [], []
    for row in products.itertuples(index=False):
        r = row._asdict()
        year = str(r["set_published"])[:4] if pd.notna(r["set_published"]) else ""
        kind = " ".join(str(r[k]) for k in ("rarity", "stage", "card_type") if pd.notna(r[k]))
        printing_names = ", ".join(json.loads(r["printings"]))
        parts = [f"{r['name']}.", f"{r['set_name']} ({year}).", f"{kind}.", f"Printings: {printing_names}."]
        if pd.notna(r["hp"]):
            parts.append(f"{int(r['hp'])} HP.")
        if pd.notna(r["attacks"]):
            parts.append(f"Attacks: {r['attacks']}.")
        if pd.notna(r["card_text"]):
            parts.append(str(r["card_text"]))

        meta = {"printings": r["printings"]}
        for field in CARD_FIELDS:
            value = r[field]
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                continue  # Chroma metadata can't hold nulls
            meta[field] = value.item() if hasattr(value, "item") else value
        ids.append(str(r["product_id"]))
        docs.append(" ".join(parts))
        metas.append(meta)
    return ids, docs, metas


class SentenceEmbedder:
    """Thin wrapper so tests can swap in a fake embedder without downloading a model."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 1000
        )


def _client(index_dir: str | Path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(index_dir), settings=Settings(anonymized_telemetry=False))


def build_index(cfg: dict, df: pd.DataFrame, embedder=None) -> int:
    """(Re)build the Chroma collection from a snapshot. Returns the number of cards indexed."""
    s = cfg["search"]
    embedder = embedder or SentenceEmbedder(s["embedding_model"])
    ids, docs, metas = card_documents(df)

    client = _client(s["index_dir"])
    # Rebuild from scratch so the index matches the snapshot exactly.
    if s["collection"] in [c.name for c in client.list_collections()]:
        client.delete_collection(s["collection"])
    collection = client.create_collection(
        s["collection"],
        metadata={
            "hnsw:space": "cosine",
            "snapshot_date": cfg["data"]["snapshot_date"],
            "embedding_model": s["embedding_model"],
        },
    )

    print(f"Embedding {len(docs):,} cards with {s['embedding_model']} ...")
    embeddings = embedder.encode(docs, batch_size=s["batch_size"])
    step = 5000  # Chroma caps the batch size per add()
    for start in range(0, len(ids), step):
        end = start + step
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metas[start:end],
            embeddings=np.asarray(embeddings[start:end]).tolist(),
        )
    return len(ids)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keyword_text(meta: dict) -> str:
    """The identifying fields a person would type: name, set, number, rarity, printings."""
    fields = " ".join(str(meta.get(k, "")) for k in ("name", "set_name", "number", "rarity"))
    return fields + " " + " ".join(json.loads(meta["printings"]))


class CardSearch:
    """Query a built index. Results are dicts ready to pass to predict_fair_value().

    Hybrid ranking:
    - candidates = nearest cards by embedding (meaning: "fiery flying dragon")
      plus best keyword matches on name/set/number (exact: "base set charizard")
    - score = cosine similarity + keyword_weight x keyword match, where the keyword
      match weights each query word by how rare it is (IDF), so "charizard" counts
      far more than "set" or "edition".
    """

    def __init__(self, index_dir: str | Path, collection: str, embedder,
                 candidate_pool: int = 50, keyword_weight: float = 0.5):
        self.collection = _client(index_dir).get_collection(collection)
        self.embedder = embedder
        self.candidate_pool = candidate_pool
        self.keyword_weight = keyword_weight

        everything = self.collection.get(include=["metadatas"])
        self._ids = everything["ids"]
        self._token_sets = [_tokens(_keyword_text(m)) for m in everything["metadatas"]]
        n = len(self._ids)
        doc_freq = Counter(t for tokens in self._token_sets for t in tokens)
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in doc_freq.items()}
        self._unknown_idf = math.log(n + 1) + 1

    def _keyword_scores(self, query: str) -> np.ndarray:
        """Per card: IDF-weighted share of the query's words found in its keyword fields."""
        weights = {t: self._idf.get(t, self._unknown_idf) for t in _tokens(query)}
        total = sum(weights.values())
        if not total:
            return np.zeros(len(self._ids))
        return np.array(
            [sum(weights[t] for t in weights.keys() & tokens) for tokens in self._token_sets]
        ) / total

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_emb = np.asarray(self.embedder.encode([query]), dtype=np.float32)[0]
        pool = min(max(self.candidate_pool, top_k), len(self._ids))

        by_meaning = self.collection.query(query_embeddings=[query_emb.tolist()], n_results=pool, include=[])
        keyword = self._keyword_scores(query)
        by_keyword = [self._ids[i] for i in np.argsort(-keyword)[:pool] if keyword[i] > 0]
        candidates = list(dict.fromkeys(by_meaning["ids"][0] + by_keyword))

        found = self.collection.get(ids=candidates, include=["metadatas", "documents", "embeddings"])
        similarity = np.asarray(found["embeddings"], dtype=np.float32) @ query_emb  # unit vectors
        position = {pid: i for i, pid in enumerate(self._ids)}

        results = []
        for pid, meta, doc, sim in zip(found["ids"], found["metadatas"], found["documents"], similarity):
            card = {k: v for k, v in meta.items() if k != "printings"}
            card["printings"] = json.loads(meta["printings"])
            card["description"] = doc
            kw = float(keyword[position[pid]])
            card["similarity"] = round(float(sim), 4)
            card["keyword_match"] = round(kw, 4)
            card["score"] = round(float(sim) + self.keyword_weight * kw, 4)
            results.append(card)
        results.sort(key=lambda c: c["score"], reverse=True)
        return results[:top_k]


def model_input(card: dict, printing: str) -> dict:
    """Turn a search result plus the chosen printing into the price model's input fields."""
    return {**{c: card.get(c) for c in RAW_INPUT_COLUMNS}, "printing": printing}


@lru_cache(maxsize=2)
def load_search(config_path: str | None = None) -> CardSearch:
    """Load the configured index and embedding model once (for the API)."""
    s = load_config(config_path)["search"]
    return CardSearch(
        s["index_dir"], s["collection"], SentenceEmbedder(s["embedding_model"]),
        candidate_pool=s["candidate_pool"], keyword_weight=s["keyword_weight"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="Embed the configured snapshot into ChromaDB")
    q = sub.add_parser("query", help="Search the index")
    q.add_argument("text")
    q.add_argument("-k", "--top-k", type=int)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.command == "build":
        n = build_index(cfg, load_snapshot(cfg))
        print(f"Indexed {n:,} cards into {cfg['search']['index_dir']}")
        return n

    results = load_search(args.config).search(args.text, args.top_k or cfg["search"]["default_top_k"])
    for r in results:
        prices = ", ".join(f"{p} ${v:,.2f}" for p, v in r["printings"].items())
        print(f"{r['score']:.3f}  {r['name']}  [{r['set_name']}, {r.get('rarity', '?')}]  {prices}")
    return results


if __name__ == "__main__":
    main()
