import json
import os
from typing import Dict, List

from openai import OpenAI

from src.config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    INDEX_DIR,
    TOP_K,
    DENSE_CANDIDATES,
    SPARSE_CANDIDATES,
    HYBRID_ALPHA,
    SOURCE_CONFIG,
)
from src.embeddings import get_embedding
from src.hybrid_store import HybridStore
from src.router import SemanticRouter

client = OpenAI(api_key=OPENAI_API_KEY)


FEW_SHOT_INSTRUCTIONS = """
You are an industrial documentation assistant for plant floor supervisors.

Rules:
1. Answer only from the retrieved context.
2. If the context does not support the answer, explicitly say you could not find a grounded answer.
3. Never invent plant instructions, measurements, or procedural steps.
4. Keep the answer concise and operational.
5. Always cite supporting chunk numbers.
6. Output valid JSON only.

Example 1
Question: What PPE is required before handling a chemical leak?
Context:
[Chunk 1] Required PPE includes safety glasses, steel-toe shoes, and chemical-resistant gloves when handling chemicals.
Output:
{"grounded":"yes","answer":"Use safety glasses, steel-toe shoes, and chemical-resistant gloves before handling the chemical leak. [Chunk 1]","citations":[1],"abstain_reason":""}

Example 2
Question: What torque value should be used for the motor housing bolts?
Context:
[Chunk 1] The manual describes inspection intervals and vibration checks.
Output:
{"grounded":"no","answer":"I could not find a fully grounded answer in the routed documentation.","citations":[],"abstain_reason":"No torque value is present in the retrieved context."}

Example 3
Question: What should happen if two consecutive samples fail dimensional inspection?
Context:
[Chunk 1] If two consecutive samples fail dimensional inspection, stop the line and investigate.
Output:
{"grounded":"yes","answer":"Stop the line and investigate the issue. [Chunk 1]","citations":[1],"abstain_reason":""}
""".strip()


class PlantDocAssistant:
    def __init__(self):
        self.router = SemanticRouter()
        self.stores = {}

        for source_key in SOURCE_CONFIG.keys():
            store_path = os.path.join(INDEX_DIR, source_key)
            if not os.path.exists(store_path):
                raise FileNotFoundError(
                    f"Index missing for source '{source_key}'. Run: python ingest.py"
                )
            self.stores[source_key] = HybridStore.load(store_path)

    def retrieve(self, question: str, source_key: str) -> List[Dict]:
        q_emb = get_embedding(question)
        return self.stores[source_key].search(
            query_text=question,
            query_embedding=q_emb,
            top_k=TOP_K,
            dense_candidates=DENSE_CANDIDATES,
            sparse_candidates=SPARSE_CANDIDATES,
            alpha=HYBRID_ALPHA,
        )

    def build_context(self, retrieved_chunks: List[Dict]) -> str:
        parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            parts.append(
                f"[Chunk {i}]\n"
                f"Heading: {chunk.get('heading', 'General')}\n"
                f"Source File: {chunk['file_name']}\n"
                f"Text: {chunk['chunk_text']}"
            )
        return "\n\n".join(parts)

    def _parse_json_response(self, raw_text: str) -> Dict:
        try:
            return json.loads(raw_text)
        except Exception:
            return {
                "grounded": "no",
                "answer": "I could not find a fully grounded answer in the routed documentation.",
                "citations": [],
                "abstain_reason": "Model output was not valid JSON."
            }

    def answer(self, question: str) -> Dict:
        routed_source, route_scores = self.router.route(question)
        retrieved_chunks = self.retrieve(question, routed_source)
        context = self.build_context(retrieved_chunks)
        source_label = SOURCE_CONFIG[routed_source]["label"]

        system_prompt = f"""
{FEW_SHOT_INSTRUCTIONS}

The routed source for this question is: {source_label}.
If the answer is unsupported, abstain.
""".strip()

        user_prompt = f"""
Question:
{question}

Retrieved Context:
{context}
""".strip()

        response = client.responses.create(
            model=OPENAI_CHAT_MODEL,
            instructions=system_prompt,
            input=user_prompt,
            temperature=0
        )

        parsed = self._parse_json_response(response.output_text)

        return {
            "question": question,
            "routed_source": routed_source,
            "source_label": source_label,
            "route_scores": route_scores,
            "retrieved_chunks": retrieved_chunks,
            "answer": parsed.get("answer", ""),
            "grounded": parsed.get("grounded", "no"),
            "citations": parsed.get("citations", []),
            "abstain_reason": parsed.get("abstain_reason", "")
        }