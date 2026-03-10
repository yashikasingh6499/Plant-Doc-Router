import json
from statistics import mean

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_JUDGE_MODEL
from src.qa import PlantDocAssistant

client = OpenAI(api_key=OPENAI_API_KEY)


def load_eval_data(path: str = "./evals/eval_dataset.jsonl"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reciprocal_rank(expected_file: str, retrieved_chunks):
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk["file_name"] == expected_file:
            return 1.0 / rank
    return 0.0


def hit_at_k(expected_file: str, retrieved_chunks, k: int):
    top = retrieved_chunks[:k]
    return any(chunk["file_name"] == expected_file for chunk in top)


def llm_judge(question, expected_answer, actual_answer, retrieved_chunks):
    context = "\n\n".join(
        [
            f"[Chunk {i}] {chunk['file_name']} :: {chunk['chunk_text']}"
            for i, chunk in enumerate(retrieved_chunks, start=1)
        ]
    )

    instructions = """
You are an evaluator for a retrieval-augmented QA system.
Score the answer using the retrieved evidence only.

Return JSON with:
- correctness: 1 or 0
- groundedness: 1 or 0
- completeness: integer 1 to 5
- notes: short string

Rules:
- correctness = 1 only if the answer matches the expected answer meaningfully.
- groundedness = 1 only if the answer is supported by the retrieved chunks.
- completeness reflects whether the answer covers the needed content.
- Output valid JSON only.
""".strip()

    prompt = f"""
Question:
{question}

Expected Answer:
{expected_answer}

System Answer:
{actual_answer}

Retrieved Evidence:
{context}
""".strip()

    response = client.responses.create(
        model=OPENAI_JUDGE_MODEL,
        instructions=instructions,
        input=prompt,
        temperature=0
    )

    try:
        return json.loads(response.output_text)
    except Exception:
        return {
            "correctness": 0,
            "groundedness": 0,
            "completeness": 1,
            "notes": "Judge output parse failed."
        }


def main():
    assistant = PlantDocAssistant()
    dataset = load_eval_data()

    route_correct = []
    answer_correct = []
    grounded = []
    completeness_scores = []
    mrr_scores = []
    hit1_scores = []
    hit3_scores = []

    detailed_rows = []

    for item in dataset:
        result = assistant.answer(item["question"])

        rr = reciprocal_rank(item["expected_file"], result["retrieved_chunks"])
        hit1 = 1 if hit_at_k(item["expected_file"], result["retrieved_chunks"], 1) else 0
        hit3 = 1 if hit_at_k(item["expected_file"], result["retrieved_chunks"], 3) else 0

        judge = llm_judge(
            question=item["question"],
            expected_answer=item["expected_answer"],
            actual_answer=result["answer"],
            retrieved_chunks=result["retrieved_chunks"]
        )

        route_ok = 1 if result["routed_source"] == item["expected_source"] else 0

        route_correct.append(route_ok)
        answer_correct.append(judge["correctness"])
        grounded.append(judge["groundedness"])
        completeness_scores.append(judge["completeness"])
        mrr_scores.append(rr)
        hit1_scores.append(hit1)
        hit3_scores.append(hit3)

        detailed_rows.append({
            "question": item["question"],
            "expected_source": item["expected_source"],
            "predicted_source": result["routed_source"],
            "expected_file": item["expected_file"],
            "answer": result["answer"],
            "route_correct": route_ok,
            "mrr": rr,
            "hit@1": hit1,
            "hit@3": hit3,
            "judge_correctness": judge["correctness"],
            "judge_groundedness": judge["groundedness"],
            "judge_completeness": judge["completeness"],
            "judge_notes": judge["notes"]
        })

    summary = {
        "route_accuracy": round(mean(route_correct), 4),
        "answer_accuracy_llm_judge": round(mean(answer_correct), 4),
        "groundedness_rate": round(mean(grounded), 4),
        "avg_completeness": round(mean(completeness_scores), 4),
        "mrr": round(mean(mrr_scores), 4),
        "hit@1": round(mean(hit1_scores), 4),
        "hit@3": round(mean(hit3_scores), 4),
        "samples": len(dataset)
    }

    print(json.dumps(summary, indent=2))

    with open("./evals/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "details": detailed_rows},
            f,
            indent=2
        )


if __name__ == "__main__":
    main()