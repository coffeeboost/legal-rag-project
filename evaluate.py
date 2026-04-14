"""
RAGAS evaluation — measure RAG quality with 3 key metrics:
  - Faithfulness:      Does the answer stay within the retrieved context?
  - Answer Relevancy:  Is the answer relevant to the question?
  - Context Recall:    Did we retrieve the right chunks?

Usage:
    python evals/evaluate.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

from src.generation.chain import RAGChain


# ── Golden eval set ───────────────────────────────────────────────────────────
# Build this manually. 20-30 Q&A pairs with known answers = very impressive on resume.
# Format: question, ground_truth_answer (what the correct answer should be)

GOLDEN_DATASET = [
    {
        "question": "What is the indemnification clause in the contract?",
        "ground_truth": "The indemnification clause requires Party A to hold harmless Party B for any third-party claims.",
    },
    {
        "question": "What is the governing law of this agreement?",
        "ground_truth": "This agreement is governed by the laws of the State of New York.",
    },
    # Add more Q&A pairs here as you build your dataset
]


def run_evaluation():
    chain = RAGChain()

    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"Running evaluation over {len(GOLDEN_DATASET)} questions...\n")

    for item in GOLDEN_DATASET:
        result = chain.query(item["question"])
        questions.append(item["question"])
        answers.append(result.answer)
        contexts.append([chunk.content for chunk in result.sources])
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("Computing RAGAS metrics...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )

    print("\n── RAGAS Results ──────────────────────────")
    print(f"  Faithfulness:      {results['faithfulness']:.3f}")
    print(f"  Answer Relevancy:  {results['answer_relevancy']:.3f}")
    print(f"  Context Recall:    {results['context_recall']:.3f}")
    print("───────────────────────────────────────────\n")

    return results


if __name__ == "__main__":
    run_evaluation()
