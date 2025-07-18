import json
from difflib import SequenceMatcher

def load_instruction_dataset(dataset_path: str):
    """Loads the instruction-tuning dataset."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def similarity(a: str, b: str) -> float:
    """Computes a simple string similarity ratio between two code snippets."""
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()

def retrieve_similar_instruction_contexts(
    refactored_code: str,
    instruction_dataset_path: str,
    top_k: int = 3,
    similarity_threshold: float = 0.6
):
    """Retrieves top-k similar instruction contexts from dataset based on output similarity."""
    dataset = load_instruction_dataset(instruction_dataset_path)
    scored = []

    for sample in dataset:
        output = sample.get("output", "")
        score = similarity(refactored_code, output)
        if score >= similarity_threshold:
            scored.append((score, sample))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [s[1] for s in scored[:top_k]]
