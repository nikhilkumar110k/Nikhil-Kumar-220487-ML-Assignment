import gradio as gr
import os
import tempfile
import shutil
from git import Repo
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

model = None
tokenizer = None
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")

language_map = {
    ".py": "python",
    ".js": "javascript",
    ".java": "java",
    ".ts": "typescript",
    ".cpp": "cpp",
    ".c": "c",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift"
}

retriever_db_path = "instruction_dataset.jsonl"
retriever_index = []
retriever_instructions = []
retriever_outputs = []

if os.path.exists(retriever_db_path):
    with open(retriever_db_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            retriever_outputs.append(entry["output"])
            retriever_instructions.append(entry["instruction"])
    retriever_index = retriever_model.encode(retriever_outputs, convert_to_tensor=True)


def retrieve_similar_instruction_contexts(code, top_k=3):
    if not retriever_index:
        return ""
    query_embedding = retriever_model.encode(code, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, retriever_index, top_k=top_k)[0]
    contexts = [retriever_instructions[hit['corpus_id']] for hit in hits]
    return "\n\n".join(contexts)

def fallback_rag_inference(prompt, fallback_docs=None):
    if fallback_docs is None:
        fallback_docs = [
            "Refactor large functions into smaller ones to improve readability and testability.",
            "Avoid deeply nested code; use guard clauses instead.",
            "Use meaningful variable and function names to make code self-documenting.",
            "Always handle errors gracefully and validate inputs.",
            "Follow the single-responsibility principle for cleaner architecture."
        ]
    import difflib
    closest_doc = difflib.get_close_matches(prompt, fallback_docs, n=1)
    return closest_doc[0] if closest_doc else "No helpful explanation found."

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        print("[LOG] Loading merged model and tokenizer...")
        merged_model_path = "./merged-codet5p-model"
        tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(merged_model_path)
        model.eval()
        print("[LOG] Merged model loaded and ready.")

def clone_repo(repo_url: str, clone_dir: str):
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    Repo.clone_from(repo_url, clone_dir)

def get_code_files(repo_dir: str):
    code_files = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for root, _, files in os.walk(repo_dir):
            for file in files:
                if file.endswith(tuple(language_map.keys())):
                    filepath = os.path.join(root, file)
                    futures.append(executor.submit(read_code_file, filepath))
        for future in as_completed(futures):
            result = future.result()
            if result:
                code_files.append(result)
    return code_files

def read_code_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return (filepath, f.read())
    except Exception:
        return None

def split_response(full_response):
    explanation_header = re.search(r"(#{2,4}\s*🧠\s*Explanation\s*:?)", full_response, re.IGNORECASE)
    if explanation_header:
        idx = explanation_header.start()
        refactored = full_response[:idx].replace("### ✨ Refactored Code:", "").strip("` \n")
        explanation = full_response[idx + len(explanation_header.group(0)):].strip()
        return refactored, explanation
    return full_response.strip(), "Explanation not found."

def analyze_and_refactor(code_snippets):
    print(f"[LOG] Starting threaded refactoring for {len(code_snippets)} files...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_file(item):
        filepath, code = item
        file_ext = os.path.splitext(filepath)[1]
        language = language_map.get(file_ext, "")

        context = retrieve_similar_instruction_contexts(code)

        prompt = (
            f"Refactor the following {language} code. Respond in Markdown format with:\n"
            f"### ✨ Refactored Code:\n```{language}\n...\n```\n\n"
            f"### 🧠 Explanation:\n...\n\n"
            f"Code:\n{code}\n\n"
            f"Context:\n{context}\n"
        )

        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, max_new_tokens=512)

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        refactored_code, explanation = split_response(full_response)

        if explanation.strip().lower() in ["", "explanation not found."]:
            explanation = fallback_rag_inference(prompt)

        print(f"[LOG] ✅ Refactored: {os.path.basename(filepath)}")
        return (filepath, refactored_code, explanation, language)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_file, code_snippets))

    print("[LOG] All files refactored.")
    return results

def format_results(results):
    output = ""
    for path, refactored_code, explanation, language in results:
        output += f"### 📄 File: {path}\n\n"
        output += f"#### ✨ Refactored Code:\n```{language}\n{refactored_code}\n```\n\n"
        output += f"#### 🧠 Explanation:\n{explanation}\n"
        output += f"\n{'-'*80}\n\n"
    return output

def process_github_repo(repo_url):
    print(f"[LOG] Received repo URL: {repo_url}")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            load_model()
            print(f"[LOG] Cloning repo into: {temp_dir}")
            clone_repo(repo_url, temp_dir)
            print(f"[LOG] ✅ Repo cloned.")
            code_files = get_code_files(temp_dir)
            print(f"[LOG] Found {len(code_files)} code files.")

            if not code_files:
                return "No supported code files found in the repository."

            print("[LOG] 🔍 Starting analysis and refactoring...")
            results = analyze_and_refactor(code_files)
            return format_results(results)
        except Exception as e:
            print(f"[ERROR] {e}")
            return f"❌ Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## 🛠️ Multi-language Code Refactor & Explanation Engine")
    repo_input = gr.Textbox(label="GitHub Repository URL", placeholder="https://github.com/user/repo")
    analyze_btn = gr.Button("Analyze & Refactor")
    output_box = gr.Textbox(label="Results", lines=30, interactive=False)

    analyze_btn.click(fn=process_github_repo, inputs=repo_input, outputs=output_box)

demo.launch()