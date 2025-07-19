import gradio as gr
import os
import tempfile
import shutil
from git import Repo
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

def clean_invalid_unicode(text):
    if isinstance(text, str):
        return text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return text

model = None
tokenizer = None
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")

explanation_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
explanation_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
explanation_pipeline = pipeline("text2text-generation", model=explanation_model, tokenizer=explanation_tokenizer)

language_map = {
    ".py": "python", ".js": "javascript", ".java": "java", ".ts": "typescript",
    ".cpp": "cpp", ".c": "c", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".php": "php", ".cs": "csharp", ".swift": "swift"
}

retriever_db_path = "instruction_dataset.jsonl"
retriever_index, retriever_instructions, retriever_outputs = [], [], []

if os.path.exists(retriever_db_path):
    with open(retriever_db_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            retriever_outputs.append(clean_invalid_unicode(entry["output"]))
            retriever_instructions.append(clean_invalid_unicode(entry["instruction"]))
    retriever_index = retriever_model.encode(retriever_outputs, convert_to_tensor=True)

def retrieve_similar_contexts(code, corpus, top_k=3):
    if not corpus:
        return ""
    query_embedding = retriever_model.encode(code, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus, top_k=top_k)[0]
    return "\n".join([corpus[hit["index"]] for hit in hits])

def retrieve_similar_output_contexts(code, top_k=3):
    return retrieve_similar_contexts(code, retriever_outputs, top_k)

def retrieve_similar_instruction_contexts(code, top_k=3):
    return retrieve_similar_contexts(code, retriever_instructions, top_k)

def retrieve_combined_instruction_output_contexts(code, top_k=3):
    instructions = retrieve_similar_instruction_contexts(code, top_k)
    outputs = retrieve_similar_output_contexts(code, top_k)
    return instructions + "\n\n" + outputs

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
    response = closest_doc[0] if closest_doc else "No helpful explanation found."
    return clean_invalid_unicode(response)

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        print("[LOG] Loading merged model and tokenizer...")
        merged_model_path = "./merged-codet5p-model"
        tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(merged_model_path)
        model.eval()
        print("[LOG] Merged model loaded and ready.")

def clone_repo(repo_url, clone_dir):
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    Repo.clone_from(repo_url, clone_dir)

def get_code_files(repo_dir):
    code_files, futures = [], []
    with ThreadPoolExecutor() as executor:
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
            content = f.read()
            return (filepath, clean_invalid_unicode(content))
    except Exception:
        return None

def generate_explanation_and_suggestions(code: str, refactored_code: str, language: str = "code") -> (str, str):
    prompt = f"""
You are a senior software developer reviewing changes in a {language} codebase.

Original Code:
{code}

Refactored Code:
{refactored_code}

Please do two things:
1. Explain what was changed and why (in simple terms).
2. Suggest improvements, highlight any potential bugs, or recommend best practices.
""".strip()

    try:
        if len(prompt.split()) > 400:
            prompt = prompt[:1500]
        result = explanation_pipeline(prompt, max_length=512, do_sample=False)
        output_text = result[0]['generated_text']
        return clean_invalid_unicode(output_text), ""  # no second split needed
    except Exception:
        fallback = fallback_rag_inference(code)
        return fallback, ""



def analyze_and_refactor(code_snippets):
    print(f"[LOG] Starting threaded refactoring for {len(code_snippets)} files...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_file(item):
        filepath, code = item
        file_ext = os.path.splitext(filepath)[1]
        language = language_map.get(file_ext, "")

        prompt = (
            f"You are a skilled software engineer.\n"
            f"Refactor the following {language} code to improve readability and maintainability.\n"
            f"Respond ONLY with the refactored code, without any extra explanation.\n\n"
            f"{code}"
        )
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(model.device)
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, max_new_tokens=512)
        refactored_code = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        explanation, suggestions = generate_explanation_and_suggestions(code, refactored_code, language)

        return (
            clean_invalid_unicode(filepath),
            clean_invalid_unicode(refactored_code),
            clean_invalid_unicode(explanation),
            clean_invalid_unicode(suggestions),
            language
        )   
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(process_file, code_snippets))

def format_results(results):
    output = ""
    for path, refactored_code, explanation, suggestions, language in results:
        output += f"### 📄 File: {path}\n\n"
        output += f"#### ✨ Refactored Code:\n```{language}\n{refactored_code}\n```\n\n"
        output += f"#### 🧠 Explanation:\n{explanation}\n\n"
        output += f"#### 💡 Suggestions:\n{suggestions if suggestions else 'No additional suggestions.'}\n"
        output += f"\n{'-'*80}\n\n"
    return clean_invalid_unicode(output)


def process_github_repo(repo_url):
    print(f"[LOG] Received repo URL: {repo_url}")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            load_model()
            print(f"[LOG] Cloning repo into: {temp_dir}")
            clone_repo(repo_url, temp_dir)
            print("[LOG] ✅ Repo cloned.")
            code_files = get_code_files(temp_dir)
            print(f"[LOG] Found {len(code_files)} code files.")
            if not code_files:
                return "No supported code files found in the repository."
            print("[LOG] 🔍 Starting analysis and refactoring...")
            results = analyze_and_refactor(code_files)
            return format_results(results)
        except Exception as e:
            print(f"[ERROR] {e}")
            return f"❌ Error: {clean_invalid_unicode(str(e))}"

with gr.Blocks() as demo:
    gr.Markdown("## 🛠️ Multi-language Code Refactor & Explanation Engine")
    repo_input = gr.Textbox(label="GitHub Repository URL", placeholder="https://github.com/user/repo")
    analyze_btn = gr.Button("Analyze & Refactor")
    output_box = gr.Textbox(label="Results", lines=30, interactive=False)

    analyze_btn.click(fn=lambda url: process_github_repo(url.strip()), inputs=repo_input, outputs=output_box)

demo.launch(share=True)
