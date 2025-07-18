import gradio as gr
import os
import tempfile
import shutil
from git import Repo
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define globals
model = None
tokenizer = None

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
                if file.endswith((".py", ".js", ".java", ".ts", ".cpp", ".c", ".go", ".rs", ".rb", ".php", ".cs", ".swift")):
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

def analyze_and_refactor(code_snippets):
    print(f"[LOG] Starting threaded refactoring for {len(code_snippets)} files...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_file(item):
        filepath, code = item
        prompt = f"Refactor the following code:\n{code}"
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, max_new_tokens=256)

        refactored_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

        explanation_prompt = f"Explain the changes made in this refactoring and why they improve the code:\n\nOriginal Code:\n{code}\n\nRefactored Code:\n{refactored_code}"
        explanation_input_ids = tokenizer(explanation_prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)

        with torch.no_grad():
            explanation_output = model.generate(input_ids=explanation_input_ids, max_new_tokens=256)

        explanation = tokenizer.decode(explanation_output[0], skip_special_tokens=True)

        print(f"[LOG] ✅ Refactored: {os.path.basename(filepath)}")
        return (filepath, refactored_code, explanation)


    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_file, code_snippets))

    print("[LOG] All files refactored.")
    return results

def format_results(results):
    output = ""
    for path, refactored_code, explanation in results:
        output += f"### 📄 File: {path}\n\n"
        output += f"#### ✨ Refactored Code:\n```go\n{refactored_code}\n```\n\n"
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
    gr.Markdown("## 🛠️ Multi-language Code Refactor & Vulnerability Analyzer")
    repo_input = gr.Textbox(label="GitHub Repository URL", placeholder="https://github.com/user/repo")
    analyze_btn = gr.Button("Analyze & Refactor")
    output_box = gr.Textbox(label="Results", lines=30, interactive=False)

    analyze_btn.click(fn=process_github_repo, inputs=repo_input, outputs=output_box)

demo.launch()
