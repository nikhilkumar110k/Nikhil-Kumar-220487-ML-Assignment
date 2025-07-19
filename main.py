import os
import tempfile
import shutil
import logging
from git import Repo
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import stat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".cs", ".go", ".rb", ".rs"}
MODEL_NAME = "merged-codet5p-model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
if not getattr(model, 'is_loaded_in_8bit', False):
    model = model.to(device)

def get_all_supported_files(repo_path):
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(repo_path)
        for file in files
        if Path(file).suffix in SUPPORTED_EXTENSIONS
    ]

def run_inference_on_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            original_code = f.read()
        if not original_code.strip():
            return filepath, "Skipped: Empty file"

        prompt = f"""Refactor the following code and provide explanation and suggestions:\n\n### Original Code:\n{original_code}\n\n### Refactored Code:\n"""

        logging.info(f"Running inference on: {filepath}")
        start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        outputs = model.generate(**inputs, max_length=2048)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        duration = time.time() - start
        logging.info(f"Inference completed for {filepath} in {duration:.2f} seconds")

        return filepath, result

    except Exception as e:
        logging.exception(f"Error processing {filepath}")
        return filepath, f"Error: {str(e)}"

def handle_remove_readonly(func, path, _):
    """Fix for Windows read-only files on rmtree"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_and_analyze_repo(repo_url):
    temp_dir = tempfile.mkdtemp()
    try:
        logging.info(f"Cloning repo: {repo_url}...")
        yield "Cloning repository...", None

        Repo.clone_from(repo_url, temp_dir)
        yield "Cloning complete. Scanning files...", None

        files = get_all_supported_files(temp_dir)
        yield f"Found {len(files)} supported files. Running inference...", None

        results = []
        for i, file in enumerate(files, 1):
            _, output = run_inference_on_file(file)
            results.append(f"## {Path(file).name}\n\n{output}")
            yield "\n\n".join(results), f"Completed {i}/{len(files)} files..."

    except Exception as e:
        yield f"Error: {str(e)}", None

    finally:
        shutil.rmtree(temp_dir)

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Code Refactoring & Explanation Tool")
    repo_input = gr.Textbox(label="GitHub Repo URL", placeholder="https://github.com/user/repo")
    run_button = gr.Button("Refactor and Explain")
    output_area = gr.Textbox(label="Refactored Output", lines=25)
    progress_text = gr.Textbox(label="Progress", interactive=False)

    run_button.click(
        fn=clone_and_analyze_repo,
        inputs=repo_input,
        outputs=[output_area, progress_text],
        show_progress=True
    )


if __name__ == "__main__":
    demo.launch()
