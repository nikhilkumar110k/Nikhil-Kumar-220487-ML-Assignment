🧠 Code Refactoring & Explanation Tool
A fully automated tool to clone a GitHub repository, refactor code, and generate detailed explanations and suggestions using a fine-tuned CodeT5+.

Supports multiple programming languages out of the box — perfect for code cleanup, reviews, or onboarding documentation.

🚀 Features
🔍 Clone GitHub repos and scan all code files

🛠️ Refactor source code automatically using a trained CodeT5+ model

💬 Explain changes and offer suggestions for improvement

💻 Supports major programming languages:

.py, .js, .ts, .java, .cpp, .c, .cs, .go, .rb, .rs

⚡ Runs with multi-threaded inference for faster processing

🌐 Gradio UI for a clean, local browser-based experience

📸 Demo

🧱 Project Structure
bash
Copy
Edit
.
├── main.py                # Main entrypoint: UI + logic
├── requirements.txt       # Python dependencies
├── models/                # (Optional) place for your model weights
└── README.md              # This file
🧑‍💻 How It Works
Paste your GitHub repo URL into the input box

The tool:

Clones the repo to a temporary folder

Scans for supported source files

Sends code to a fine-tuned LLM

Generates:

✅ Refactored code

🧠 Explanation of changes

💡 Suggestions for improvements

Results are shown directly in your browser

🔧 Setup & Installation
1. Clone this repo
bash
Copy
Edit
git clone https://github.com/yourname/refactor-ai
cd refactor-ai
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. (Optional) Add your own model
Place your fine-tuned CodeT5+ model in the merged-codet5p-model/ directory (or update the MODEL_NAME in main.py).

4. Run the app
bash
Copy
Edit
python main.py
🧠 Model
This tool uses a fine-tuned CodeT5+ model with LoRA support for lightweight, efficient inference. The model is trained to:

Refactor code

Explain the changes in simple terms

Provide optimization and readability suggestions

You can replace it with any Seq2Seq model trained on similar data.
