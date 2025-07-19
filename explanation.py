from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_explanation(refactored_code: str, language: str = "code") -> str:
    if not refactored_code.strip() or len(refactored_code) > 1000:
        return "Explanation could not be generated due to invalid or too long input."
    
    try:
        prompt = f"Explain the purpose of the following {language} code:\n\n{refactored_code}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_length=512)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error during explanation: {str(e)}"

