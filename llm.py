from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    torch.cuda.empty_cache()

    print("📦 Loading and filtering dataset...")
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train[:60%]")  
    keywords = ["refactor", "optimize", "clean", "secure", "improve", "fix"]
    filtered_dataset = dataset.filter(
        lambda x: any(kw in x["instruction"].lower() for kw in keywords)
    )

    def format_instruction(example):
        return {
            "text": f"Instruction: {example['instruction']}\n\nCode:\n{example['input']}\n\nRefactored/Improved:\n{example['output']}"
        }

    formatted_dataset = filtered_dataset.map(format_instruction)

    model_id = "Salesforce/codet5p-770m"
    print("🤖 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_8bit=True,
        torch_dtype=torch.float16
    )

    print("🔧 Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, peft_config)

    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=384  
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = formatted_dataset.map(
        tokenize,
        batched=True,
        remove_columns=formatted_dataset.column_names
    )

    print("🛠️ Setting training arguments...")
    training_args = TrainingArguments(
        output_dir="./codet5p-alpaca-refactor-lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )

    print("🚀 Training started...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    print("💾 Saving LoRA fine-tuned model...")
    model.save_pretrained("./codet5p-alpaca-refactor-lora")
    tokenizer.save_pretrained("./codet5p-alpaca-refactor-lora")

    try:
        print("🧠 Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained("./merged-codet5p-model")
        tokenizer.save_pretrained("./merged-codet5p-model")
    except Exception as e:
        print(f"❗ Could not merge model: {e}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
