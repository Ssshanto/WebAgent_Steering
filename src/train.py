import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def train():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_path = "data/training_dataset.json"
    output_dir = "finetuned_model"
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    print(f"Loading model {model_name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # Try BF16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        num_train_epochs=1,
        max_steps=30,
        save_strategy="no",
        fp16=False,
        bf16=True, # Enable BF16
        optim="paged_adamw_32bit",
        report_to="none",
        dataset_text_field="text",
        max_length=1024,
        packing=False
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(output_dir)
    print("Done!")

if __name__ == "__main__":
    train()