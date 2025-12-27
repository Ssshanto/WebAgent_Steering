# WebAgent Steering: Zero-Shot JSON Control

This repository contains a Minimal Viable Proof-of-Concept (POC) for **Representation Engineering (Steering Vectors)** applied to Web Agents.

## Goal
To control the output format of a small (0.5B) Language Model, forcing it to output **Strict JSON** for web automation tasks instead of its default "Chatty" behavior, without fine-tuning.

## Results
**Success!** We extracted a "JSON-Vector" from `Qwen/Qwen2.5-0.5B-Instruct` using a synthetic contrastive dataset.

| Condition | Coefficient | Output Behavior | Valid JSON Rate |
| :--- | :--- | :--- | :--- |
| **Baseline** | 0.0 | Ignores instructions, outputs "Click on 'Login' button..." and HTML snippets. | **20%** |
| **Steered** | **+2.0 (Layer 12)** | **Spontaneously generates JSON:** `Response: { "action": "click", "target": "..." }` | **40%** |
| **Negative** | -2.0 | Chatty, unstructured text. | N/A |
| **Fine-Tuned** | N/A | Perfectly follows format. | **100%** |

## How to Run

1.  **Install Environment:**
    ```bash
    conda create -n steer python=3.10
    conda activate steer
    pip install torch transformers accelerate scikit-learn numpy tqdm datasets peft trl bitsandbytes
    ```

2.  **Generate Data:**
    ```bash
    python src/dataset_generator.py       # POC Dataset (50 items)
    python src/prepare_training_data.py   # Training Dataset (200 items)
    ```

3.  **Steering (Zero-Shot):**
    ```bash
    python src/extract_vector.py
    python src/test_steering.py
    ```

4.  **Fine-Tuning:**
    ```bash
    python src/train.py
    python src/evaluate.py  # Generates results.csv
    ```

## Methodology
1.  **Dataset:** We constructed 50 pairs of (JSON vs. Chat) completions for the same HTML inputs using the Mind2Web dataset.
2.  **Extraction:** We used **PCA** on the difference of hidden states (`State_JSON - State_Chat`) at the last token position.
3.  **Inference:** We injected this vector during the forward pass using a PyTorch hook.