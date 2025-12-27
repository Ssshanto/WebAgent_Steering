# WebAgent Steering: Zero-Shot JSON Control

This repository contains a Minimal Viable Proof-of-Concept (POC) for **Representation Engineering (Steering Vectors)** applied to Web Agents.

## Goal
To control the output format of a small (0.5B) Language Model, forcing it to output **Strict JSON** for web automation tasks instead of its default "Chatty" behavior, without fine-tuning.

## Results
**Success!** We extracted a "JSON-Vector" from `Qwen/Qwen2.5-0.5B-Instruct` using a synthetic contrastive dataset.

| Condition | Coefficient | Output Behavior |
| :--- | :--- | :--- |
| **Baseline** | 0.0 | Ignores instructions, outputs "Click on 'Login' button..." and HTML snippets. |
| **Steered** | **+2.0 (Layer 12)** | **Spontaneously generates JSON:** `Response: { "action": "click", "target": "..." }` |
| **Negative** | -2.0 | Chatty, unstructured text. |

## How to Run

1.  **Install Environment:**
    ```bash
    conda create -n steer python=3.10
    conda activate steer
    pip install torch transformers accelerate scikit-learn numpy tqdm datasets
    ```

2.  **Generate Dataset (Contrastive Pairs):**
    ```bash
    python src/dataset_generator.py
    ```
    *Creates `data/contrastive_dataset.json` from Mind2Web.*

3.  **Extract Steering Vector:**
    ```bash
    python src/extract_vector.py
    ```
    *Saves vectors to `vectors/steering_vectors.json`.*

4.  **Test/Sweep:**
    ```bash
    python src/sweep_layers.py
    ```
    *Runs the model with and without the vector on a test prompt.*

## Methodology
1.  **Dataset:** We constructed 50 pairs of (JSON vs. Chat) completions for the same HTML inputs using the Mind2Web dataset.
2.  **Extraction:** We used **PCA** on the difference of hidden states (`State_JSON - State_Chat`) at the last token position.
3.  **Inference:** We injected this vector during the forward pass using a PyTorch hook.