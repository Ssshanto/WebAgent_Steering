# LLM Steering for WebAgents - Research Notes

**Goal:** ACL 2026 Short Paper.

## Project Directions
- **Steering for Planning Integrity:** Ensuring WebAgents maintain long-horizon goals without drifting during multi-step navigation.
- **Decision-Correctness Steering:** Modifying activations to favor "safe" or "valid" DOM interactions over hallucinatory ones.
- **Cross-Domain Transfer:** Investigating if steering vectors for "task-adherence" transfer across different web environments (e.g., e-commerce to travel booking).

## Future Work: Moving Beyond Synthetic Pairs
**Current Limitation:**
Our current steering vectors are derived from *synthetic* contrastive pairs (Fixed "I will click..." vs. Fixed JSON).
*   **Risk:** The synthetic negative might not match the model's *actual* internal bias (e.g., the model might be thinking in "Imperative English" rather than "Future Tense English").
*   **Consequence:** Subtracting a vector that represents a bias the model *doesn't have* introduces noise/damage (the Orthogonality Problem).

**Proposed Solution: Data-Driven Self-Correction**
Instead of guessing the negative behavior, we should record it.
1.  **Run Inference:** Let the model generate naturally on the prompt (capturing the *actual* failure mode).
2.  **Teacher Force:** Force the model to generate the correct JSON (capturing the desired state).
3.  **Compute Vector:** `Vector = Mean(State_TeacherForced - State_NaturalFailure)`.
4.  **Benefit:** This ensures the vector is mathematically aligned with the exact transformation needed to correct the model's specific "bad habits."

## Relevant Papers
- **Zou et al. (2023):** "Representation Engineering: A Top-Down Approach to AI Transparency" (Foundational for RepE).
- **Turner et al. (2023):** "Activation Addition: Steering Language Models Without Optimization" (Initial steering vectors).
- **Li et al. (2023):** "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model" (Truthfulness focus).

## Todo
- [x] Select target benchmark (WebArena / Mind2Web / WebShop).
- [x] Identify candidate layers for steering in Llama-3/Mistral (Used Qwen2.5-0.5B).
- [ ] Implement Self-Correction Steering (Data-Driven extraction).