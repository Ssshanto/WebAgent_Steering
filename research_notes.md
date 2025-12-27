# LLM Steering for WebAgents - Research Notes

**Goal:** ACL 2026 Short Paper.

## Project Directions
- **Steering for Planning Integrity:** Ensuring WebAgents maintain long-horizon goals without drifting during multi-step navigation.
- **Decision-Correctness Steering:** Modifying activations to favor "safe" or "valid" DOM interactions over hallucinatory ones.
- **Cross-Domain Transfer:** Investigating if steering vectors for "task-adherence" transfer across different web environments (e.g., e-commerce to travel booking).

## Relevant Papers
- **Zou et al. (2023):** "Representation Engineering: A Top-Down Approach to AI Transparency" (Foundational for RepE).
- **Turner et al. (2023):** "Activation Addition: Steering Language Models Without Optimization" (Initial steering vectors).
- **Li et al. (2023):** "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model" (Truthfulness focus).

## Todo
- [ ] Select target benchmark (WebArena / Mind2Web / WebShop).
- [ ] Identify candidate layers for steering in Llama-3/Mistral.
