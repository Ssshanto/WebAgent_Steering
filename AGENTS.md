# WebAgent_Steering

## active state
- rollback point: `39e72ec` (`2026-02-09`, CAA + BrowserGym MiniWob strategy)
- objective: test whether inference-time CAA improves zero-shot BrowserGym MiniWob++ accuracy under matched task seeds
- benchmark: BrowserGym MiniWob++ only
- default comparison seed: `0`
- default evaluation: `--episodes-per-task 3`
- remote runtime: `cvpc` (host `deeplearningpc01`), env `/mnt/code/steer`, tmux `steer`
- remote repository: `/mnt/code/Reaz/WebAgent_Steering`
- remote GPUs: `2x NVIDIA GeForce RTX 3090`
- MiniWob server: `MINIWOB_URL=http://localhost:8080/miniwob/`

## current research state
- BrowserGym-faithful scaffold is one action per model call and one environment step per action; older retained outputs with multi-line action execution are not directly comparable.
- Current Gemma prompt-only PoC prompt pair is `gemma_tree_vs_label`, targeting AX-tree bid grounding and executable action discipline.
- Gemma-3-4B `gemma_tree_vs_label` prompt-only on 5 tasks / 100 matched BrowserGym-seed episodes (`click-button,click-link,click-option,choose-list,focus-text`, seed `42`): baseline `45/100`, positive `70/100`, negative `0/100`; parse-fail baseline `42%`, positive `7%`, negative `100%`.
- Current best CAA vector is `gemma_tree_pos_minus_base`: positive prompt minus empty baseline prompt, Gemma-3-4B layer `17`, `--steer-position last`, alpha around `1000`.
- Targeted alpha knob on the 25 seeds where baseline failed and positive succeeded: baseline `0/25`; alpha `-500` `0/25`, `250` `10/25`, `500` `15/25`, `750` `19/25`, `1000` `19/25`; random vector alpha `750` was `0/25`.
- Full seed-42 5-task / 100-episode confirmation: baseline `45/100`, CAA L17 alpha `1000` `65/100`; parse-fail `42% -> 11%`; paired gains/losses `20/0`.
- Fresh seed-43 5-task / 100-episode transfer: baseline `43/100`, CAA L17 alpha `1000` `64/100`; parse-fail `45% -> 11%`; paired gains/losses `21/0`.
- Full seed-42 controls at L17 alpha `1000`: random vector `44/100`, reverse-sign CAA `28/100`, syntax-only vector `38/100`. These controls support specificity: the successful CAA direction is not just large-norm perturbation or generic action syntax.
- Effect is strongest on `focus-text`, with smaller gains on `click-link` and `click-option`; treat the behavior as "bid grounding / valid action selection", not broad web-agent competence yet.
- SAE readout used Gemma Scope 2 Gemma-3-4B-IT layer-17 width-16k L0-medium residual SAE. The injected vector most increases feature `1246`, Neuronpedia label `UI elements and styling` with top tokens `buttons`, `decorative`, `button`, `buttons`, `popup`; this matches the web-action/UI grounding interpretation.
- Decoder-alignment readout also shows a bid-number cluster: feature `200`, label `numbers and quantities`, and feature `62`, label `numbers and quantities`; raw top decoder feature `510` is unlabeled and should not be used as the semantic claim.
- Gemma Scope 2 transcoders can be scored locally, but Neuronpedia labels for Gemma-3-4B transcoders were not available; do not claim named transcoder semantics yet.
- Direct SAE feature vectors were prepared but not evaluated because CUDA became unavailable on `cvpc` (`torch.cuda.is_available() == False`, `cudaGetDeviceCount` invalid ordinal). Cached vectors: `sae1246_dec`, `sae1246_enc`, `sae1246_200_62_dec`, `sae1246_200_62_enc`.
- Historical Qwen results exist but are not the current path; prioritize Gemma-3-4B because prompt-only separation is clean and SAE/transcoder analysis is planned later.

## method
- compute contrastive activation vectors from positive/negative steering prompts
- cache one vector per transformer layer under `vectors/<model>/seed_<seed>/`
- run paired baseline and steered episodes with identical `(task, seed)`
- report success, parse-fail rate, and baseline-vs-steered delta
- for Gemma runs, separate reward from action validity: report success, parse-fail, invalid-bid rate, action-type validity, and per-task deltas

## research operating principles
- Act like a researcher, not a job runner: every experiment must test a concrete hypothesis and have a pre-decided interpretation path.
- Stop stale or failing runs quickly when infrastructure, prompts, parsing, or metrics show the run cannot answer the hypothesis; do not spend hours debugging a low-value branch.
- Pivot aggressively toward the highest-information next test: prefer small targeted slices, paired controls, and action-validity diagnostics before broad sweeps.
- Do not conclude from weak or noisy evidence. Return only after obtaining a meaningful positive result, a meaningful negative result, or a clearly documented blocker that prevents the decisive test.
- Treat prompt-only baselines as serious competitors. Steering claims must beat or complement strong prompting, repair/retry, random-vector, reverse-sign, and syntax-only controls.
- Preserve GPU time and tokens: reuse cached vectors/baselines, run matched slices first, use both GPUs only for independent high-value jobs, and kill redundant or dominated settings.
- Verify code correctness before trusting surprising results. Check JSONL trajectories, parsed actions, paired gains/losses, and metric definitions when outcomes change.
- Log decisions in `RESEARCH.md`: why a run was started, why it was stopped or expanded, what result changed the research direction, and what should not be rerun.

## next executable tasks
- Stage 0, preserve the confirmed artifacts; do not recompute the vector unless intentionally changing the vector definition.
- Stage 1, restore CUDA on `cvpc` before further model runs; do not run Gemma-3-4B experiments on CPU.
- Stage 2, evaluate direct SAE feature vectors on the 25 targeted seed slice: `sae1246_dec`, `sae1246_enc`, `sae1246_200_62_dec`, `sae1246_200_62_enc` with raw alphas `250,500,750,1000`.
- Stage 3, measure average L17 residual magnitude on the same prompt slice and report all raw alphas as `alpha / avg_l17_norm`.
- Stage 4, test larger successful-vector alphas on the 25 targeted seed slice: `1250,1500,2000`, reported as percent of average L17 residual magnitude.
- Stage 5, if direct SAE feature steering works, confirm the best feature setting on the full 100-slice.
- Do not claim broad web-agent behavior until improvement transfers beyond the bid-grounding task family.

## command skeletons
- Remote env prefix: `cd /mnt/code/Reaz/WebAgent_Steering && export HF_HOME=/mnt/code/huggingface MINIWOB_URL=http://localhost:8080/miniwob/ BROWSERGYM_CHROMIUM_EXECUTABLE=/usr/bin/chromium-browser TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`
- Confirmed seed-42 setting: `CUDA_VISIBLE_DEVICES=1 /mnt/code/steer/bin/python scripts/run_sweep.py --model gemma-3-4b --prompt-type gemma_tree_pos_minus_base --vector-method prompt --plan-file /tmp/gemma_100_plan.txt --episode-steps 6 --layers 17 --alphas=1000 --steer-position last --steer-only --base-jsonl results/gemma3_4b_100_baseline_from_prompt_json.jsonl --train-steps 25 --cache-dir vectors --seed 0 --out-dir results/gemma3_4b_pos_minus_base_100_L17_a1000`
- Transfer seed-43 setting: `CUDA_VISIBLE_DEVICES=1 /mnt/code/steer/bin/python scripts/run_sweep.py --model gemma-3-4b --prompt-type gemma_tree_pos_minus_base --vector-method prompt --tasks click-button,click-link,click-option,choose-list,focus-text --plan-file /tmp/gemma_transfer_seed43_100_plan.txt --episode-steps 6 --layers 17 --alphas=1000 --steer-position last --train-steps 25 --cache-dir vectors --seed 0 --out-dir results/gemma3_4b_pos_minus_base_transfer_seed43_L17_a1000`

## file hierarchy
- `src/miniwob_steer.py`: BrowserGym evaluation, model loading, CAA vector computation, inference hook
- `scripts/run_sweep.py`: model-once layer/alpha sweep runner
- `README.md`: setup and command examples
- `RESEARCH.md`: durable method basis and result summaries
- `requirements.txt`: runtime dependencies
- `results/`: ignored experiment outputs
- `vectors/`: ignored cached steering vectors

## repository policy
- keep only code that directly supports CAA MiniWob evaluation or sweep execution
- no standalone toy probes, one-off manifests, generated queue scaffolds, or VLM leftovers
- prefer short comments explaining research choices over defensive boilerplate
- heavy experiments run on `cvpc`; code changes happen locally and sync through non-destructive `rsync`
- do not use `rsync --delete`; remote-only files must not be overwritten or removed casually
