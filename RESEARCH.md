# Research Notes

## objective

Test whether inference-time steering improves zero-shot web-agent accuracy for small language models, primarily by improving action grounding, without fine-tuning.

Primary questions:

1. Can activation steering move an LLM toward valid, correct web actions under matched task seeds?
2. Can the behavioral effect be mechanistically interpreted with SAE features?
3. Can interpretability-derived directions become the best steering method, rather than only explaining raw CAA after the fact?
4. Can the method be made dataset-independent enough to improve action grounding outside the original MiniWob scaffold?

Current success criteria: paired baseline/steered improvement on identical task seeds; parse-fail, invalid-bid, and per-task deltas reported; feature-steering or projection/ablation evidence before making a mechanistic claim; ID-remapping/frozen-state evidence before making a dataset-independent grounding claim.

Current narrowed claim: Gemma-3-4B has a clean CAA vector for AX-tree bid grounding and executable action selection. The effect improves reward on two matched 100-episode seed streams and behaves like an alpha knob on targeted failure seeds. It is not yet evidence for broad web-agent competence, dataset-independent action grounding, or SAE-mediated causality.

## method

### Contrastive Activation Addition (CAA)

1. Build BrowserGym prompt from goal, accessibility tree, DOM, action space, and previous error.
2. Append positive/negative steering instruction.
3. Extract activation difference by layer.
4. Normalize and cache vectors.
5. Add `alpha * vector` at the target layer during generation.
6. Evaluate paired baseline and steered episodes.

### Experimental Setup

- dataset: BrowserGym MiniWob++
- default vector samples: 200 for broader runs; 25 for quick proof-of-concept runs
- default evaluation: `--episodes-per-task 3`
- default seed for comparisons: `0`
- default layer choice: mid-depth unless swept
- current scaffold: one model action per environment step; multi-line model outputs are not executed as action sequences

## current results

### Interface-Variant Grounding Scaffold, May 2026

Implemented the zero-shot action-interface steering scaffold for testing whether CAA can act as a representation-level adapter under id-schema shift. The implementation is experiment infrastructure only; no new GPU result should be claimed from this note.

Added a shared interface-variant layer used by vector construction, frozen one-step evaluation, and stepped MiniWob evaluation. Supported modes are `original`, `permuted`, `alphanumeric`, `structured`, `uuid`, `handle`, `mixed`, `longprefix`, `stale_ids`, `fake_examples`, and `decoy_labels`. Each transformed prompt carries a reversible shown-id to real-bid map so stepped MiniWob actions can execute against real BrowserGym bids.

New intended ladder:

1. Sanity check frozen metrics on target25 before any broad sweep.
2. Build source-mode vectors only from `original,permuted,alphanumeric,fake_examples,stale_ids`.
3. Evaluate held-out schemas `structured,uuid,handle,mixed,longprefix,decoy_labels` under the same visible prompt.
4. Promote only vectors that improve held-out `valid_current_id` by at least 15 points and preserve stepped success with paired losses under 5%.

New metrics to report by condition and interface mode: `parse_valid`, `action_type_valid`, `valid_current_id`, `copied_example_id`, `stale_id`, `label_as_id`, `bogus_argument`, and stepped `success`/reward. Do not tune alpha on held-out schemas; use MiniWob-selected layer/alpha settings.

First target25 interface-variant result on `cvpc`, Gemma-3-4B, confirmed `gemma_tree_pos_minus_base` L17 alpha `1000`, minimal prompt:

| Interface mode | Frozen baseline valid current id | Frozen CAA valid current id | Delta | Positive prompt valid current id |
| --- | ---: | ---: | ---: | ---: |
| `original` | 0.00 | 0.76 | +0.76 | 0.96 |
| `permuted` | 0.00 | 0.76 | +0.76 | 0.88 |
| `alphanumeric` | 0.48 | 0.84 | +0.36 | 0.96 |
| `mixed` | 0.52 | 0.88 | +0.36 | 0.88 |
| `stale_ids` | 0.12 | 0.80 | +0.68 | 0.28 |
| `structured` | 0.00 | 0.32 | +0.32 | 0.96 |
| `decoy_labels` | 0.00 | 0.16 | +0.16 | 0.88 |
| `fake_examples` | 0.00 | 0.12 | +0.12 | 0.96 |
| `longprefix` | 0.04 | 0.08 | +0.04 | 0.16 |
| `handle` | 0.00 | 0.00 | +0.00 | 0.28 |
| `uuid` | 0.88 | 0.80 | -0.08 | 0.96 |

Stepped target25 confirmation after guarding against accidental execution of original real bids under remapped schemas:

| Interface mode | Baseline success | CAA success | Paired gains/losses | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `stale_ids` | 3/25 | 19/25 | 16/0 | Strong positive: current CAA ignores stale distractors and restores executable current-bid behavior on the same minimal prompt. |
| `structured` | 0/25 | 10/25 | 10/0 | Meaningful positive under a held-out nonnumeric id schema, though frozen valid-id remains far below positive prompting. |

Important code-correctness note: the first stepped `structured` run allowed invalid shown ids such as original numeric bids to pass through to BrowserGym and accidentally execute. This was fixed with `executable_action_from_shown`, which maps invalid shown ids to `__invalid_interface_id__` before environment stepping. Use only `*_structured_guarded.jsonl` for stepped structured claims.

Decision: the confirmed CAA direction has a real zero-shot interface-grounding effect beyond original numeric ids, especially for stale distractors and some structured ids. It is not a general interface adapter: it fails `handle`, weakly handles `longprefix`, regresses `uuid` frozen valid-id, and remains weaker than explicit positive prompting on most held-out schemas. Next high-information test is to build source-mode interface-general vectors and require gains on `handle`/`longprefix`/`uuid`, not to rerun current CAA on broader slices.

### Multi-Dataset Action-Grounding Smoke, May 2026

Setup state on `cvpc`:

- MiniWob++ server is live at `MINIWOB_URL=http://localhost:8080/miniwob/`.
- WebArena BrowserGym package was installed but no backend URLs were configured. Downloaded the official WebArena `shopping_admin_final_0719.tar` from the CMU mirror with a resumable range downloader, loaded Docker image `shopping_admin_final_0719:latest`, and started `shopping_admin` on `http://localhost:7780/admin`. BrowserGym reset/`noop()` preflight passes for WebArena task `0`.
- Only the WebArena shopping-admin backend is configured. Map, reddit, gitlab, wiki, and shopping frontend are still placeholders and should not be used for experiments until their backends are installed.
- WorkArena package is installed, but benchmark reset is blocked without ServiceNow credentials or access to gated Hugging Face dataset `ServiceNow/WorkArena-Instances`. Online fetch returns `401 GatedRepoError`; no `SNOW_INSTANCE_URL`, `SNOW_INSTANCE_UNAME`, `SNOW_INSTANCE_PWD`, `SNOW_INSTANCE_POOL`, or Hugging Face token was available on `cvpc`. Do not run WorkArena experiments until a real ServiceNow instance is provided and `workarena-install --instance-url ... --instance-password ...` has completed.

New generic action-space CAA prompt:

```text
Read the current state and the listed action space carefully. Choose exactly one allowed action. Fill every action argument using only identifiers, handles, or values present in the current state. Output only that executable action, with no explanation.
```

Targeted MiniWob 25-slice result at Gemma-3-4B L17 was weak. Average L17 residual norm on the vector-construction slice was `31930.5`.

| Vector | Alpha | Alpha / avg L17 norm | Targeted 25 success | Notes |
| --- | ---: | ---: | ---: | --- |
| `action_space_pos_minus_base` | `500,750,1000,1500` | `1.6%,2.3%,3.1%,4.7%` | `1/25` each | far below confirmed bid CAA `19/25` |
| `action_space_pos_minus_base` | `3193` | `10%` | `2/25` | best generic setting, still weak |
| `action_space_pos_minus_base` | `6386,12772` | `20%,40%` | `0/25` | high normalized strengths collapse action validity |
| `action_space_pos_minus_neg` | `500,750,1000,1500,1597,3193,6386,12772` | `1.6%-40%` | best `1/25` | destructive/unchecked negative did not help |

Decision: generic action-space CAA is not competitive with the original MiniWob bid-grounding CAA. Do not promote it as the main vector.

WebArena shopping-admin smoke used tasks `0,1,2`, four environment steps, Gemma-3-4B, and the live local CMS. Baseline copied the Action Space example `fill('b534', 'Montre', True)` on all tasks, yielding `0/3` success, `100%` parse/error episodes, `0%` valid-current-id, and `100%` invalid/bogus argument.

WebArena steering/prompt variants:

| Condition | Result | Observation |
| --- | ---: | --- |
| baseline with action examples | `0/3` | repeated invalid `fill('b534', ...)` copied from action examples |
| `action_space_pos_minus_base`, alpha `3193` | `0/3` | changed example id to invalid `48` with examples; with examples removed, returned to `b534` |
| confirmed `gemma_tree_pos_minus_base`, alpha `1000` | `0/3` | mostly unchanged; one trajectory briefly emitted `click(bid='723')` then fell back to `b534` |
| prompt guard only | `0/3` | explicitly saying examples are fake did not break `b534` |
| prompt guard plus no action examples | `0/3` | `b534` persisted, indicating a strong memorized/default-action attractor |

Interpretation: current Gemma-3-4B text-only scaffold does not show dataset-independent action grounding on WebArena shopping-admin. The MiniWob CAA effect is still best framed as AX-tree bid grounding in MiniWob-like tasks, not broad WebArena competence. The WebArena smoke is useful negative evidence: prompt examples and/or WebArena task priors induce a stable invalid fill action that neither the generic action-space CAA nor the MiniWob bid CAA fixes.

### Prompt-only

| Model | Prompt Pair | Slice | Baseline | Positive | Negative | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Gemma-3-4B | `gemma_tree_vs_label` | 5 tasks / 100 matched BrowserGym-seed episodes | 45/100 | 70/100 | 0/100 | current best prompt-only result; positive has 25 paired gains, 0 paired losses; parse-fail 42% -> 7%; negative parse-fail 100% |

`gemma_tree_vs_label` targets AX-tree bid grounding and executable action discipline. The positive prompt repairs a real failure mode, but the negative prompt is too destructive for clean CAA because it reached `100%` parse-fail.

Positive prompt:

```text
Use the current Accessibility Tree as the source of truth. The bid is the bracketed id before each element. Choose the element matching the goal and output only one action using that exact bid.
```

Negative prompt:

```text
Misread bids. Use the visible label, color, or an Action Space example id as the bid instead of the bracketed number from the current Accessibility Tree.
```

Trajectory interpretation: the positive prompt mainly repairs bid grounding and executable action discipline. Baseline often emits visible labels or bogus bids such as `tt`; positive reduces this failure sharply. This is meaningful, but should be framed as a specific web-agent control variable rather than general reasoning.

`gemma_tree_vs_guess` is the current cleaner CAA candidate. It keeps the same positive prompt and replaces the hard negative with a softer unchecked-action negative:

Soft negative prompt:

```text
Output exactly one executable action, but do not verify the current Accessibility Tree bid. Guess a plausible bid from nearby elements, examples, or earlier actions even if it may target the wrong element.
```

### CAA

| Model | Prompt Pair | Vector Samples | Layer / Alpha | Slice | Baseline | Steered | Status |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| Gemma-3-4B | `gemma_tree_pos_minus_base` | 25 | L17 / alpha `250,500,750,1000` | 25 positive-gain targeted seeds | 0/25 | 10/25, 15/25, 19/25, 19/25 | clear alpha knob; negative alpha `-500` was 0/25; random vector alpha `750` was 0/25 |
| Gemma-3-4B | `gemma_tree_pos_minus_base` | 25 | L17 / alpha `1000` | 5-task seed-42 100 episodes | 45/100 | 65/100 | confirmed full-slice improvement; parse-fail 42% -> 11%; paired gains/losses 20/0 |
| Gemma-3-4B | `gemma_tree_pos_minus_base` | 25 | L17 / alpha `1000` | 5-task seed-43 100 episodes | 43/100 | 64/100 | fresh seed-stream transfer; parse-fail 45% -> 11%; paired gains/losses 21/0 |
| Gemma-3-4B | random vector | n/a | L17 / alpha `1000` | 5-task seed-42 100 episodes | 45/100 | 44/100 | full-slice negative control; no improvement |
| Gemma-3-4B | `gemma_tree_pos_minus_base` reversed | 25 | L17 / alpha `-1000` | 5-task seed-42 100 episodes | 45/100 | 28/100 | sign control; harms performance |
| Gemma-3-4B | `action_syntax` | 25 | L17 / alpha `1000` | 5-task seed-42 100 episodes | 45/100 | 38/100 | syntax-only control; does not reproduce successful CAA |

The effective vector is positive-minus-baseline, not positive-minus-destructive-negative. Earlier low-alpha screens were not informative because the useful scale is much larger, roughly alpha `750-1000` at L17.

Per-task pattern for L17 alpha `1000`:

| Slice | Task | Baseline | Steered | Parse Fail Base -> Steered | Gains / Losses |
| --- | --- | ---: | ---: | ---: | ---: |
| seed42 | click-button | 20/20 | 20/20 | 0 -> 0 | 0 / 0 |
| seed42 | click-link | 18/20 | 19/20 | 2 -> 0 | 1 / 0 |
| seed42 | click-option | 1/20 | 2/20 | 18 -> 10 | 1 / 0 |
| seed42 | choose-list | 4/20 | 4/20 | 5 -> 1 | 0 / 0 |
| seed42 | focus-text | 2/20 | 20/20 | 17 -> 0 | 18 / 0 |
| seed43 | click-button | 18/20 | 18/20 | 0 -> 0 | 0 / 0 |
| seed43 | click-link | 19/20 | 20/20 | 7 -> 0 | 1 / 0 |
| seed43 | click-option | 1/20 | 2/20 | 18 -> 9 | 1 / 0 |
| seed43 | choose-list | 5/20 | 5/20 | 1 -> 1 | 0 / 0 |
| seed43 | focus-text | 0/20 | 19/20 | 19 -> 1 | 19 / 0 |

Mechanistic read: the vector mainly pushes outputs from label/string/bogus-bid actions such as `click('viverra')` or `fill('tt', ...)` toward numeric AX-tree bid actions such as `click('14')` or `fill('12', ...)`. It strongly fixes `focus-text`, mildly helps click bid selection, and does not yet solve low baseline competence on `choose-list` or `click-option`.

Historical Qwen CAA runs exist but are stale relative to the current Gemma path and should not drive the next search.

### SAE / Transcoder Readout

Readout used the confirmed normalized CAA vector `vectors/gemma-3-4b/seed_0/gemma_tree_pos_minus_base_L17.pt` and Gemma Scope 2 Gemma-3-4B-IT layer-17 width-16k L0-medium dictionaries.

| Dictionary | Scoring | Top Feature | Score | Neuronpedia Label | Interpretation |
| --- | --- | ---: | ---: | --- | --- |
| Residual SAE | encoder sensitivity to adding the vector | 1246 | 0.0614 | `UI elements and styling` | Best semantic match for the causal intervention; top tokens include `buttons`, `decorative`, `button`, `buttons`, `popup`. |
| Residual SAE | decoder cosine | 510 | 0.7652 | unlabeled | Raw top decoder feature, not interpretable enough for the main claim. |
| Residual SAE | decoder cosine | 200 | 0.7537 | `numbers and quantities` | Interpretable bid-number component; related feature 62 has the same label and top tokens `0,7,6,9,5,8,4,3`. |
| Transcoder | decoder cosine | 223 | 0.7746 | no available Neuronpedia transcoder label | Scoreable locally, but not label-supported on Neuronpedia for this dashboard. |

Current semantic claim: the vector primarily activates a UI/action-context latent and aligns with numeric bid latents. This fits the observed behavioral effect: fewer invalid string-label actions and more correct numeric bid actions. Do not claim a named transcoder mechanism until labeled Gemma-3-4B transcoder dashboards are available or labels are generated.

Direct feature-steering results:

| Vector | Definition | Targeted 25 result | Status |
| --- | --- | ---: | --- |
| `sae1246_dec` | residual SAE feature 1246 decoder/write direction | 0/25 at alpha `250,500,750,1000` | no behavioral sufficiency |
| `sae1246_enc` | residual SAE feature 1246 encoder/sensitivity direction | 0/25 at alpha `250,500,750,1000` | no behavioral sufficiency |
| `sae1246_200_62_dec` | normalized decoder-direction sum of UI feature 1246 and numeric features 200/62 | 0/25 at alpha `250,500,750,1000` | no behavioral sufficiency |
| `sae1246_200_62_enc` | normalized encoder-direction sum of UI feature 1246 and numeric features 200/62 | 2/25 at `250`, 2/25 at `500`, 0/25 at `750,1000`; fine sweep: 0/25 at `100,150,200`, 7/25 at `300`, 12/25 at `400`, 4/25 at `600` | weak but real partial sufficiency |

Best direct SAE feature setting so far is `sae1246_200_62_enc`, L17, alpha `400`. It transfers to full seed streams but remains weaker than raw CAA:

| Method | Slice | Baseline | Steered | Parse Fail Base -> Steered | Gains / Losses |
| --- | --- | ---: | ---: | ---: | ---: |
| CAA `gemma_tree_pos_minus_base`, alpha `1000` | seed42 100 | 45/100 | 65/100 | 42 -> 11 | 20 / 0 |
| SAE `sae1246_200_62_enc`, alpha `400` | seed42 100 | 45/100 | 59/100 | 42 -> 22 | 14 / 0 |
| CAA `gemma_tree_pos_minus_base`, alpha `1000` | seed43 100 | 43/100 | 64/100 | 45 -> 11 | 21 / 0 |
| SAE `sae1246_200_62_enc`, alpha `400` | seed43 100 | 43/100 | 56/100 | 45 -> 26 | 13 / 0 |

Per-task pattern for SAE `sae1246_200_62_enc`, L17 alpha `400`:

| Slice | Task | Baseline | Steered | Parse Fail Base -> Steered | Gains / Losses |
| --- | --- | ---: | ---: | ---: | ---: |
| seed42 | click-button | 20/20 | 20/20 | 0 -> 0 | 0 / 0 |
| seed42 | click-link | 18/20 | 19/20 | 2 -> 0 | 1 / 0 |
| seed42 | click-option | 1/20 | 3/20 | 18 -> 11 | 2 / 0 |
| seed42 | choose-list | 4/20 | 4/20 | 5 -> 4 | 0 / 0 |
| seed42 | focus-text | 2/20 | 13/20 | 17 -> 7 | 11 / 0 |
| seed43 | click-button | 18/20 | 18/20 | 0 -> 0 | 0 / 0 |
| seed43 | click-link | 19/20 | 20/20 | 7 -> 3 | 1 / 0 |
| seed43 | click-option | 1/20 | 2/20 | 18 -> 11 | 1 / 0 |
| seed43 | choose-list | 5/20 | 5/20 | 1 -> 3 | 0 / 0 |
| seed43 | focus-text | 0/20 | 11/20 | 19 -> 9 | 11 / 0 |

Projection/residual mediation tests for the selected SAE subspace `{1246,200,62}` are now decisive against a localized mediation claim:

| Vector | Slice | Alpha(s) | Result | Interpretation |
| --- | --- | ---: | ---: | --- |
| CAA projected onto SAE encoder subspace | target25 | `250,400,750` | 2/25, 12/25, 0/25 | partial sufficiency only; high alpha collapses |
| CAA residual after removing SAE encoder subspace | target25 | `250,400,750,1000` | 11/25, 14/25, 18/25, 19/25 | most of the CAA effect remains outside the selected SAE subspace |
| CAA residual after removing SAE encoder subspace | seed42 100 | `1000` | 45/100 -> 64/100; parse-fail 42 -> 13; gains/losses 19/0 | essentially matches raw CAA 65/100 |

Frozen one-step grounding on target25 shows the same pattern:

| Condition | Valid current id | Bogus bid |
| --- | ---: | ---: |
| baseline | 0/25 | 19/25 |
| raw CAA alpha `1000` | 19/25 | 1/25 |
| positive prompt | 24/25 | 1/25 |
| SAE `sae1246_200_62_enc` alpha `400` | 13/25 | 8/25 |
| CAA projection onto SAE subspace alpha `400` | 13/25 | 8/25 |
| CAA residual alpha `1000` | 18/25 | 1/25 |

ID-remapping falsification on target25 further limits the claim:

| Condition | Original IDs | Alphanumeric `x00` IDs | Structured `node-000` IDs |
| --- | ---: | ---: | ---: |
| baseline | 0/25 | 11/25 | 0/25 |
| raw CAA alpha `1000` | 19/25 | 19/25 | 6/25 |
| positive prompt | not rerun; target slice was selected from positive-gain cases | 22/25 | 20/25 |

Interpretation: the selected SAE UI+numeric encoder direction is partially sufficient for action grounding, but it is not the main mediator of the successful CAA direction. The residual CAA retains almost all of the behavioral effect after removing the selected SAE subspace. The current practical claim is also weaker than prompting: a direct positive prompt beats raw CAA on remapped IDs and frozen valid-ID metrics. Do not claim that steering these interpretable latents improves understanding. The defensible claim is narrower: these latents are causally helpful probes/interventions for MiniWob bid grounding, while the main CAA mechanism remains distributed or outside the selected SAE features.

## limitations

- The current Gemma positive prompt is scaffold-aware. That is acceptable for MiniWob text-only, but claims should be about AX-tree bid grounding unless adjacent-task transfer supports broader behavior.
- The original `gemma_tree_vs_label` negative prompt is useful as a hard contrast but too broad for a clean vector. The successful vector is `gemma_tree_pos_minus_base`.
- Targeted slices are useful for mechanism checks but do not estimate full MiniWob++ accuracy.
- Existing older CAA retained results are summary-level or from a different action-execution assumption. Treat them as historical until rerun.
- Response-based vector extraction is non-standard CAA; prompt-based extraction is available via `--vector-method prompt`.
- Parse-fail reduction can masquerade as task competence. Always report both success and parse-fail deltas.
- Current effect is dominated by action validity and bid grounding, especially `focus-text`. Do not overclaim general planning or web-agent reasoning.

## next steps

### TALES Migration Pilot, 2026-05-04

Setup completed on `cvpc` without changing the MiniWob steering environment:

- Created isolated Python 3.12 venv at `/mnt/code/tales312` for `tale-suite` state collection.
- Added `scripts/collect_tales_samples.py` to collect static TALES states with `admissible_commands=True`.
- Added `scripts/run_tales_frozen.py` for frozen one-step admissible-command evaluation using the existing Gemma steering stack.
- Added `scripts/summarize_tales.py` for TALES action-interface metrics.
- Collected `results/tales_textworld_250_samples.jsonl`: 160 `TWCookingLevel*` states, no skipped envs after supporting TALES' 4-value Gym step API.

Initial diagnostic:

- Clean current-admissible list was too easy. On a 10-state smoke, baseline/positive/steer/random/reverse all reached `10/10` exact admissible command.
- Switched to `--candidate-mode mixed`: current admissible commands are mixed with stale commands and decoys from other TALES states; metric still counts only true current admissible commands.

Main frozen 100-state mixed-command result, Gemma-3-4B, L17, alpha 1000, 40 vector states:

| condition | exact admissible | action-type valid | copied decoy | copied stale | invented |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.77 | 0.86 | 0.20 | 0.01 | 0.02 |
| positive prompt | 0.78 | 0.85 | 0.19 | 0.00 | 0.03 |
| CAA `pos_minus_base` | 0.79 | 0.87 | 0.19 | 0.00 | 0.02 |
| random vector | 0.77 | 0.86 | 0.21 | 0.00 | 0.02 |
| reverse vector | 0.77 | 0.86 | 0.20 | 0.01 | 0.02 |

Alpha sweep for `pos_minus_base`, L17:

- alpha 500: exact `0.78`
- alpha 1000: exact `0.79`
- alpha 1500: exact `0.79`
- alpha 2500: exact `0.79`
- alpha 5000: exact `0.78`, action-type validity dropped to `0.82`

Alternate contrast `tales_admissible_pos_minus_neg`, L17 alpha 1000:

- baseline `0.77`
- positive prompt `0.78`
- CAA `0.78`
- random `0.77`
- reverse `0.77`

Interpretation: the TALES pipeline is working, but basic CAA does not yet provide a meaningful effect. The best observed improvement is only `+2 exact-admissible points` over baseline and `+1` over positive prompting on the 100-state mixed-command diagnostic. This is not enough to claim MI-guided agent performance improvement. Next TALES work should pivot to MI-guided feature/layer selection or a harder/lower-baseline subset; do not expand this basic CAA setting to full TALES reward runs.

### Immediate Gemma-3-4B Queue

1. Preserve the confirmed CAA and SAE result artifacts. Do not recompute vectors unless intentionally changing the vector definition.
2. Do not spend more GPU time on direct SAE alpha search for `{1246,200,62}`. Projection/residual tests show these features are not the main mediator.
3. If pursuing interpretability, search for additional features explaining the residual CAA direction, then rerun projection/residual mediation. Require residual weakening before making a mechanism claim.
4. If pursuing practical utility, treat positive prompting and prompt+repair as the baseline to beat. Current CAA and SAE steering do not beat positive prompting on remapped IDs.
5. Measure average L17 residual norm and report raw alphas as `alpha / avg_l17_norm`; do not use this as a substitute for behavioral controls.
6. Only construct synthetic dataset-independent AX-grounding vectors if the next feature search or a new vector survives structured ID remapping.

Runnable remote prefix:

```bash
cd /mnt/code/Reaz/WebAgent_Steering
export HF_HOME=/mnt/code/huggingface
export MINIWOB_URL=http://localhost:8080/miniwob/
export BROWSERGYM_CHROMIUM_EXECUTABLE=/usr/bin/chromium-browser
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

Confirmed seed-42 run:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/code/steer/bin/python scripts/run_sweep.py \
  --model gemma-3-4b \
  --prompt-type gemma_tree_pos_minus_base \
  --vector-method prompt \
  --plan-file /tmp/gemma_100_plan.txt \
  --episode-steps 6 \
  --layers 17 \
  --alphas=1000 \
  --steer-position last \
  --steer-only \
  --base-jsonl results/gemma3_4b_100_baseline_from_prompt_json.jsonl \
  --train-steps 25 \
  --cache-dir vectors \
  --seed 0 \
  --out-dir results/gemma3_4b_pos_minus_base_100_L17_a1000
```

Transfer seed-43 run:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/code/steer/bin/python scripts/run_sweep.py \
  --model gemma-3-4b \
  --prompt-type gemma_tree_pos_minus_base \
  --vector-method prompt \
  --tasks click-button,click-link,click-option,choose-list,focus-text \
  --plan-file /tmp/gemma_transfer_seed43_100_plan.txt \
  --episode-steps 6 \
  --layers 17 \
  --alphas=1000 \
  --steer-position last \
  --train-steps 25 \
  --cache-dir vectors \
  --seed 0 \
  --out-dir results/gemma3_4b_pos_minus_base_transfer_seed43_L17_a1000
```

## references

- Turner et al. 2023, Activation Addition: Steering Language Models Without Optimization.
- Rimsky et al. 2024, Steering Llama 2 via Contrastive Activation Addition.
- Shi et al. 2017, World of Bits / MiniWob++.
- Deng et al. 2023, Mind2Web.
- Zhou et al. 2024, WebArena.
