# BrowserGym Migration Summary

**Date**: 2025-02-02  
**Status**: ✅ Complete

## Overview

Successfully migrated the WebAgent Steering research codebase from direct MiniWob++ usage to the **BrowserGym** framework.

## What Changed

### 1. Dependencies
- ❌ Removed: `miniwob`, `selenium`
- ✅ Added: `browsergym-miniwob`, `beautifulsoup4`, `lxml`
- ✅ Browser: Playwright (via `playwright install chromium`)

### 2. Environment Creation
```python
# Before
import miniwob
gym.register_envs(miniwob)
env = gym.make(f"miniwob/{task}-v1")

# After
import browsergym.miniwob
env = gym.make(f"browsergym/miniwob.{task}")
```

### 3. Observations
```python
# Before
obs["utterance"]      # Task instruction
obs["dom_elements"]   # List of DOM elements with ref attributes

# After
obs["goal"]           # Task instruction
obs["dom_object"]     # Rich DOM snapshot (needs parsing)
obs["screenshot"]     # Direct numpy array (no need for custom capture)
obs["last_action_error"]  # Built-in error feedback
```

### 4. Actions
```python
# Before
from miniwob.action import ActionTypes
act = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=5)

# After
action_str = 'click("5")'  # String-based high-level actions
env.step(action_str)
```

### 5. Element IDs
- Changed from `ref` to `bid` throughout:
  - Prompts: `click bid=5` instead of `click ref=5`
  - Action parsing: regex updated to match `bid=`
  - DOM extraction: looks for `bid` attributes

### 6. DOM Processing
```python
# Before
def dom_to_html(dom_elements):
    # Custom HTML generation from element list
    
# After
from browsergym.utils.obs import flatten_dom_to_str
dom_str = flatten_dom_to_str(obs["dom_object"])
# Parse with BeautifulSoup to extract bid attributes
```

### 7. Removed Code
- Selenium WebDriver monkeypatch (no longer needed)
- Custom screenshot capture function
- ActionTypes enum usage

## What Stayed the Same

✅ **Research Core Preserved**:
- Steering vector computation logic
- Contrastive Activation Addition (CAA) methodology
- Model architectures and layer selection
- All steering prompt configurations
- Experimental parameters and defaults

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browser
playwright install chromium

# Setup MiniWob++ server
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git
cd miniwob-plusplus
npm install && npm run build
npm run serve  # http://localhost:8080

# Set environment variable
export MINIWOB_URL="http://localhost:8080"
```

## Running Experiments

```bash
# Basic run
python src/miniwob_steer.py

# With custom parameters
python src/miniwob_steer.py \
    --model 0.5b \
    --layer 14 \
    --coeff 4.0 \
    --prompt-type accuracy \
    --train-steps 200 \
    --eval-steps 400 \
    --tasks all \
    --out results.jsonl
```

## Benefits

1. **More Stable**: Playwright is more reliable than Selenium
2. **Better Error Handling**: Built-in action error feedback in observations
3. **Richer Data**: Access to accessibility tree and structured DOM
4. **Extensible**: Easy to add WebArena, WorkArena, VisualWebArena tasks
5. **Actively Maintained**: BrowserGym has active development

## Testing Checklist

To verify the migration works:

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Install Playwright: `playwright install chromium`
- [ ] Setup MiniWob++ server
- [ ] Set `MINIWOB_URL` environment variable
- [ ] Run syntax check: `python -m py_compile src/miniwob_steer.py` ✅
- [ ] Run short test: `python src/miniwob_steer.py --train-steps 10 --eval-steps 10 --tasks click-test`
- [ ] Verify results.jsonl output format
- [ ] Compare baseline accuracy with previous results

## Known Limitations

### VLM Set-of-Marks (SoM)
The current VLM implementation with Set-of-Marks annotation has a **limitation**:

BrowserGym's DOM object doesn't include bounding box coordinates by default. The `extract_element_positions_from_dom()` function currently returns placeholder positions.

**Solutions**:
1. Use Playwright's `page.locator().bounding_box()` to get real positions
2. Use BrowserGym's accessibility tree which has partial position info
3. Disable SoM and use text-only prompts for VLM

This doesn't affect text-based LLM experiments (the primary research focus).

## Rollback Plan

If issues arise, the old code is available in git history:
```bash
git log --oneline --all  # Find commit before migration
git checkout <commit-hash> src/miniwob_steer.py requirements.txt
```

## Next Steps

1. **Test with Real Data**: Run a full experiment to verify results match expectations
2. **Benchmark Comparison**: Compare steering improvements with previous results
3. **Expand Benchmarks**: Try WebArena or WorkArena tasks
4. **VLM Fix**: Implement proper bounding box extraction if VLM experiments needed

## Files Modified

- ✏️ `src/miniwob_steer.py` - Main implementation (1246 lines)
- ✏️ `requirements.txt` - Dependencies
- ✏️ `README.md` - Setup instructions and migration notes
- ✏️ `RESEARCH.md` - Migration documentation
- 📄 `MIGRATION.md` - This file

## Version Info

- **Python**: 3.13.11
- **BrowserGym**: Latest (to be installed)
- **Playwright**: Latest (to be installed)
- **Migration Date**: February 2, 2025

---

**Migration Status**: ✅ **COMPLETE**

All code changes have been applied. Ready for dependency installation and testing.
