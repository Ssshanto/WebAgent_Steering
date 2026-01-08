import argparse
import numpy as np
import torch
import gymnasium as gym
import miniwob
from src.miniwob_steer import SteeredModel, compute_vector, PROMPT_CONFIGS, POS_INSTR, NEG_INSTR, MODEL_MAP

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Monkey-patch MiniWoB to use the correct Chrome binary
from miniwob.selenium_instance import SeleniumInstance
from selenium import webdriver

def patched_create_driver(self):
    assert not hasattr(self, "driver"), f"Instance {self.index} already has a driver"
    options = webdriver.ChromeOptions()
    options.binary_location = "/usr/bin/chromium-browser"  # Fix for cvpc
    options.add_argument(f"window-size={self.window_width},{self.window_height}")
    if self.headless:
        options.add_argument("headless")
        options.add_argument("disable-gpu")
        options.add_argument("no-sandbox")
    else:
        options.add_argument("app=" + self.url)
    self.driver = webdriver.Chrome(options=options)
    self.driver.implicitly_wait(5)
    if self.headless:
        self.driver.get(self.url)
    
    from selenium.webdriver.support.wait import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    try:
        WebDriverWait(self.driver, 5).until(
            EC.element_to_be_clickable((By.ID, self.SYNC_SCREEN_ID))
        )
    except Exception as e:
        import logging
        logging.error("Page did not load properly. Wrong URL?")
        raise e
    self.inner_width, self.inner_height = self.driver.execute_script(
        "return [window.innerWidth, window.innerHeight];"
    )
SeleniumInstance.create_driver = patched_create_driver

def main():
    # Setup
    model_name = "0.5b"
    layer = 13
    tasks = ["click-test", "click-button", "click-link", "focus-text"] # Representative subset
    steps = 50 
    
    print("Loading model...")
    model = SteeredModel(MODEL_MAP[model_name], layer, coeff=1.0, vector_method="response")
    
    # 1. Compute Vector A (Reasoning: format_accuracy)
    print("\nComputing Vector A (format_accuracy)...")
    # Patch global prompts
    import src.miniwob_steer
    src.miniwob_steer.POS_INSTR = PROMPT_CONFIGS["format_accuracy"]["pos"]
    src.miniwob_steer.NEG_INSTR = PROMPT_CONFIGS["format_accuracy"]["neg"]
    
    compute_vector(model, tasks, steps, 80, 80)
    vec_a = model.vector.numpy()
    
    # 2. Compute Vector B (Syntax: composite_1)
    print("\nComputing Vector B (composite_1)...")
    src.miniwob_steer.POS_INSTR = PROMPT_CONFIGS["composite_1"]["pos"]
    src.miniwob_steer.NEG_INSTR = PROMPT_CONFIGS["composite_1"]["neg"]
    
    compute_vector(model, tasks, steps, 80, 80)
    vec_b = model.vector.numpy()
    
    # 3. Analysis
    sim = cosine_similarity(vec_a, vec_b)
    print(f"\n{'='*40}")
    print(f"VECTOR ALIGNMENT ANALYSIS (L{layer})")
    print(f"{'='*40}")
    print(f"Cosine Similarity: {sim:.4f}")
    
    # Orthogonality check
    if abs(sim) < 0.3:
        status = "ORTHOGONAL (Independent)"
    elif abs(sim) < 0.7:
        status = "PARTIALLY ALIGNED"
    else:
        status = "HIGHLY ALIGNED (Redundant)"
    print(f"Status: {status}")
    
    # 4. Combination Test
    print(f"\n{'='*40}")
    print(f"COMBINATION TEST (A + B)")
    print(f"{'='*40}")
    
    # Create combined vector
    vec_combined = vec_a + vec_b
    vec_combined = vec_combined / np.linalg.norm(vec_combined)
    
    model.set_vector(torch.tensor(vec_combined))
    model.coeff = 4.0 # Use optimal coeff
    
    # Quick eval on click-test (known sensitivity)
    env = gym.make("miniwob/click-test-v1")
    successes = 0
    total = 20
    
    print(f"Running {total} episodes on click-test with Combined Vector...")
    for _ in range(total):
        obs, _ = env.reset()
        prompt = src.miniwob_steer.build_prompt(obs, 80)
        out = model.generate(prompt, steer=True, max_new_tokens=80)
        action = src.miniwob_steer.parse_action(out)
        reward, _ = src.miniwob_steer.step_env(env, action)
        if reward > 0: successes += 1
        
    print(f"Combined Accuracy: {successes/total:.1%}")

if __name__ == "__main__":
    main()
