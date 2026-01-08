import json
import random
from collections import defaultdict

FILES = {
    "Reasoning Amplifier (Format Accuracy)": "results/format_accuracy.jsonl",
    "Syntax Enforcer (Composite 1)": "results/composite_1.jsonl"
}

def analyze_and_sample():
    random.seed(42)  # For reproducible samples
    
    for name, filepath in FILES.items():
        print(f"\n{'='*80}")
        print(f"ANALYSIS: {name}")
        print(f"{ '='*80}")
        
        recs = [json.loads(l) for l in open(filepath) if l.strip()]
        
        # 1. Per-Task Impact
        task_stats = defaultdict(lambda: {"base_succ": 0, "steer_succ": 0, "base_fail": 0, "steer_fail": 0, "total": 0})
        
        for r in recs:
            t = r["task"]
            task_stats[t]["total"] += 1
            task_stats[t]["base_succ"] += int(r.get("base_success", False))
            task_stats[t]["steer_succ"] += int(r.get("steer_success", False))
            task_stats[t]["base_fail"] += int(r.get("base_action") is None)
            task_stats[t]["steer_fail"] += int(r.get("steer_action") is None)

        print(f"{'Task':<20} | {'Acc Delta':<8} | {'Parse Delta':<8} | {'Details'}")
        print("-" * 80)
        
        # Sort by Accuracy Delta
        sorted_tasks = sorted(task_stats.items(), key=lambda x: (x[1]["steer_succ"] - x[1]["base_succ"]), reverse=True)
        
        for t, s in sorted_tasks:
            acc_delta = (s["steer_succ"] - s["base_succ"]) / s["total"] * 100
            parse_delta = (s["steer_fail"] - s["base_fail"]) / s["total"] * 100
            
            # Filter for significant changes only to keep it readable
            if abs(acc_delta) > 5.0 or abs(parse_delta) > 10.0:
                print(f"{t:<20} | {acc_delta:>+7.1f}% | {parse_delta:>+7.1f}% | Base: {s['base_succ']}->{s['steer_succ']} succ, {s['base_fail']}->{s['steer_fail']} parse err")

        # 2. Sample Analysis
        print("\n--- SAMPLE ANALYSIS (10 Random Episodes) ---")
        samples = random.sample(recs, 10)
        
        for i, s in enumerate(samples, 1):
            task = s["task"]
            base_out = s.get("base_output", "").replace("\n", "\\n")[:100]
            steer_out = s.get("steer_output", "").replace("\n", "\\n")[:100]
            base_ok = "OK" if s.get("base_success") else "NO"
            steer_ok = "OK" if s.get("steer_success") else "NO"
            
            # Categorize change
            if base_ok == "NO" and steer_ok == "OK":
                status = "IMPROVED"
            elif base_ok == "OK" and steer_ok == "NO":
                status = "REGRESSED"
            elif base_ok == "NO" and steer_ok == "NO":
                # Check if parse improved
                if s.get("base_action") is None and s.get("steer_action") is not None:
                    status = "FIXED_FORMAT (But Failed)"
                else:
                    status = "SAME_FAIL"
            else:
                status = "SAME_PASS"

            print(f"{i}. [{task}] {status}")
            print(f"   Base:  {base_ok} {base_out}")
            print(f"   Steer: {steer_ok} {steer_out}")

if __name__ == "__main__":
    analyze_and_sample()
