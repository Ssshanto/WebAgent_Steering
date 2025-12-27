from datasets import load_dataset

def debug_schema():
    print("Loading Mind2Web dataset (train split)...")
    dataset = load_dataset("osunlp/Mind2Web", split="train", streaming=True)
    
    print("Fetching first example...")
    for i, sample in enumerate(dataset):
        if len(sample['actions']) > 0:
            action = sample['actions'][0]
            print("\nOperation field:")
            print(action['operation'])
            print("\nPos Candidates field:")
            print(action['pos_candidates'])
        break

if __name__ == "__main__":
    debug_schema()