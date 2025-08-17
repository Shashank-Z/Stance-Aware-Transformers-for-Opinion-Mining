import re
import csv
from datasets import load_dataset

# Load dataset
dataset = load_dataset("timchen0618/Kialo")

def clean_text(text):
    """Cleans text by removing unwanted characters."""
    if text:
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # Remove markdown links
        text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
        return text.strip()
    return ""

def create_pairs(split):
    """Generate sentence pairs from a dataset split."""
    pairs = []
    for example in dataset[split]:  
        if example["type"] != "binary":  # Skip "multi" type data
            continue

        question = clean_text(example["question"])
        perspectives = [clean_text(p) for p in example["perspectives"]]

        if len(perspectives) == 2:
            # Question and first perspective (Agreeing - Label 1)
            pairs.append([f"{question}", f"{perspectives[0]}", 1])
            
            # Question and second perspective (Disagreeing - Label 0)
            pairs.append([f"{question}", f"{perspectives[1]}", 0])
    
    return pairs

def save_to_csv(pairs, filename):
    """Save pairs to CSV file."""
    with open(filename, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X1", "X2", "Label"]) 
        writer.writerows(pairs)

# Process test and validation sets 
test_pairs = create_pairs("test")
val_pairs = create_pairs("validation")

# Save to CSV
save_to_csv(test_pairs, "test_data.csv")
save_to_csv(val_pairs, "val_data.csv")

print("Test and validation datasets saved as 'test_data.csv' and 'val_data.csv'.")
