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

def create_triplets(split):
    """Generate anchor, pro, and con triplets from a dataset split."""
    triplets = []
    for example in dataset[split]:  
        if example["type"] != "binary":  # Skip "multi" type data
            continue

        anchor = clean_text(example["question"])
        perspectives = [clean_text(p) for p in example["perspectives"]]

        if len(perspectives) == 2:
            # Question as anchor, first perspective as pro, second as con
            triplets.append([anchor, perspectives[0], perspectives[1]])
    
    return triplets

def save_to_csv(triplets, filename):
    """Save triplets to CSV file."""
    with open(filename, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Anchor", "Pro", "Con"])  
        writer.writerows(triplets)

# Process test and validation sets 
test_triplets = create_triplets("test")
val_triplets = create_triplets("validation")

# Save to CSV
save_to_csv(test_triplets, "test_data.csv")
save_to_csv(val_triplets, "val_data.csv")

print("Test and validation datasets saved as 'test_data.csv' and 'val_data.csv'.")
