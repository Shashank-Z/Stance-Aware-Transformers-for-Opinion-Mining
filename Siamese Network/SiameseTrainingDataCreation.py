import re
import os
import csv

def parse_text_file(file_path):
    """Parse the hierarchical structure of the text file and extract the discussion title and arguments."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if not lines:
        return None, []  

    discussion_title = lines[0].strip().replace("Discussion Title: ", "")
    arguments = []

    for line in lines[1:]:  
        line = line.strip()
        match = re.match(r"(\d+(\.\d+)*)\.\s+(Pro|Con):\s*(.*)", line)  # Match Pro/Con arguments
        if match:
            _, _, stance, argument = match.groups()
            if not is_reference_sentence(argument):  # Ignore references like "-> See 1.3.2."
                arguments.append((stance, argument.strip()))

    return discussion_title, arguments

def create_training_pairs(discussion_title, arguments):
    """Generate sentence pairs based on the given conditions."""
    pairs = []

    # Pairs with the anchor (discussion title)
    for stance, argument in arguments:
        label = 1 if stance == "Pro" else 0  # Anchor agrees with Pro (1), disagrees with Con (0)
        pairs.append([discussion_title, argument, label])

    return pairs

def clean_text(text):
    """Remove encoding artifacts and unwanted characters."""
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # Remove markdown links like [text](url)
    text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    return text.strip()

def is_reference_sentence(sentence):
    """
    Check if the sentence contains a reference like:
    -> See 1.1.3
    (Pro) -> See 1.1.3.4
    (Con) -> See 2.3
    """
    pattern = r"^\(?(Pro|Con)?\)?\s*-> See \d+(?:\.\d+)*\.?$"
    return bool(re.match(pattern, sentence.strip()))


def process_folder(folder_path, output_csv):
    """Process all .txt files in a folder and write training pairs to CSV."""
    
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X1", "X2", "Label"])  
        
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                discussion_title, arguments = parse_text_file(file_path)

                if discussion_title and arguments:
                    training_data = create_training_pairs(discussion_title, arguments)
                    cleaned_data = [
                        (clean_text(s1), clean_text(s2), label)
                        for s1, s2, label in training_data
                        if not is_reference_sentence(s1) and not is_reference_sentence(s2)  # Filter invalid sentences
                    ]

                    writer.writerows(cleaned_data)  # Write batch to CSV

    print(f"Training data saved to {output_csv}")

folder_path = "train"
output_csv = "SiameseTraining.csv"
process_folder(folder_path, output_csv)
