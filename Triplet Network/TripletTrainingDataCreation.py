import re
import os
import csv
import itertools
import random

MAX_TRIPLETS = 180  # Limit number of triplets per file because it was creating GBs of data

def parse_text_file(file_path):
    """Extracts discussion title, all Pro arguments, and all Con arguments."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if not lines:
        return None, [], []  

    discussion_title = lines[0].strip().replace("Discussion Title: ", "")
    pro_arguments = []
    con_arguments = []

    for line in lines[1:]:  
        line = line.strip()
        match = re.match(r"(\d+(\.\d+)*)\.\s+(Pro|Con):\s*(.*)", line)  # Match Pro/Con arguments
        if match:
            _, _, stance, argument = match.groups()
            argument = argument.strip()
            if is_reference_sentence(argument):  # Ignore references like "-> See 1.3.2."
                continue

            if stance == "Pro":
                pro_arguments.append(argument)
            elif stance == "Con":
                con_arguments.append(argument)

    return discussion_title, pro_arguments, con_arguments

def clean_text(text):
    """Remove encoding artifacts and unwanted characters."""
    if text:
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # Remove markdown links like [text](url)
        text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
        return text.strip()
    return None

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
    """Processes all .txt files and writes up to 180 triplets per file to CSV."""
    
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Anchor", "Pro", "Con"]) 
        
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                discussion_title, pro_arguments, con_arguments = parse_text_file(file_path)

                if discussion_title and pro_arguments and con_arguments:
                    # Generate all possible Pro-Con pairs
                    triplets = list(itertools.product(pro_arguments, con_arguments))

                    # Randomly select up to 180 triplets 
                    if len(triplets) > MAX_TRIPLETS:
                        triplets = random.sample(triplets, MAX_TRIPLETS)

                    for pro, con in triplets:
                        cleaned_triplet = [
                            clean_text(discussion_title),
                            clean_text(pro),
                            clean_text(con),
                        ]
                        writer.writerow(cleaned_triplet)

    print(f"Triplet training data saved to {output_csv}")

folder_path = "train"  
output_csv = "TripletTraining.csv"
process_folder(folder_path, output_csv)
