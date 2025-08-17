import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Configuration & Model Loading ---

# Set a device to run the models on (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the test dataset
try:
    df_test = pd.read_csv("test_data.csv")
except FileNotFoundError:
    print("Error: 'test_data.csv' not found. Please run SiameseTestDataCreation.py first.")
    exit()

# Load both the base model and fine-tuned model
try:
    base_model_name = "sentence-transformers/all-mpnet-base-v2"
    fine_tuned_model_path = "fine_tuned_Siamese_mpnet"
    
    base_model = SentenceTransformer(base_model_name, device=device)
    fine_tuned_model = SentenceTransformer(fine_tuned_model_path, device=device)
except Exception as e:
    print(f"Error loading a model. Ensure paths are correct and model files are intact.")
    print(e)
    exit()

# --- Similarity Computation ---

def compute_similarity(model, text_pairs):
    """Encodes text pairs in a batch and computes cosine similarity."""
    embeddings1 = model.encode([pair[0] for pair in text_pairs], convert_to_tensor=True, device=device)
    embeddings2 = model.encode([pair[1] for pair in text_pairs], convert_to_tensor=True, device=device)
    
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return [cosine_scores[i][i].item() for i in range(len(text_pairs))]

test_texts = list(zip(df_test["X1"].astype(str), df_test["X2"].astype(str)))
true_labels = df_test["Label"].tolist()

# Get similarity scores from both models
base_model_scores = compute_similarity(base_model, test_texts)
fine_tuned_scores = compute_similarity(fine_tuned_model, test_texts)

# --- Quantitative Evaluation ---

# a threshold to convert scores into binary predictions
THRESHOLD = 0.5

# Get binary predictions
base_predictions = [1 if score > THRESHOLD else 0 for score in base_model_scores]
fine_tuned_predictions = [1 if score > THRESHOLD else 0 for score in fine_tuned_scores]

# --- Calculate Metrics ---

# Base Model Metrics
accuracy_base = accuracy_score(true_labels, base_predictions)
precision_base, recall_base, f1_base, _ = precision_recall_fscore_support(true_labels, base_predictions, average='binary')

# Fine-Tuned Model Metrics
accuracy_ft = accuracy_score(true_labels, fine_tuned_predictions)
precision_ft, recall_ft, f1_ft, _ = precision_recall_fscore_support(true_labels, fine_tuned_predictions, average='binary')


# --- Print Comparison ---

print("\n" + "="*60)
print(f"{'Model Performance Comparison (Threshold = 0.5)':^60}")
print("="*60)
print(f"{'Metric':<12} | {'Base Model':^20} | {'Fine-Tuned Model':^22}")
print("-"*60)
print(f"{'Accuracy':<12} | {accuracy_base:^20.4f} | {accuracy_ft:^22.4f}")
print(f"{'Precision':<12} | {precision_base:^20.4f} | {precision_ft:^22.4f}")
print(f"{'Recall':<12} | {recall_base:^20.4f} | {recall_ft:^22.4f}")
print(f"{'F1-Score':<12} | {f1_base:^20.4f} | {f1_ft:^22.4f}")
print("="*60)
