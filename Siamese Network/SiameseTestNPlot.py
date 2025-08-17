import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load test dataset
df_test = pd.read_csv("test_data.csv") 

# Load models and move to GPU if available
base_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
fine_tuned_model = SentenceTransformer("fine_tuned_Siamese_mpnet", device=device)

# Function to compute cosine similarity on GPU
def compute_similarity(model, text1, text2):
    embedding1 = model.encode(text1, convert_to_tensor=True).to(device)
    embedding2 = model.encode(text2, convert_to_tensor=True).to(device)
    return util.pytorch_cos_sim(embedding1, embedding2).item()

# Compute cosine similarities for both models
similarities_base = []
similarities_fine_tuned = []
labels = []

for _, row in df_test.iterrows():
    x1, x2, label = row["X1"], row["X2"], int(row["Label"])
    
    # Compute similarities on GPU
    sim_base = compute_similarity(base_model, x1, x2)
    sim_fine_tuned = compute_similarity(fine_tuned_model, x1, x2)
    
    similarities_base.append(sim_base)
    similarities_fine_tuned.append(sim_fine_tuned)
    labels.append(label)

# Convert results to DataFrame
results_df = pd.DataFrame({
    "Similarity Base": similarities_base,
    "Similarity Fine-Tuned": similarities_fine_tuned,
    "Label": labels
})

# Plot KDE of cosine similarities
plt.figure(figsize=(12, 6))
sns.kdeplot(data=results_df[results_df["Label"] == 1], x="Similarity Base", label="Base Model (Pro)", linestyle="--")
sns.kdeplot(data=results_df[results_df["Label"] == 0], x="Similarity Base", label="Base Model (Con)", linestyle="--")
sns.kdeplot(data=results_df[results_df["Label"] == 1], x="Similarity Fine-Tuned", label="Fine-Tuned Model (Pro)", linewidth=2)
sns.kdeplot(data=results_df[results_df["Label"] == 0], x="Similarity Fine-Tuned", label="Fine-Tuned Model (Con)", linewidth=2)

plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.title("Cosine Similarity Distribution: Base vs. Fine-Tuned Model")
plt.legend()
plt.show()


