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
fine_tuned_model = SentenceTransformer("fine_tuned_triplet_mpnet", device=device)

# Function to compute cosine similarity
def compute_similarity(model, text1, text2):
    embedding1 = model.encode(text1, convert_to_tensor=True).to(device)
    embedding2 = model.encode(text2, convert_to_tensor=True).to(device)
    return util.pytorch_cos_sim(embedding1, embedding2).item()

# Lists to store results
similarities_base_pro = []
similarities_base_con = []
similarities_fine_tuned_pro = []
similarities_fine_tuned_con = []


# Compute cosine similarities for triplets
for _, row in df_test.iterrows():
    anchor, pro, con = row["Anchor"], row["Pro"], row["Con"]
    
    # Base Model Similarities
    sim_base_pro = compute_similarity(base_model, anchor, pro)
    sim_base_con = compute_similarity(base_model, anchor, con)

    # Fine-Tuned Model Similarities
    sim_fine_tuned_pro = compute_similarity(fine_tuned_model, anchor, pro)
    sim_fine_tuned_con = compute_similarity(fine_tuned_model, anchor, con)
    
    # Store results
    similarities_base_pro.append(sim_base_pro)
    similarities_base_con.append(sim_base_con)
    similarities_fine_tuned_pro.append(sim_fine_tuned_pro)
    similarities_fine_tuned_con.append(sim_fine_tuned_con)

# Convert results to DataFrame
results_df = pd.DataFrame({
    "Base Pro": similarities_base_pro,
    "Base Con": similarities_base_con,
    "Fine-Tuned Pro": similarities_fine_tuned_pro,
    "Fine-Tuned Con": similarities_fine_tuned_con
})

# Plot KDE of cosine similarities for (Anchor, Pro) and (Anchor, Con)
plt.figure(figsize=(12, 6))
sns.kdeplot(data=results_df, x="Base Pro", label="Base Model (Anchor, Pro)", linestyle="--")
sns.kdeplot(data=results_df, x="Base Con", label="Base Model (Anchor, Con)", linestyle="--")
sns.kdeplot(data=results_df, x="Fine-Tuned Pro", label="Fine-Tuned Model (Anchor, Pro)", linewidth=2)
sns.kdeplot(data=results_df, x="Fine-Tuned Con", label="Fine-Tuned Model (Anchor, Con)", linewidth=2)

plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.title("Triplet Cosine Similarity Distribution: Base vs. Fine-Tuned Model")
plt.legend()
plt.show()
