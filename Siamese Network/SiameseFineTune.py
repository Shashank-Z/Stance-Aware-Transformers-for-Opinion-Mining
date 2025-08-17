import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, datasets
from torch.utils.data import DataLoader
import os
os.environ["WANDB_DISABLED"] = "true"

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load dataset
df = pd.read_csv("SiameseTraining.csv")
df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
df = df.dropna(subset=["Label"])  # Drop rows where 'Label' is NaN
df["Label"] = df["Label"].astype(int)  # Convert 'Label' back to integer (0 or 1)


# Prepare dataset for training
train_examples = [
    InputExample(texts=[str(row["X1"]), str(row["X2"])], label=float(row["Label"]))
    for _, row in df.iterrows()
]

# Load pre-trained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

# Define loss function and DataLoader
train_dataloader = DataLoader(train_examples, batch_size=8, shuffle=True)
train_loss = losses.CosineSimilarityLoss(model)

# Train model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,  
    warmup_steps=50,  
    output_path="fine_tuned_Siamese_mpnet",
    use_amp=torch.cuda.is_available(),  # Enable AMP only if CUDA is available
)

# Save model
model.save("fine_tuned_Siamese_mpnet")
print("Fine-tuning complete! Model saved as 'fine_tuned_Siamese_mpnet'.")