import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
os.environ["WANDB_DISABLED"] = "true"

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load dataset 
df = pd.read_csv("TripletTraining.csv")

# Prepare dataset for training: Each row is a triplet (anchor, positive, negative)
train_examples = [
    InputExample(texts=[str(row["Anchor"]), str(row["Pro"]), str(row["Con"])])
    for _, row in df.iterrows()
]

# Load pre-trained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

# Define DataLoader and Triplet Loss
train_dataloader = DataLoader(train_examples, batch_size=8, shuffle=True)
train_loss = losses.TripletLoss(model)

# Fine-tune the model using triplet loss
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3, 
    warmup_steps=50,
    output_path="fine_tuned_triplet_mpnet",
    use_amp=torch.cuda.is_available(),
    optimizer_params={"lr": 2e-5}  
)


# Save the fine-tuned model
model.save("fine_tuned_triplet_mpnet")
print("Fine-tuning complete! Model saved as 'fine_tuned_triplet_mpnet'.")