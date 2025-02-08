import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from pathlib import Path
from resnet import ResNet18  # Importing the custom ResNet model
import csv
from torch.utils.tensorboard import SummaryWriter

# Sets the device - also forces gpu use
device = torch.device("cuda")

# Define paths for dataset, model checkpoints, and logs
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure checkpoint directory exists

# Define logs directory and CSV file for tracking training metrics
LOGS_DIR = ROOT_DIR / "logs"
METRICS_FILE = LOGS_DIR / "training_metrics.csv"

# Create CSV file and write headers if it does not already exist
if not METRICS_FILE.exists():
    with open(METRICS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])

# Initialize TensorBoard writer to store training logs
TENSORBOARD_LOGS_DIR = LOGS_DIR / "tensorboard"
TENSORBOARD_LOGS_DIR.mkdir(parents=True, exist_ok=True)
tensorboard_writer = SummaryWriter(log_dir=TENSORBOARD_LOGS_DIR)

# Define hyperparameters for training
BATCH_SIZE = 32  # Number of images per training batch
LR = 0.001  # Learning rate chosen based on common best practices for Adam optimizer
EPOCHS = 20  # Number of epochs to train the model

# Define preprocessing transformations for input images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale images to a standard range
])

# Load the dataset using ImageFolder, which assigns labels based on folder names
train_dataset = datasets.ImageFolder(root=DATA_DIR / "train", transform=transform)
val_dataset = datasets.ImageFolder(root=DATA_DIR / "val", transform=transform)

# Create DataLoaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Initialize the ResNet model
model = ResNet18().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss is ideal for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=LR)  # Adam optimizer for adaptive learning rate updates

# Training loop

def train():
    print("\nStarting Training...\n")
    best_val_loss = float("inf")  # Track the lowest validation loss

    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode
        train_loss, correct, total = 0, 0, 0  # Track metrics for each epoch

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Reset gradients to prevent accumulation
            outputs = model(inputs)  # Forward pass through the model
            loss = criterion(outputs, labels)  # Compute loss between predictions and actual labels

            loss.backward()  # Backpropagation to compute gradients
            optimizer.step()  # Update model parameters using computed gradients

            # Update tracking metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)  # Get class with highest probability
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Print update every 10 batches for real-time monitoring
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(train_loader)}] -> Loss: {loss.item():.4f}")

        train_accuracy = 100.0 * correct / total  # Calculate training accuracy for the epoch

        # Validation step
        val_loss, val_correct, val_total = 0, 0, 0
        model.eval()  # Set model to evaluation mode (no weight updates)
        with torch.no_grad():  # Disable gradient calculation for efficiency
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = 100.0 * val_correct / val_total  # Compute validation accuracy
        avg_train_loss = train_loss / len(train_loader)  # Compute average training loss
        avg_val_loss = val_loss / len(val_loader)  # Compute average validation loss

        print(f"Epoch [{epoch+1}/{EPOCHS}] -> Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%\n")

        # Log metrics to TensorBoard for visualization
        tensorboard_writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        tensorboard_writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
        tensorboard_writer.add_scalar("Accuracy/Train", train_accuracy, epoch + 1)
        tensorboard_writer.add_scalar("Accuracy/Validation", val_accuracy, epoch + 1)

        # Save training metrics to CSV file
        with open(METRICS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy])

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved at - {checkpoint_path}")

    tensorboard_writer.close()  # Close the TensorBoard writer after training

if __name__ == "__main__":
    print(f"Using device: {device}")
    train()
