import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from pathlib import Path
from resnet import ResNet18  # Importing our trained ResNet model
import csv

# Sets the device - also forces gpu use
device = torch.device("cuda")

# Define paths for dataset, model checkpoint, and logs
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"  # Load the best model

# Define logs directory and test metrics file
LOGS_DIR = ROOT_DIR / "logs"
TEST_METRICS_FILE = LOGS_DIR / "test_metrics.csv"

# Create the CSV file and write headers if it does not already exist
if not TEST_METRICS_FILE.exists():
    with open(TEST_METRICS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Overall Accuracy", "Total Correct", "Cat Accuracy", "Total Cats Correct", "Dog Accuracy", "Total Dogs Correct"])  # Column headers

# Ensure the model checkpoint exists before running evaluation
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found at {CHECKPOINT_PATH}")

# Define hyperparameters
BATCH_SIZE = 32  # Keeping batch size the same as training

# Define transformations to ensure test images match the training pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize using the same mean and std as training
])

# Load the test dataset
print("Loading the test dataset - ")
test_dataset = datasets.ImageFolder(root=DATA_DIR / "test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Initialize the trained model and load saved weights
model = ResNet18().to(device)  # Load the trained ResNet model
model.load_state_dict(torch.load(CHECKPOINT_PATH))  # Load trained weights
model.eval()  # Set the model to evaluation mode (disables gradient updates)

# Function to evaluate the model on test data
def test():
    print("\nStarting Model Testing - \n")

    correct, total = 0, 0
    class_correct = [0, 0]  # [Cat correct, Dog correct]
    class_total = [0, 0]  # [Cat total, Dog total]

    with torch.no_grad():  # Disable gradient calculations to speed up inference
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)  # Get predicted class (0 for cat, 1 for dog)
            correct += predicted.eq(labels).sum().item()  # Count correct predictions
            total += labels.size(0)

            # Track per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                class_correct[label] += (predicted[i].item() == label)

    # Calculate overall and per-class accuracy
    test_accuracy = 100.0 * correct / total
    cat_accuracy = 100.0 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    dog_accuracy = 100.0 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0

    print(f"Overall Test Accuracy: {test_accuracy:.2f}% ({correct}/{total} correct)")
    print(f"Cat Accuracy: {cat_accuracy:.2f}% ({class_correct[0]}/{class_total[0]})")
    print(f"Dog Accuracy: {dog_accuracy:.2f}% ({class_correct[1]}/{class_total[1]})")

    # Append test results to the CSV file
    with open(TEST_METRICS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"{test_accuracy:.2f}%", f"({correct}/{total})", f"{cat_accuracy:.2f}%", f"({class_correct[0]}/{class_total[0]})", f"{dog_accuracy:.2f}%", f"({class_correct[1]}/{class_total[1]})"])

# Run the test function if executed directly
if __name__ == "__main__":
    print(f"Using device: {device}")
    test()
