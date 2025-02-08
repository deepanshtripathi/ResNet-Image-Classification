import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import csv
from resnet import ResNet18  # Importing the trained ResNet model

# Sets the device - also forces gpu use
device = torch.device("cuda")

# Define paths for model checkpoint and logs
ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_FILE = LOGS_DIR / "predictions.csv"

# Ensure the model checkpoint exists before proceeding
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found at {CHECKPOINT_PATH}")

# Define transformations to match training preprocessing pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    transforms.Resize((64, 64)),  # Resize images to match model input size
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize using the same mean and std as training
])

# Initialize the trained model and load saved weights
model = ResNet18().to(device)  # Load the custom ResNet model
model.load_state_dict(torch.load(CHECKPOINT_PATH))  # Load trained model weights
model.eval()  # Set the model to evaluation mode to disable gradient updates

# Function to make a prediction on a single image and log results
def predict(image_path):
    print("\nLoading image and making a prediction - \n")

    # Ensure the image file exists before proceeding
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found at {image_path}")

    # Load the image using PIL
    image = Image.open(image_path)

    # Apply transformations to prepare the image for the model
    image = transform(image).unsqueeze(0)  # Add batch dimension for inference

    # Move image to the same device as the model
    image = image.to(device)

    # Perform inference without computing gradients (improves efficiency)
    with torch.no_grad():
        output = model(image)  # Get raw model predictions
        probabilities = torch.softmax(output, dim=1)  # Convert logits to probabilities
        confidence, predicted_class = torch.max(probabilities, 1)  # Extract class with highest confidence

    # Map numerical class index to human-readable labels
    class_names = ["Cat", "Dog"]
    predicted_label = class_names[predicted_class.item()]

    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")

    # Save the prediction result to CSV for future reference
    save_prediction(image_path, predicted_label, confidence.item())

# Function to log predictions to a CSV file
def save_prediction(image_path, predicted_label, confidence):
    # Create the file and write headers if it does not already exist
    if not PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Path", "Predicted Class", "Confidence Score"])  # Column headers

    # Append the new prediction result to the CSV file
    with open(PREDICTIONS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([image_path, predicted_label, f"{confidence * 100:.2f}%"])

    print(f" The prediction is saved to {PREDICTIONS_FILE}")

# Run the script interactively if executed directly
if __name__ == "__main__":
    print(f"Using device: {device}")
    image_path = input("Enter the path to an image: ")
    predict(image_path)
