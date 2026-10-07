import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os

# Import your trained CNN model
from model.cnn_model import CNNModel

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = CNNModel()
model.load_state_dict(torch.load("saved_model.pth", map_location=device))
model.to(device)
model.eval()

# Define image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Prediction function
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    return "Real ✅" if predicted.item() == 0 else "Fake ❌"

# Main execution from terminal
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("⚠️ Usage: python predict.py path/to/image.jpg")
    else:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
        else:
            result = predict_image(image_path)
            print(f"🔎 Prediction result: {result}")

