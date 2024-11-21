import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model architecture (must match the trained model)
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2)  # Assuming binary classification: genuine vs. forged
)
model = model.to(device)

# Load the saved model weights
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_signature(image_path):
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
    
    # Convert prediction to human-readable format
    class_names = ['Genuine', 'Forged']
    pred_class = class_names[preds.item()]
    pred_prob = probs[0][preds.item()].item() * 100

    return pred_class, pred_prob

# Example usage
image_path = r"C:\path_to_signature_image.jpg"  # Update this path to the signature image you want to predict
pred_class, pred_prob = predict_signature(image_path)
print(f"Prediction: {pred_class} ({pred_prob:.2f}%)")
