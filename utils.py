import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Skin type labels
class_names = ['Dry', 'Normal', 'Oily']

# Image transformation for prediction
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model
def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model_path = os.path.join("model", "best_resnet50_skin_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict skin type
def predict_skin_type(image, model):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Load products
def load_products():
    df = pd.read_csv("data/products.csv")
    return df

# Get recommendations
def recommend_products(skin_type, df, top_n=3):
    filtered = df[df['SkinType'].str.lower() == skin_type.lower()]
    return filtered.head(top_n).to_dict(orient='records')