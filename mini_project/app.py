import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os

# ====== Config ======
st.set_page_config(page_title="SmartSkin: Skincare Recommender", layout="centered")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Dry', 'Normal', 'Oily']

# ====== Load Model ======
@st.cache_resource
def load_model():
    try:
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(class_names))

        model_path = "model/best_resnet50_skin_model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# ===== Image Transform =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== Prediction Function ======
def predict_skin_type(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# ====== Streamlit UI ======
st.title("💡 SmartSkin: Personalized Skincare Recommender")
st.write("Upload a clear image of your face to detect your skin type and receive skincare product suggestions!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing your skin type..."):
        prediction = predict_skin_type(image)
    
    st.success(f"✅ Detected Skin Type: **{prediction}**")

    # === Dummy Product Recommendations ===
    if prediction == 'Dry':
        st.info("💧 Recommended: Hydrating cleanser, moisturizing cream, and gentle exfoliator.")
    elif prediction == 'Oily':
        st.info("🧼 Recommended: Oil-free cleanser, mattifying toner, and clay mask.")
    else:
        st.info("🌿 Recommended: Gentle foaming cleanser, balancing toner, and light moisturizer.")

