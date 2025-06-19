import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
from model_definition import CNN_GNN_Model

# Class names from dataset
class_names = ['Apple Scab Leaf', 'Apple rust leaf', 'Blueberry leaf', 'Peach leaf', 'Tomato leaf']

# Image transforms (match validation)
transform = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    model = CNN_GNN_Model(num_classes=len(class_names))
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def predict(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)[0].numpy()
        predicted_class = np.argmax(probs)
    return class_names[predicted_class], probs

st.title("🌿 Plant Leaf Disease Detection")
st.write("Powered by CNN-GNN hybrid model (EfficientNet-B3 + GCN)")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label, probs = predict(image)
        st.success(f"Predicted: **{label}**")
        st.subheader("Confidence Scores")
        for cname, prob in zip(class_names, probs):
            st.write(f"{cname}: {prob * 100:.2f}%")
