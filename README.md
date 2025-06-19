# 🌿 Plant Disease Detection using CNN-GNN Hybrid Model

A plant leaf disease classification system using EfficientNet (CNN) + Graph Neural Network (GCN) with a Streamlit-based user interface.

## 🚀 Features
- CNN-GNN hybrid deep learning model
- EfficientNet-B3 for feature extraction
- GCN for learning spatial relationships between leaf embeddings
- Real-time prediction UI using Streamlit
- Tested on PlantDoc dataset (subset)

## 📁 Dataset Used
[PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)

## 🧠 Libraries
- PyTorch
- PyTorch Geometric
- Streamlit
- TIMM (EfficientNet)
- Albumentations

## 🎯 Model Accuracy
- Achieved validation accuracy: ~90%
- Final test accuracy: ~70%

## 🖥️ How to Run in VS Code
```bash
pip install -r requirements.txt
streamlit run app.py
