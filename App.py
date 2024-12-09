import streamlit as st
import torch
from torchvision import models, transforms, datasets
from torch import nn
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

# Configurations
num_classes = 10  # Modifier en fonction de votre dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation personnalisée pour l'égalisation d'histogramme
class HistogramEqualization:
    def __call__(self, img):
        return ImageOps.equalize(img)

# Pipeline de transformation des images
transform = transforms.Compose([
    HistogramEqualization(),  # Égalisation d'histogramme
    transforms.Resize((224, 224)),  # Redimensionnement
    transforms.RandomHorizontalFlip(p=0.5),  # Retourner horizontalement
    transforms.RandomRotation(10),  # Rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Jitter de couleur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Whitening
])

# Charger les modèles
def load_model(model_name):
    if model_name == "ResNet50":
        model = models.resnet50(weights="DEFAULT")
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
    elif model_name == "ConvNeXt":
        model = models.convnext_base(weights="DEFAULT")
        model.classifier[2] = nn.Linear(in_features=model.classifier[2].in_features, out_features=num_classes, bias=True)
    model.load_state_dict(torch.load(f"{model_name}_transfer_learning.pth"))
    model = model.to(device)
    model.eval()
    return model

# Prédire l'image
def predict_image(image, model_name):
    model = load_model(model_name)
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Analyse exploratoire
def display_exploratory_analysis():
    st.subheader("Analyse exploratoire des données")
    image_folder = "C:/Users/ouedraogo080976/Desktop/OpenClassroom/PROJET 7/Développez_une_preuve_de_concept_OUEDRAOGO_Mahamady/data/train"  # Chemin vers votre dataset
    dataset = datasets.ImageFolder(root=image_folder)
    st.image([dataset[i][0] for i in range(5)], width=100)
    st.write("Exemple d'images provenant de notre dataset.")

# Interface utilisateur avec Streamlit
st.title("Dashboard de classification d'images avec ResNet50 et ConvNeXt")

# Afficher l'analyse exploratoire des données
display_exploratory_analysis()

# Sélectionner le modèle
model_name = st.selectbox("Choisissez un modèle", ["ResNet50", "ConvNeXt"])

# Télécharger une image pour la prédiction
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Affichage de l'image transformée
    transformed_image = transform(image)
    st.image(transformed_image.permute(1, 2, 0).numpy(), caption="Image transformée", use_column_width=True, clamp=True)

    # Prédiction sur l'image téléchargée
    if st.button("Faire une prédiction"):
        predicted_class = predict_image(image, model_name)
        st.write(f"Classe prédite : {predicted_class}")

        # Afficher les graphiques de comparaison des pertes et précisions
        st.subheader("Comparaison des pertes et précisions")
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Exemple de graphes de pertes
        epochs = range(1, 11)  # 10 époques
        axs[0].plot(epochs, np.random.rand(10), label="Train Loss")
        axs[0].plot(epochs, np.random.rand(10), label="Validation Loss")
        axs[0].set_title('Comparaison des pertes')
        axs[0].set_xlabel('Époques')
        axs[0].set_ylabel('Perte')
        axs[0].legend()

        # Exemple de graphes de précisions
        axs[1].plot(epochs, np.random.rand(10) * 100, label="Train Accuracy")
        axs[1].plot(epochs, np.random.rand(10) * 100, label="Validation Accuracy")
        axs[1].set_title('Comparaison des précisions')
        axs[1].set_xlabel('Époques')
        axs[1].set_ylabel('Précision (%)')
        axs[1].legend()

        st.pyplot(fig)

# Lancer le serveur Streamlit
if __name__ == "__main__":
    st.write("Chargement du modèle et des images...")
