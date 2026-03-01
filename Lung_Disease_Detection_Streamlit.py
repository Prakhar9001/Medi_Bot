import os
import streamlit as st
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import gdown

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model weights path and URL
MODEL_PATH = 'resnet101_lung_model.pth'
weights_url = "https://drive.google.com/uc?id=1i9D0BURrD_mlRnpmdM234PMWpFBxmMkX"  # Direct download URL

# Download weights from Google Drive if not found locally
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model weights..."):
        gdown.download(weights_url, MODEL_PATH, quiet=False)

# Load ResNet50 architecture and replace the final layer
model = models.resnet50(weights=None)  # No pre-trained weights

# Adjust the fully connected layer to match the state dict
num_classes = 5  # Adjust for your use case
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),  # Add an intermediate layer if required
    nn.ReLU(),
    nn.Linear(512, num_classes)
)
model = model.to(device)

# Load custom weights safely
try:
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)  # Safe loading
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # Set model to evaluation mode
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# Define image transformations for inference
image_size = 224
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.Resize((image_size, image_size)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])  # Normalization for ResNet
])

# Define class names
class_names = ['COVID', 'Normal', 'Pneumonia', 'Pneumothorax', 'Tuberculosis']

# Streamlit UI
st.title("Lung Disease Detection using ResNet101")
st.write("Upload a chest X-ray image to detect lung disease.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open the uploaded image and convert to grayscale
    image = Image.open(uploaded_file).convert("L")  # Load as grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]  # Convert to NumPy array
        pred_class = torch.argmax(output, dim=1).item()

    confidence_threshold = 50.0  # Set threshold to 50%

    # Get max probability and corresponding class
    max_prob = probs[pred_class] * 100

    if max_prob < confidence_threshold:
        st.write(f"Model is uncertain with a maximum confidence of {max_prob:.2f}%.")
    else:
        st.write(f"Predicted Class: **{class_names[pred_class]}**")
        st.write(f"Confidence: {max_prob:.2f}%")

    # Format probabilities as percentages with corresponding class names
    formatted_probs = {class_names[i]: f"{probs[i] * 100:.2f}%" for i in range(num_classes)}

    # Display the results
    st.write("Class Probabilities:")
    for class_name, prob in formatted_probs.items():
        st.write(f"{class_name}: {prob}")
