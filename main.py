import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import Label, Button

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet Model
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)  # Adjust output layer for 7 classes
model.to(device)

# Load Trained Weights
try:
    model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    model.eval()  # Set model to evaluation mode
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model: {e}")

# Define image transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match EfficientNet input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

# Full Disease Name Mapping
label_map = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

# Function to predict image
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Open image
    image = transform(image).unsqueeze(0).to(device)  # Apply transformations and move to device
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Get predicted class
    
    return label_map[predicted.item()]

# Function to open file dialog and predict
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        result = predict_image(file_path)
        result_label.config(text=f"Predicted Disease: {result}")

# Initialize Tkinter Window
root = tk.Tk()
root.title("Skin Disease Classifier")
root.geometry("400x300")

# UI Elements
title_label = Label(root, text="Skin Disease Classification", font=("Arial", 16))
title_label.pack(pady=10)

upload_button = Button(root, text="Select Image", command=open_image, font=("Arial", 12))
upload_button.pack(pady=10)

result_label = Label(root, text="Prediction: ", font=("Arial", 14))
result_label.pack(pady=10)

# Run Tkinter GUI
root.mainloop()
