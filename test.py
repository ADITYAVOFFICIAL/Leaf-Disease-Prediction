import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet34

# Load the pre-trained model
model = resnet34(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Assuming we have 2 classes: 'healthy' and 'spotted'

# Load the state dictionary
checkpoint = torch.load('resnet34_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return 'healthy' if predicted.item() == 0 else 'spotted'

st.title('Leaf Health Classifier')
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Convert image to RGB
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(image)
    st.write(f'The leaf is {label}.')