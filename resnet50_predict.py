import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from ResNet import ResNet50

# Define CIFAR-10 classes
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet50(10).to(device)
model.load_state_dict(torch.load('cifar10_resnet50.pth', map_location=device))
model.eval()

# Define transformations for CIFAR-10
transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def predict_image(img_path):
    # Load and preprocess the image
    image = Image.open(img_path).convert('RGB')
    image = transform_test(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]

    return [predicted_class]
