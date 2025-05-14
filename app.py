from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
import base64
from torchvision import transforms
import timm

app = Flask(__name__)

# Load the model (make sure to adjust the path)
model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=6)
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define transformations
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Emotion classes
emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

def predict_emotion(image):
    image_tensor = val_test_transforms(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
    return emotions[preds.item()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        emotion = predict_emotion(image)
        return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)