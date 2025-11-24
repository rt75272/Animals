"""Flask Web Application for Animal Classification.

User-friendly interface for uploading images and getting predictions.
"""
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

app = Flask(__name__)
# Set maximum file upload size to 16MB.
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# Directory to store uploaded images.
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# Allowed file extensions for image uploads.
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
# Global variables for model and classes.
model = None
device = None
class_names = {}
idx_to_class = {}
translation = {}

def allowed_file(filename):
    """Check if file extension is allowed.

    Args:
        filename: Name of the uploaded file.
    Returns:
        Boolean indicating if file extension is valid.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model(model_path='models/best_model.pth', class_mapping_path='models/class_mapping.json'):
    """Load the trained model and class mappings.

    Args:
        model_path: Path to saved model checkpoint.
        class_mapping_path: Path to class mapping JSON file.
    """
    global model, device, class_names, idx_to_class
    # Determine compute device (GPU or CPU).
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load class mapping from JSON file.
    with open(class_mapping_path, 'r') as f:
        mapping = json.load(f)
        class_names = mapping['class_to_idx']
        idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
    num_classes = len(class_names)
    # Initialize ResNet50 model architecture.
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    # Replace final layer to match training architecture with enhanced classifier.
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    # Load trained weights from checkpoint.
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    # Set model to evaluation mode.
    model.eval()
    print(f"Model loaded successfully from {model_path}")
    print(f"Number of classes: {num_classes}")
    print(f"Device: {device}")

def load_translation(translation_path='data/translation.json'):
    """Load the translation from scientific to common names.

    Args:
        translation_path: Path to translation JSON file.
    """
    global translation
    with open(translation_path, 'r') as f:
        translation = json.load(f)

def get_transform():
    """Get base image preprocessing transform for a single crop.

    Returns:
        Composed transforms for image preprocessing.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_tta_transforms(num_crops=5):
    """Get a list of transforms for test-time augmentation.

    Args:
        num_crops: Number of random crops to generate.
    Returns:
        List of composed transforms for TTA.
    """
    transforms_list = []
    base = [
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    for _ in range(num_crops):
        transforms_list.append(transforms.Compose(base))
    return transforms_list

def predict_image(image_path, top_k=5, use_tta=True, tta_crops=5):
    """Make prediction on an image with optional test-time augmentation.

    Args:
        image_path: Path to the image file.
        top_k: Number of top predictions to return.
        use_tta: Whether to apply test-time augmentation.
        tta_crops: Number of augmented crops to average.
    Returns:
        List of dictionaries with class names and probabilities.
    """
    if model is None:
        raise ValueError("Model not loaded. Call load_model() first.")
    image = Image.open(image_path).convert('RGB')
    # Prepare tensors for either single-crop or TTA inference.
    with torch.no_grad():
        if use_tta and tta_crops > 1:
            transforms_list = get_tta_transforms(num_crops=tta_crops)
            logits_sum = None
            for t in transforms_list:
                tensor = t(image).unsqueeze(0).to(device)
                outputs = model(tensor)
                if logits_sum is None:
                    logits_sum = outputs
                else:
                    logits_sum += outputs
            outputs = logits_sum / float(tta_crops)
        else:
            transform = get_transform()
            tensor = transform(image).unsqueeze(0).to(device)
            outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
    # Format results with scientific and common names.
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        class_name = idx_to_class[idx.item()]
        common_name = translation.get(class_name, class_name.replace('-', ' ').title())
        probability = prob.item() * 100
        results.append({
            'scientific_name': class_name,
            'common_name': common_name,
            'probability': probability
        })
    return results

@app.route('/')
def index():
    """Main page route.

    Returns:
        Rendered index.html template.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction.

    Returns:
        JSON response with predictions or error message.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)'}), 400
    try:
        # Create upload directory if it doesn't exist.
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        # Sanitize filename for security.
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Make prediction on uploaded image.
        predictions = predict_image(filepath, top_k=5)
        # Generate URL for uploaded image.
        image_url = url_for('static', filename=f'uploads/{filename}')
        return jsonify({
            'success': True,
            'image_url': image_url,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict_url', methods=['POST'])
def predict_url():
    """Handle image URL prediction.

    Returns:
        JSON response with predictions or error message.
    """
    try:
        data = request.get_json()
        image_url = data.get('url')
        if not image_url:
            return jsonify({'error': 'No URL provided'}), 400
        # Download image from URL and save locally.
        import urllib.request
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = 'url_image.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        urllib.request.urlretrieve(image_url, filepath)
        # Make prediction on downloaded image.
        predictions = predict_image(filepath, top_k=5)
        # Generate URL for saved image.
        saved_image_url = url_for('static', filename=f'uploads/{filename}')
        return jsonify({
            'success': True,
            'image_url': saved_image_url,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/about')
def about():
    """About page route.

    Returns:
        Rendered about.html template with model info.
    """
    return render_template('about.html',
                         num_classes=len(class_names),
                         device=str(device))

@app.route('/classes')
def classes():
    """Show all available classes route.

    Returns:
        Rendered classes.html template with species list.
    """
    # Create list of all classes with their common names.
    all_classes = []
    for sci_name in sorted(class_names.keys()):
        common_name = translation.get(sci_name, sci_name.replace('-', ' ').title())
        all_classes.append({
            'scientific': sci_name,
            'common': common_name
        })
    return render_template('classes.html', classes=all_classes)

def initialize_app():
    """Initialize the application.

    Returns:
        Boolean indicating successful initialization.
    """
    print("Initializing Animal Classifier Web App...")
    # Check if trained model exists.
    model_path = 'models/best_model.pth'
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("Please train the model first by running: python train_model.py")
        return False
    # Load model and translations.
    try:
        load_model()
        load_translation()
        print("Application initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing application: {e}")
        return False

if __name__ == '__main__':
    if initialize_app():
        print("\n" + "="*60)
        print("Starting Flask server...")
        print("Open your browser and go to: http://localhost:5000")
        print("="*60 + "\n")
        # Start Flask development server.
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\nFailed to initialize application.")
        print("Please ensure you have trained the model first.")
