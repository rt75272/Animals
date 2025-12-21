"""Flask Web Application for Animal Classification.

User-friendly interface for uploading images and getting predictions.
"""
import json
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, jsonify, url_for
import requests

import sys
from pathlib import Path

# Add root directory to sys.path to allow imports from config and src
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import paths, params
from src.training.train import AnimalClassifier

# Initialize the application
app = Flask(__name__, static_folder='static', template_folder='templates')
# Set maximum file upload size to 16MB.
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# Directory to store uploaded images.
app.config['UPLOAD_FOLDER'] = str(paths.UPLOAD_DIR)
# Allowed file extensions for image uploads.
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
# Global variables for model and classes.
model = None
device = None
# class_to_idx (string -> int)
class_names: dict[str, int] = {}
# idx_to_class (int -> string)
idx_to_class: dict[int, str] = {}
# scientific/class-folder-name -> common name (optional)
translation: dict[str, str] = {}

def allowed_file(filename):
    """Check if file extension is allowed.

    Args:
        filename: Name of the uploaded file.
    Returns:
        Boolean indicating if file extension is valid.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def _load_idx_to_class(mapping_path: Path) -> dict[int, str]:
    with open(mapping_path, 'r') as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid mapping format in {mapping_path}")
    return {int(k): v for k, v in raw.items()}


def load_model(
    model_path: Path,
    idx_to_class_path: Path = paths.CLASSES_JSON_PATH,
):
    """Load the trained model and class mappings.

    Args:
        model_path: Path to saved model checkpoint.
        class_mapping_path: Path to class mapping JSON file.
    """
    global model, device, class_names, idx_to_class

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not idx_to_class_path.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {idx_to_class_path}. "
            "Run training first to generate models/idx_to_class.json"
        )

    # Load mapping produced by training.
    idx_to_class = _load_idx_to_class(idx_to_class_path)
    class_names = {class_name: int(index) for index, class_name in idx_to_class.items()}

    # Read num_classes from checkpoint if present, otherwise from mapping.
    checkpoint = torch.load(model_path, map_location='cpu')
    checkpoint_num_classes = checkpoint.get('num_classes') if isinstance(checkpoint, dict) else None
    num_classes = int(checkpoint_num_classes) if checkpoint_num_classes else len(idx_to_class)

    classifier = AnimalClassifier(
        num_classes,
        model_name=params.MODEL_NAME,
        pretrained=False,
        freeze_layers=False,
    )
    classifier.model.load_state_dict(checkpoint['model_state_dict'])
    model = classifier.model
    device = classifier.device
    model.eval()

    print(f"Model loaded successfully from {model_path}")
    print(f"Number of classes: {num_classes}")
    print(f"Device: {device}")

def load_translation(translation_path=paths.TRANSLATION_FILE):
    """Load the translation from scientific to common names.

    Args:
        translation_path: Path to translation JSON file.
    """
    global translation
    if not translation_path.exists():
        translation = {}
        return
    with open(translation_path, 'r') as f:
        data = json.load(f)
    translation = data if isinstance(data, dict) else {}

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

def get_predictions_from_image(image, top_k=params.TOP_K, use_tta=params.USE_TTA, tta_crops=params.TTA_CROPS):
    """
    Core prediction function that takes a PIL Image object.

    Args:
        image: A PIL Image object.
        top_k: Number of top predictions to return.
        use_tta: Whether to apply test-time augmentation.
        tta_crops: Number of augmented crops to average.
    Returns:
        List of dictionaries with class names and probabilities.
    """
    if model is None:
        raise ValueError("Model not loaded. Call load_model() first.")
    
    # Ensure image is RGB
    image = image.convert('RGB')

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

    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        class_name = idx_to_class[idx.item()]
        scientific_name = class_name
        common_name = translation.get(
            class_name,
            class_name.replace('-', ' ').replace('_', ' ').title(),
        )
        results.append({
            'class_name': class_name,
            'scientific_name': scientific_name,
            'common_name': common_name,
            'probability': prob.item() * 100
        })
    return results

def predict_image(image_path, top_k=params.TOP_K, use_tta=params.USE_TTA, tta_crops=params.TTA_CROPS):
    """Make prediction on an image from a file path.

    Args:
        image_path: Path to the image file.
        top_k: Number of top predictions to return.
        use_tta: Whether to apply test-time augmentation.
        tta_crops: Number of augmented crops to average.
    Returns:
        List of dictionaries with class names and probabilities.
    """
    try:
        image = Image.open(image_path)
        return get_predictions_from_image(image, top_k, use_tta, tta_crops)
    except Exception as e:
        app.logger.error(f"Error opening or processing image at {image_path}: {e}", exc_info=True)
        raise

@app.route('/')
def index():
    """Main page route.

    Returns:
        Rendered index.html template.
    """
    return render_template('index.html')

import logging
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and prediction by reading the file stream."""
    app.logger.info("Received a request to /predict")
    if 'file' not in request.files:
        app.logger.error("No file part in the request")
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        app.logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            app.logger.info(f"Processing image stream for file: {file.filename}")
            
            # Read image directly from the stream
            image = Image.open(file.stream)
            
            app.logger.info("Starting prediction from stream...")
            predictions = get_predictions_from_image(image)
            app.logger.info(f"Prediction successful. Top prediction: {predictions[0]['class_name']}")
            
            response_data = {'predictions': predictions}
            app.logger.info(f"Sending response: {response_data}")
            return jsonify(response_data)
        except UnidentifiedImageError:
            app.logger.error(f"Cannot identify image file: {file.filename}. The file may be corrupt or not a valid image.")
            return jsonify({'error': 'Cannot identify image file. It may be corrupt or not a supported format.'}), 400
        except Exception as e:
            app.logger.error(f"An error occurred during prediction from stream: {e}", exc_info=True)
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500
            
    app.logger.error("File object was not valid")
    return jsonify({'error': 'An unexpected error occurred with the file upload.'}), 500


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

        # Use requests to get the image from the URL
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(image_url, headers=headers, stream=True, timeout=10)
        
        # Check for successful request and content type
        if response.status_code != 200:
            return jsonify({'error': f'Failed to download image. Status code: {response.status_code}'}), 400
        
        content_type = response.headers.get('content-type')
        if not content_type or not content_type.startswith('image/'):
            return jsonify({'error': 'The provided URL does not point to a valid image file.'}), 400

        # Ensure the uploads directory exists
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        upload_dir.mkdir(exist_ok=True)
        
        # Save the valid image
        filename = 'url_image.jpg'
        filepath = upload_dir / filename
        with open(filepath, 'wb') as f:
            f.write(response.content)

        # Make prediction on the downloaded image
        predictions = predict_image(filepath, top_k=params.TOP_K)
        
        # Generate URL for the saved image to display on the frontend
        saved_image_url = url_for('static', filename=f'uploads/{filename}')
        
        return jsonify({
            'success': True,
            'image_url': saved_image_url,
            'predictions': predictions
        })
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to download image: {str(e)}'}), 500
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
    # Check if a fine-tuned model exists, otherwise fall back to the base model.
    finetuned_model_path = paths.MODELS_DIR / 'best_model_finetuned.pth'
    base_model_path = paths.MODELS_DIR / 'best_model.pth'
    
    if finetuned_model_path.exists():
        model_path_to_load = finetuned_model_path
    elif base_model_path.exists():
        model_path_to_load = base_model_path
    else:
        print(f"Warning: No trained model found.")
        print("Please train the model first by running: python app.py train")
        print("(or: python run.py train)")
        return False
        
    # Load model and translations.
    try:
        load_model(model_path=model_path_to_load, idx_to_class_path=paths.CLASSES_JSON_PATH)
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
