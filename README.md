# 🐾 Animal Classifier - AI-Powered Species Recognition

A complete machine learning pipeline built with **PyTorch** and **Flask** for classifying animal species from images. This project includes data preparation, model training, and a user-friendly web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)

## 🌟 Features

- **Deep Learning Model**: ResNet-50 architecture trained on 150+ animal species
- **Web Interface**: Beautiful, responsive Flask web application
- **High Accuracy**: Transfer learning with fine-tuned weights
- **Easy to Use**: Simple drag-and-drop interface or URL input
- **Top-5 Predictions**: View confidence scores for multiple predictions
- **Comprehensive Coverage**: Mammals, birds, reptiles, fish, insects, and extinct species

## 📁 Project Structure

```
Animals/
├── data/                      # Animal images organized by species
│   ├── translation.json       # Scientific to common name mapping
│   ├── acinonyx-jubatus/     # Cheetah images
│   ├── felis-catus/          # Cat images
│   └── ...                    # 150+ species folders
├── templates/                 # Flask HTML templates
│   ├── index.html            # Main page
│   ├── about.html            # About page
│   └── classes.html          # Species list page
├── static/                    # Static assets
│   ├── css/
│   │   └── style.css         # Stylesheet
│   └── uploads/              # Uploaded images
├── models/                    # Trained model checkpoints
├── prepare_data.py           # Data preparation script
├── train_model.py            # Model training script
├── app.py                    # Flask web application
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (8GB+ recommended for training)
- GPU optional (significantly speeds up training)

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd /home/bob/Animals
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Step 1: Prepare the Dataset

Organize images into train/validation/test splits:

```bash
python prepare_data.py
```

This will:
- Split images into 70% train, 15% validation, 15% test
- Create organized directory structure in `data/dataset/`
- Save class mapping information

#### Step 2: Train the Model

Train the deep learning model:

```bash
python train_model.py
```

Training parameters (can be modified in the script):
- **Model**: ResNet-50 (pretrained on ImageNet)
- **Epochs**: 20 (default)
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam with ReduceLROnPlateau scheduler

Training will:
- Use data augmentation for better generalization
- Save the best model based on validation accuracy
- Generate training history plots
- Create checkpoints every 5 epochs

**Note**: Training can take several hours depending on your hardware. On a modern GPU, expect 1-2 hours. On CPU, it may take 6-12 hours.

#### Step 3: Run the Web Application

Start the Flask server:

```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

## 🖥️ Using the Web Interface

### Upload an Image
1. **Drag and drop** an image onto the upload box, or
2. **Click "Choose File"** to browse your files, or
3. **Enter an image URL** and click "Analyze"

### View Predictions
- See the top 5 most likely species
- View confidence scores as percentages
- See both common and scientific names

### Browse Species
- Visit the "Species List" page to see all 150+ recognized animals
- Use the search box to find specific species

## 📊 Model Performance

The model achieves high accuracy through:
- **Transfer Learning**: Pretrained ResNet-50 on ImageNet
- **Fine-tuning**: All layers trained on animal dataset
- **Data Augmentation**: Random crops, flips, rotations, color jitter
- **Validation**: Separate validation set for hyperparameter tuning

## 🛠️ Customization

### Modify Training Parameters

Edit `train_model.py`:

```python
train_model(
    data_dir='data/dataset',
    model_name='resnet50',      # or 'resnet18' for faster training
    num_epochs=20,              # increase for better accuracy
    batch_size=32,              # decrease if out of memory
    learning_rate=0.001,        # adjust learning rate
    save_dir='models'
)
```

### Change Data Split Ratios

Edit `prepare_data.py`:

```python
prepare_dataset(
    train_ratio=0.7,   # 70% training
    val_ratio=0.15,    # 15% validation
    test_ratio=0.15    # 15% testing
)
```

### Adjust Server Settings

Edit `app.py`:

```python
app.run(
    debug=True,        # Set to False in production
    host='0.0.0.0',   # Accept external connections
    port=5000         # Change port if needed
)
```

## 📦 Dependencies

- **torch** (2.1.0): Deep learning framework
- **torchvision** (0.16.0): Computer vision utilities
- **flask** (3.0.0): Web framework
- **pillow** (10.1.0): Image processing
- **numpy** (1.26.2): Numerical computations
- **matplotlib** (3.8.2): Plotting and visualization
- **scikit-learn** (1.3.2): ML utilities
- **tqdm** (4.66.1): Progress bars
- **werkzeug** (3.0.1): WSGI utilities

## 🎯 Supported Species

The model can recognize 150+ animal species including:

- **Mammals**: Lions, tigers, elephants, pandas, wolves, etc.
- **Birds**: Eagles, parrots, penguins, hummingbirds, etc.
- **Reptiles**: Crocodiles, snakes, turtles, lizards, etc.
- **Fish**: Sharks, whales, dolphins, etc.
- **Insects**: Butterflies, bees, ants, etc.
- **Extinct**: Dinosaurs (T-Rex, Triceratops, etc.), mammoths, etc.

See the full list at `/classes` in the web interface.

## 🔧 Troubleshooting

### Model not found error
```
FileNotFoundError: models/best_model.pth not found
```
**Solution**: Train the model first using `python train_model.py`

### Out of memory during training
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in `train_model.py` (try 16 or 8)

### Import errors
```
ModuleNotFoundError: No module named 'torch'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Flask port already in use
```
OSError: [Errno 48] Address already in use
```
**Solution**: Change the port in `app.py` or kill the process using port 5000

## 🚀 Performance Tips

1. **Use GPU**: If available, PyTorch will automatically use CUDA
2. **Increase batch size**: On powerful GPUs, increase to 64 or 128
3. **Use mixed precision**: Add AMP for faster training on modern GPUs
4. **More epochs**: Train longer for better accuracy (30-50 epochs)
5. **Learning rate scheduling**: Already implemented with ReduceLROnPlateau

## 📝 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

Feel free to:
- Add more animal species
- Improve the model architecture
- Enhance the web interface
- Optimize performance
- Fix bugs

## 📧 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure you've trained the model before running the app
4. Check that your Python version is 3.8+

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **Flask Team**: For the lightweight web framework
- **ResNet Authors**: For the powerful CNN architecture
- **ImageNet**: For pretrained weights

---

**Built with ❤️ using PyTorch and Flask**
