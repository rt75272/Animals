# config/params.py

# Data preparation parameters
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model and training parameters
MODEL_NAME = 'resnet50'
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
LABEL_SMOOTHING = 0.05
WEIGHT_DECAY = 0.02
GRADIENT_CLIP_NORM = 1.0

# Early stopping parameters
PATIENCE = 15
MIN_DELTA = 0.01

# Fine-tuning parameters
FINETUNE = True
FINETUNE_LAYERS = ('layer3',)
FINETUNE_EPOCHS = 15
FINETUNE_LR_FACTOR = 0.1

# Inference parameters
TOP_K = 5
USE_TTA = True
TTA_CROPS = 5
