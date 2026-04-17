import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    print("Loading model...")

    # Load model from YAML configuration (custom architecture)
    # Alternatively, you can load pretrained weights for better stability:
    # model = YOLO('yolo11n.pt')

    model = YOLO('')

    print("Model loaded.")

    print("Training model...")

    model.train(
        data='',  # Path to dataset config (contains image paths and class info)
        cache=False,            # Whether to cache images for faster training
        imgsz=1280,             # Input image size (will be resized to imgsz x imgsz)
        epochs=300,             # Total number of training epochs
        batch=8,                # Batch size (number of images per iteration)
        close_mosaic=50,        # Disable Mosaic augmentation in the last N epochs
        workers=16,             # Number of dataloader workers (threads)
        patience=30,            # Early stopping if no improvement for N epochs
        device='0',             # GPU device ID (e.g., '0' for single GPU)
        amp=False,              # Use Automatic Mixed Precision (AMP) or not
        optimizer='SGD'         # Optimizer type (SGD or Adam)
    )

    print("Training completed.")