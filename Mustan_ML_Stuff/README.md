# Mustan ML Stuff

Machine Learning projects and pipelines for computer vision and deep learning.

## Projects

### 1. Eye Movement Detection Pipeline
Real-time eye movement detection and classification into 10 categories.
- 📁 Location: `eye_pipeline/`
- 📖 [View Documentation](eye_pipeline/README.md)

### 2. Eye Detection & Tracking with YOLOv8
Object detection pipeline for eye tracking with movement analysis.
- 📁 Location: `eye_detection_tracking.py`
- 📖 [View Guide](eye_dataset_guide.md)

### 3. PyTorch Model Inspector
Analyze and test any PyTorch `.pth` model file.
- 📁 Location: `model_inspector.py`, `test_pretrained_model.py`
- 📖 [View Documentation](README_model_inspector.md)

## Quick Start

```bash
# Install dependencies
pip install -r requirements_yolo.txt

# Run eye movement detection
cd eye_pipeline
python main.py

# Inspect a model
python test_pretrained_model.py

# Run eye tracking
python eye_detection_tracking.py
```

## Sharing Models with Collaborators

Models are ignored by default in `.gitignore` to prevent large files in git. Here are three ways to share models:

### Option 1: Whitelist Specific Models (Small Models < 100MB)

Edit `.gitignore` and uncomment specific models:
```gitignore
# Whitelist your model
!pretrainedModel.pth
!eye_movement_model.pth
!models/baseline_model.pth
```

Then commit normally:
```bash
git add pretrainedModel.pth
git commit -m "Add pretrained model"
git push
```

### Option 2: Git LFS (Recommended for Large Models)

Git Large File Storage handles large files efficiently:

```bash
# One-time setup
git lfs install

# Track model files
git lfs track "*.pth"
git lfs track "*.pt"

# Commit the tracking file
git add .gitattributes
git commit -m "Configure Git LFS"

# Add and commit models
git add models/
git commit -m "Add models via Git LFS"
git push
```

**Collaborators clone with:**
```bash
git lfs install
git clone <repository-url>
```

### Option 3: External Storage (Best for Very Large Models > 1GB)

Host models externally and provide download links:

**Popular Options:**
- 🔗 **Google Drive**: Public sharing link
- ☁️ **AWS S3**: Cloud storage with wget/curl
- 🤗 **HuggingFace Hub**: ML model hosting
- 📦 **GitHub Releases**: Attach to releases
- 🗄️ **Dropbox/OneDrive**: Shared folders

**Example: Create a download script**

Create `download_models.sh`:
```bash
#!/bin/bash
# Download pretrained model
wget -O pretrainedModel.pth "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
echo "Model downloaded successfully!"
```

Or Python script `download_models.py`:
```python
import requests
import os

def download_model(url, filename):
    """Download model from URL"""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✓ {filename} downloaded!")

if __name__ == "__main__":
    # Add your model URLs
    models = {
        "pretrainedModel.pth": "YOUR_DOWNLOAD_URL",
        "eye_movement_model.pth": "YOUR_DOWNLOAD_URL"
    }
    
    for filename, url in models.items():
        if not os.path.exists(filename):
            download_model(url, filename)
        else:
            print(f"✓ {filename} already exists")
```

Then in your README, tell collaborators:
```bash
# Download models before running
python download_models.py
```

## Recommended Approach

| Model Size | Recommended Method | Why |
|-----------|-------------------|-----|
| < 10MB | Whitelist in git | Simple, no extra setup |
| 10MB - 100MB | Git LFS | Good balance, version controlled |
| > 100MB | External Storage | Faster clones, no size limits |

## Project Structure

```
Mustan_ML_Stuff/
├── README.md                        # This file
├── requirements_yolo.txt            # Dependencies
├── .gitignore                       # Git ignore rules
│
├── eye_pipeline/                    # Eye movement detection
│   ├── main.py
│   ├── README.md
│   └── modules/
│
├── eye_detection_tracking.py        # YOLOv8 eye tracking
├── eye_dataset_guide.md
│
├── model_inspector.py               # Model analysis tool
├── test_pretrained_model.py
├── README_model_inspector.md
│
├── example.py                       # Example scripts
└── fastapi.py
```

## Dependencies

All projects use the same dependencies:

```bash
pip install -r requirements_yolo.txt
```

Key packages:
- `ultralytics` - YOLOv8
- `torch` - PyTorch
- `opencv-python` - Computer vision
- `numpy` - Numerical computing
- `pillow` - Image processing

## Contributing

When contributing models or datasets:

1. **Small files (< 10MB)**: Commit directly
2. **Large files (> 10MB)**: Use Git LFS or external storage
3. **Datasets**: Always use external storage or DVC
4. **Update documentation**: Keep READMEs current

## License

Part of the RUBIX26_15_ICodeSpeed repository.
