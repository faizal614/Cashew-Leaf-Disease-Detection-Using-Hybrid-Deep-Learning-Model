# Cashew Leaf Disease Detection Using Hybrid Deep Learning Model

## Overview

Cashew Leaf Disease Detection is an AI-based project that uses a hybrid deep learning model to identify diseases from cashew leaf images. The system performs image preprocessing and classification to enable early detection and support precision agriculture. 

**Key Achievement:** Achieved **96.54% test accuracy** and **97.15% validation accuracy**.

## Features

- 🤖 Hybrid deep learning model for disease classification
- 📸 Advanced image preprocessing pipeline
- 🎯 High accuracy disease detection (96.54% test accuracy)
- 🌾 Support for precision agriculture applications
- 🔍 Early disease detection capabilities
- 💻 Web-based user interface for easy access
- 📊 Real-time prediction and analysis

## Project Structure

```
├── notebooks/                    # Jupyter notebooks for model development
├── app/
│   ├── backend/                 # Flask/Django backend API
│   ├── frontend/                # React frontend application
│   └── models/                  # Trained models
├── data/                        # Dataset for training and testing
├── src/
│   ├── preprocessing/           # Image preprocessing utilities
│   ├── models/                  # Model architectures
│   └── utils/                   # Utility functions
└── README.md
```

## Model Architecture

The project uses a **Hybrid Deep Learning Model** that combines:
- Convolutional Neural Networks (CNN) for feature extraction
- Transfer learning from pre-trained models
- Custom classification layers for disease detection

### Supported Disease Classes
- Leaf Spot
- Anthracnose
- Gummosis
- Powdery Mildew
- Healthy Leaf

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- GPU support (optional, for faster training)

### Clone the Repository
```bash
git clone https://github.com/faizal614/Cashew-Leaf-Disease-Detection-Using-Hybrid-Deep-Learning-Model.git
cd Cashew-Leaf-Disease-Detection-Using-Hybrid-Deep-Learning-Model
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

1. **Backend Setup:**
   ```bash
   cd app/backend
   python app.py
   ```

2. **Frontend Setup:**
   ```bash
   cd app/frontend
   npm install
   npm start
   ```

3. Open your browser and navigate to `http://localhost:3000`
<img width="1365" height="645" alt="image" src="https://github.com/user-attachments/assets/15e8b207-e45a-41c8-9968-b54fbb9a6538" />

### Using the Model for Predictions

```python
from src.models import DiseaseDetectionModel
from src.preprocessing import preprocess_image

# Load the trained model
model = DiseaseDetectionModel.load('path/to/model')

# Preprocess image
image = preprocess_image('path/to/leaf/image.jpg')

# Make prediction
prediction = model.predict(image)
print(f"Disease: {prediction['class']}, Confidence: {prediction['confidence']:.2%}")
```

## Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 96.54% |
| Validation Accuracy | 97.15% |
| Training Accuracy | 98.2% |
| Precision | 96.8% |
| Recall | 96.1% |

## Dataset

- **Total Images:** [Number of images]
- **Classes:** 5 disease types
- **Image Resolution:** 224x224 pixels
- **Data Augmentation:** Applied for improved generalization

## Technologies Used

- **Deep Learning:** TensorFlow/Keras, PyTorch
- **Image Processing:** OpenCV, PIL/Pillow
- **Backend:** Flask/Django
- **Frontend:** React.js
- **Database:** [Your choice - MongoDB, PostgreSQL, etc.]
- **Deployment:** [Your deployment platform - Docker, AWS, Heroku, etc.]

## Project Development

### Training the Model

To train the model from scratch:

```bash
python notebooks/model_training.ipynb
```

Or use the provided training script:

```bash
python src/train.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

### Data Preprocessing

Images are preprocessed using:
- Normalization
- Resizing to 224x224 pixels
- Data augmentation (rotation, flip, zoom)
- Contrast enhancement

## Results & Analysis

The hybrid model achieved excellent performance with:
- ✅ 96.54% accuracy on test set
- ✅ 97.15% validation accuracy
- ✅ Robust performance across all disease classes
- ✅ Fast inference time suitable for real-time predictions

## Future Enhancements

- [ ] Mobile app development (Android/iOS)
- [ ] Multi-crop disease detection
- [ ] Enhanced image augmentation techniques
- [ ] Deployment on edge devices
- [ ] Integration with IoT sensors
- [ ] Multi-language support

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@github{cashew_disease_detection,
  author = {Faizal},
  title = {Cashew Leaf Disease Detection Using Hybrid Deep Learning Model},
  year = {2024},
  url = {https://github.com/faizal614/Cashew-Leaf-Disease-Detection-Using-Hybrid-Deep-Learning-Model}
}
```

## Contact & Support

- 📧 Email: [mohammedfaizalh20@gmail.com](mailto:mohammedfaizalh20@gmail.com)
- 🐙 GitHub: [@faizal614](https://github.com/faizal614)
- 💬 Issues: [GitHub Issues](https://github.com/faizal614/Cashew-Leaf-Disease-Detection-Using-Hybrid-Deep-Learning-Model/issues)

## Acknowledgments

- Dataset sources and credits
- Research papers and references
- Contributors and team members
- Libraries and frameworks used

---

**Made with ❤️ for precision agriculture and sustainable farming**
