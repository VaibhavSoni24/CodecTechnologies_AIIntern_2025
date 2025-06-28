# 🍎 Fruit Image Classifier - Deep Learning CNN Model

An advanced fruit classification system built using **Convolutional Neural Networks (CNN)** and **TensorFlow/Keras**. This project demonstrates the power of deep learning in computer vision by accurately identifying and classifying fruits from images using the comprehensive Fruits-360 dataset.

## 🎯 Project Overview

This is **Project-5** of the Codec Technologies internship program - my second self mini-project after completing the 1st month training. The system implements a sophisticated deep learning model that can:
- **Classify 131+ fruit varieties** with high accuracy
- **Process 100x100 standardized images** for optimal performance  
- **Provide detailed fruit information** including nutritional facts and characteristics
- **Achieve real-time predictions** on new fruit images

## 🏆 Key Achievements

✅ **Training on 100% dataset** - Utilizes the complete Fruits-360 dataset for maximum learning  
✅ **80/20 intelligent split** - Proper validation methodology for reliable performance metrics  
✅ **Real-time predictions** - Upload any fruit image and get instant classification  
✅ **Rich fruit details** - Automatically fetches nutritional and botanical information  
✅ **Standardized pipeline** - Clean, reproducible workflow from data to deployment  

## 📊 Dataset Specifications

### Fruits-360 100x100 Dataset
- **Source**: [GitHub Repository](https://github.com/fruits-360/fruits-360-100x100)
- **Image Size**: 100×100 pixels (standardized)
- **Total Classes**: 131+ fruit varieties
- **Total Images**: 90,000+ high-quality fruit images
- **Format**: RGB color images
- **Quality**: Professional photography with consistent lighting

### Dataset Structure
```
fruits-360/
├── Training/          # Original training set
│   ├── Apple Braeburn/
│   ├── Apple Golden 1/
│   ├── Banana/
│   ├── Orange/
│   └── ... (131+ classes)
└── Test/             # Original test set
    ├── Apple Braeburn/
    ├── Apple Golden 1/
    └── ... (same classes)
```

### Sample Fruit Classes
- **Apples**: Braeburn, Golden, Granny Smith, Red varieties
- **Citrus**: Orange, Lemon, Lime, Grapefruit  
- **Tropical**: Mango, Pineapple, Papaya, Kiwi
- **Berries**: Strawberry, Blueberry, Raspberry
- **Stone Fruits**: Peach, Plum, Apricot
- **And many more!**

## 🧠 Deep Learning Architecture

### CNN Model Design
```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3))
    MaxPooling2D((2,2))
    
    Conv2D(64, (3,3), activation='relu')
    MaxPooling2D((2,2))
    
    Conv2D(128, (3,3), activation='relu')
    MaxPooling2D((2,2))
    
    Flatten()
    Dense(256, activation='relu')
    Dropout(0.5)
    Dense(131+, activation='softmax')  # Number of fruit classes
])
```

### Model Specifications
- **Input Layer**: 100×100×3 (RGB images)
- **Convolutional Layers**: 3 layers with increasing filters (32→64→128)
- **Pooling**: MaxPooling for dimension reduction
- **Fully Connected**: 256-neuron dense layer with dropout
- **Output**: Softmax activation for multi-class classification
- **Optimizer**: Adam for efficient training
- **Loss Function**: Categorical crossentropy

## 🛠️ Technology Stack

### Core Libraries
- **TensorFlow 2.x** - Deep learning framework
- **Keras** - High-level neural network API
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning utilities

### Computer Vision
- **ImageDataGenerator** - Data augmentation and preprocessing
- **PIL/OpenCV** - Image processing
- **Image preprocessing** - Normalization and resizing

### Data Management
- **Requests** - Dataset downloading
- **ZipFile** - Archive extraction
- **OS/Shutil** - File system operations

## 🚀 Features & Capabilities

### 1. Intelligent Data Pipeline
- **Automatic download** from GitHub repository
- **Smart data merging** (Training + Test → Complete dataset)
- **80/20 random split** for proper validation
- **Real-time data augmentation** during training

### 2. Advanced Model Training
- **Progressive learning** with 20 epochs
- **Dropout regularization** to prevent overfitting
- **Batch processing** with optimized batch size (64)
- **Real-time validation** monitoring

### 3. Comprehensive Evaluation
- **Training/Validation accuracy plots**
- **Loss curve visualization**
- **Model performance metrics**
- **Confusion matrix analysis** (optional)

### 4. Production-Ready Prediction
- **Single image prediction** with confidence scores
- **Batch image processing** capability
- **Model persistence** (.h5 format)
- **Class label mapping** for human-readable results

### 5. Rich Fruit Information
- **Nutritional facts** from metadata CSV
- **Botanical classification**
- **Growing region information**
- **Seasonal availability data**

## 📋 Step-by-Step Implementation

### Phase 1: Environment Setup
1. Install required deep learning libraries
2. Import necessary modules and dependencies
3. Configure GPU acceleration (if available)

### Phase 2: Data Acquisition
1. Download Fruits-360 dataset from GitHub
2. Extract and organize image directories
3. Merge training and test sets for complete dataset

### Phase 3: Data Preprocessing
1. Create ImageDataGenerator with normalization
2. Implement 80/20 train-validation split
3. Configure batch processing and augmentation

### Phase 4: Model Architecture
1. Design CNN architecture with optimal layers
2. Compile model with appropriate optimizer
3. Display model summary and parameter count

### Phase 5: Training Process
1. Train model with validation monitoring
2. Implement early stopping (optional)
3. Save training history for analysis

### Phase 6: Model Evaluation
1. Generate accuracy and loss plots
2. Evaluate model performance metrics
3. Analyze validation results

### Phase 7: Model Deployment
1. Save trained model for reuse
2. Create prediction pipeline
3. Implement class label mapping

### Phase 8: Fruit Information System
1. Download metadata CSV from repository
2. Create fruit detail lookup function
3. Integrate with prediction system

## 📈 Expected Performance

### Model Metrics
- **Training Accuracy**: 95%+ expected
- **Validation Accuracy**: 90%+ expected
- **Training Time**: ~20-30 minutes (20 epochs)
- **Inference Speed**: <1 second per image
- **Model Size**: ~50-100 MB

### Hardware Requirements
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for dataset and model
- **GPU**: Optional but recommended for faster training
- **CPU**: Multi-core processor preferred

## 🔧 Installation & Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd Project-5
```

### 2. Install Dependencies
```bash
pip install tensorflow numpy matplotlib scikit-learn pandas requests
```

### 3. Run the Notebook
```bash
jupyter notebook fruit_prediction_model.ipynb
```

### 4. Alternative: Google Colab
- Upload notebook to Google Colab
- Enable GPU acceleration
- Run all cells sequentially

## 📁 Project Structure

```
Project-5/
├── README.md                           # Comprehensive project documentation
├── fruit_prediction_model.ipynb       # Complete implementation notebook
├── fruits-360/                        # Downloaded dataset (auto-generated)
│   └── fruits-360-100x100-master/
├── fruits-merged/                      # Processed dataset (auto-generated)
├── fruit_classifier_model.h5          # Trained model (after training)
└── svd_model.pkl                       # Saved model (optional)
```

## 🎯 Usage Examples

### Basic Prediction
```python
# Load and predict single image
predicted_fruit = predict_image('path/to/your/apple.jpg')
print(f"Predicted Fruit: {predicted_fruit}")

# Get detailed information
details = fruit_details(predicted_fruit)
print(f"Nutritional Info: {details}")
```

### Batch Processing
```python
# Process multiple images
fruits = ['apple.jpg', 'banana.jpg', 'orange.jpg']
for fruit_img in fruits:
    prediction = predict_image(fruit_img)
    print(f"{fruit_img} → {prediction}")
```

## 🎓 Learning Outcomes

This project demonstrates mastery of:

### Deep Learning Concepts
- **Convolutional Neural Networks** architecture design
- **Image preprocessing** and data augmentation
- **Transfer learning** principles
- **Model optimization** techniques

### Computer Vision Skills
- **Image classification** pipeline development
- **Data visualization** for model analysis
- **Real-time prediction** implementation
- **Model deployment** strategies

### Software Engineering
- **End-to-end pipeline** development
- **Reproducible research** methodology
- **Code organization** and documentation
- **Error handling** and edge cases

## 🌟 Unique Features

### 1. Complete Dataset Utilization
Unlike typical tutorials that use subsets, this project leverages the **entire Fruits-360 dataset** for maximum learning potential.

### 2. Professional Data Pipeline
Implements industry-standard practices including:
- Automated data download and extraction
- Proper train-validation splitting
- Real-time data augmentation

### 3. Production-Ready Architecture
The model is designed for real-world deployment with:
- Optimized inference speed
- Robust error handling
- Scalable prediction pipeline

### 4. Rich Information Integration
Goes beyond simple classification by providing:
- Detailed fruit metadata
- Nutritional information
- Botanical classifications

## 🚀 Future Enhancements

### Technical Improvements
- **Transfer Learning**: Implement pre-trained models (ResNet, VGG, EfficientNet)
- **Data Augmentation**: Advanced techniques (rotation, zoom, brightness)
- **Ensemble Methods**: Combine multiple models for higher accuracy
- **Mobile Deployment**: TensorFlow Lite conversion for mobile apps

### Feature Additions
- **Web Interface**: Flask/Django web application
- **Mobile App**: React Native or Flutter implementation
- **API Development**: RESTful API for integration
- **Database Integration**: Store predictions and user data

### Advanced Analytics
- **Confidence Thresholding**: Reject low-confidence predictions
- **Similarity Scoring**: Find similar fruits
- **Nutritional Calculator**: Estimate calories and nutrients
- **Recipe Suggestions**: Recommend recipes based on detected fruits

## 🏢 About Codec Technologies Internship

This project represents the culmination of deep learning training in the Codec Technologies internship program. As the second self-directed mini-project, it showcases:

- **Advanced technical skills** in deep learning and computer vision
- **Project management** capabilities in ML projects
- **Research and implementation** of state-of-the-art techniques
- **Documentation and presentation** of technical work

## 📊 Model Performance Visualization

The notebook includes comprehensive visualizations:

### Training Progress
- **Accuracy curves** (training vs validation)
- **Loss curves** over epochs
- **Learning rate scheduling** effects

### Prediction Analysis
- **Confidence score distributions**
- **Class prediction frequencies**
- **Error analysis** for misclassified images

### Data Insights
- **Class distribution** in dataset
- **Image quality** metrics
- **Dataset statistics** and characteristics

## 🎯 Business Applications

This fruit classifier has real-world applications in:

### Retail & E-commerce
- **Automated inventory** management
- **Product cataloging** for online stores
- **Quality control** in food processing

### Health & Nutrition
- **Dietary tracking** applications
- **Nutritional analysis** tools
- **Meal planning** assistants

### Agriculture & Research
- **Crop monitoring** systems
- **Botanical research** tools
- **Educational platforms** for fruit identification

## 📝 Conclusion

The Fruit Image Classifier represents a comprehensive exploration of deep learning in computer vision. By combining cutting-edge CNN architecture with practical deployment considerations, this project demonstrates the power of modern AI in solving real-world classification problems.

The project successfully bridges the gap between academic learning and practical implementation, providing a solid foundation for advanced computer vision projects and real-world AI applications.

---

**Note**: This project is built for educational and demonstration purposes as part of the Codec Technologies internship program. It showcases professional-level implementation of deep learning concepts with practical, deployable results.

## 🤝 Acknowledgments

- **Codec Technologies** for the comprehensive internship program
- **Fruits-360 Dataset** creators for the high-quality dataset
- **TensorFlow/Keras** community for excellent documentation
- **Open source community** for supporting libraries and tools