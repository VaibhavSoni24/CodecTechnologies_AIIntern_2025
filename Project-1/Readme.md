# Handwritten Letter Recognition using Neural Networks

## Project Overview
This project implements a Multi-Layer Perceptron (MLP) neural network to recognize handwritten letters using the EMNIST dataset. The system is trained to identify lowercase English alphabet characters and can be used to convert handwritten text into typed text.

## Dataset
The project uses the **Extended Modified National Institute of Standards and Technology (EMNIST)** dataset, which is a set of handwritten character digits derived from the NIST Special Database 19. The dataset contains:
- 60,000 training images
- 10,000 testing images
- Each image is a 28x28 grayscale representation of a letter

## Project Structure
The project is organized in the following steps:

1. **Data Preparation**
   - Loading the EMNIST dataset
   - Normalizing pixel values (0-1)
   - Splitting into training and testing sets
   - Flattening 28x28 images into 784-element vectors

2. **Model Creation and Training**
   - Implementing a basic MLP with a single hidden layer (50 neurons)
   - Training the model on the training dataset
   - Evaluating model performance on both training and testing datasets

3. **Model Evaluation and Visualization**
   - Visualizing prediction errors using confusion matrices
   - Exploring specific misclassification examples (e.g., 'i' vs 'l')
   - Creating a more complex MLP with multiple hidden layers for comparison

4. **Real-world Application**
   - Processing scanned handwritten texts
   - Applying image preprocessing techniques (Gaussian blur, ROI extraction)
   - Converting handwritten text to typed format
   - Adding space detection for improved readability

## Libraries Used
- **emnist**: For loading the EMNIST dataset
- **matplotlib**: For data visualization
- **scikit-learn**: For implementing the MLP classifier
- **numpy**: For array operations and data manipulation
- **cv2** (OpenCV): For image processing and transformations

## How to Use
1. Run the notebook cells in sequence
2. Experiment with different MLP configurations by modifying hyperparameters
3. Test the model on your own handwritten letters by:
   - Adding images to the input directory
   - Following the preprocessing steps
   - Running the prediction on your images

## Results
The project demonstrates:
- How neural networks can be trained to recognize handwritten characters
- The impact of neural network architecture on performance 
- Real-world application of letter recognition to convert handwritten text to typed text
- Preprocessing techniques to improve recognition accuracy

## References
- EMNIST Dataset: https://arxiv.org/abs/1702.05373v1
- Python MNIST: https://github.com/sorki/python-mnist
- Neural Network Sample Code: https://github.com/crash-course-ai/lab1-neural-networks