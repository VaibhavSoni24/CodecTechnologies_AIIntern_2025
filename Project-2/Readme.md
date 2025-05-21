# Sentence Prediction with LSTM

## Project Overview

This project implements a language model that can predict and generate natural language sentences based on an initial prompt. The model is built using PyTorch and trained on VlogBrothers scripts data. The system processes text data, builds a vocabulary, trains a neural network, and generates new sentences that mimic the style and content of the training data.

## Features

- Data loading and processing from online text sources
- Text tokenization and preprocessing with SpaCy
- Word frequency analysis and visualization
- Custom text simplification and normalization
- Handling of rare words by substituting them with "UNK" tokens
- Training of an LSTM-based encoder-decoder neural network
- Text generation with both deterministic and sampling-based approaches
- Temperature-based sampling to control generation creativity

## Requirements

- Python 3.6+
- PyTorch
- SpaCy (with 'en' language model)
- NumPy
- Matplotlib
- Seaborn
- CUDA-capable GPU (recommended for faster training)

## Project Structure

The project is organized in a single Jupyter notebook file (`Sentence_prediction.ipynb`) with the following sections:

1. **Data Loading and Preprocessing**
   - Loading text data
   - Tokenization
   - Text cleaning and normalization

2. **Data Analysis**
   - Word frequency distribution
   - Vocabulary size analysis
   - Rare word identification

3. **Model Architecture**
   - Embedding layer for word vector representation
   - LSTM-based recurrent neural network
   - Decoder network for word prediction

4. **Model Training**
   - Batch preparation
   - Loss calculation
   - Learning rate adjustment
   - Training and validation loops

5. **Text Generation**
   - Deterministic (argmax) generation
   - Sampling-based generation with temperature control
   - Probability ranking of generated sentences

## Usage

1. Open the Jupyter notebook `Sentence_prediction.ipynb`
2. Run all cells in order
3. To customize text generation:
   - Modify the `prefix` variable to change the starting prompt
   - Adjust `words_to_generate` to control output length
   - Change the temperature value (higher for more creative, lower for more deterministic output)

## Customization Options

Throughout the notebook, there are several parameters marked with `# CHANGEME` or `# ADVANCED_CHANGEME` comments that can be modified:

- **Input text source**: Replace the URL with another text source
- **Sequence length**: Change the context window size for prediction
- **Model architecture**: Adjust embedding size and hidden layer dimensions
- **Training parameters**: Modify batch size, learning rate, and number of epochs
- **Text generation**: Change the starting prompt, output length, and sampling temperature

## Results

The trained model can generate coherent sentences that follow the style and content patterns of the VlogBrothers scripts. The notebook displays both the highest probability (argmax) generation and multiple sampled outputs with their associated probabilities.

## Future Improvements

- Save and load model weights for reuse
- Implement a more sophisticated tokenization approach
- Add character-level modeling for better handling of rare words
- Incorporate attention mechanisms for improved context awareness
- Fine-tune hyperparameters for better performance

---

*Created as part of the Codec Technologies Internship Project-2*