# NLP-CNN Sentiment Analysis

This project implements a Convolutional Neural Network (CNN) for sentiment analysis using text data. The model is designed to classify the sentiment (positive or negative) from text reviews, leveraging Natural Language Processing (NLP) techniques and deep learning.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Results](#results)
9. [Contributors](#contributors)
10. [License](#license)
11. [Let's Connect](#lets-connect)

## Project Overview

In this project, we explore sentiment analysis on textual data using CNNs, a deep learning architecture typically used for image processing but adapted here for NLP tasks. The model processes text data, which is tokenized and embedded before being passed through the CNN layers. This project demonstrates how CNNs can be effectively utilized for text classification tasks like sentiment analysis.

The pipeline includes:
- Text preprocessing
- Word embedding (e.g., Word2Vec, GloVe)
- CNN model implementation
- Model training and evaluation on sentiment-labeled datasets

## Dataset

The dataset used in this project is a collection of text reviews, each labeled as either positive or negative. The text data includes features like:
- **Text**: The raw text of the review.
- **Sentiment**: A binary label (1 for positive, 0 for negative).

The dataset is preprocessed and stored in the `data/` directory for training and testing the model.

## Data Preprocessing

Before feeding the data into the model, the following preprocessing steps are performed:
- Tokenization of text data to break the text into individual words.
- Removal of stopwords, punctuation, and unnecessary symbols.
- Word embeddings are generated using pre-trained embeddings like Word2Vec or GloVe.
- Padding of sequences to ensure uniform input length for the CNN model.

This ensures the data is clean and structured for efficient model training.

## Modeling

The CNN model consists of the following key layers:
1. **Embedding Layer**: Converts words into vectors using pre-trained embeddings.
2. **Convolutional Layer**: Extracts n-gram features from the text by applying convolutional filters.
3. **Max-Pooling Layer**: Reduces dimensionality by selecting the most important features.
4. **Fully Connected Layer**: Maps the output to the final sentiment classes (positive or negative).

Hyperparameter tuning is performed to optimize the model's performance.

## Evaluation

The performance of the model is evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

These metrics help in understanding the model's effectiveness in classifying sentiments correctly.

## Installation

To run this project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/3m0r9/NLP-CNN-Sentiment.git
   ```
2. Navigate to the project directory:
   ```bash
   cd NLP-CNN-Sentiment
   ```
3. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
4. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Preprocess the dataset:
   ```bash
   python preprocess_data.py --input data/raw_reviews.csv --output data/processed_reviews.csv
   ```
2. Train the CNN model:
   ```bash
   python train_model.py --input data/processed_reviews.csv
   ```
3. To evaluate the model on test data:
   ```bash
   python evaluate_model.py --input data/test_reviews.csv
   ```

## Results

The CNN model achieved the following performance on the test set:
- **Accuracy**: 88%
- **Precision**: 0.86
- **Recall**: 0.89
- **F1-score**: 0.87

Further details and visualizations (e.g., confusion matrix, loss/accuracy plots) can be found in the `results/` directory.

## Contributors

- **Imran Abu Libda** - [3m0r9](https://github.com/3m0r9)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Let's Connect

- **GitHub** - [3m0r9](https://github.com/3m0r9)
- **LinkedIn** - [Imran Abu Libda](https://www.linkedin.com/in/imran-abu-libda/)
- **Email** - [imranabulibda@gmail.com](mailto:imranabulibda@gmail.com)
- **Medium** - [Imran Abu Libda](https://medium.com/@imranabulibda_23845)

