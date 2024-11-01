# Sentiment Analysis with DistilBERT Fine-Tuning

This project demonstrates how to fine-tune BERT, specifically DistilBERT, for sentiment analysis on IMDb movie reviews. The fine-tuned model classifies movie reviews as positive or negative.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
3. [Dataset](#dataset)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Testing](#testing)
6. [License](#license)

---

### Project Overview
This repository showcases the steps required to fine-tune the DistilBERT model from Hugging Face’s `transformers` library for sentiment analysis. DistilBERT is a smaller, faster, and efficient version of BERT, making it ideal for quick and effective fine-tuning tasks.

### Setup Instructions

1. **Clone this repository:**
   ```bash
   git clone https://github.com/driessenslucas/fine_tuned_bert_sentiment.git
   cd fine_tuned_bert_sentiment
   ```

2. **Install dependencies:**
   Install the required libraries:
   ```bash
   pip install torch transformers pandas requests
   ```

### Dataset
The IMDb movie reviews dataset is used for training and evaluating the model. The dataset is available in compressed CSV format (`movie_data.csv.gz`). It is split as follows:
- **Training Set**: 35,000 reviews
- **Validation Set**: 5,000 reviews
- **Test Set**: 10,000 reviews

Each row in the dataset contains:
- `review`: The movie review text
- `sentiment`: Sentiment label (0 for negative, 1 for positive)

### Model Training and Evaluation
The core model is DistilBERT, which is fine-tuned to adapt to the specific task of sentiment analysis. Here’s a summary of the key components:

1. **Tokenizer**: `DistilBertTokenizerFast` is used to tokenize the review texts, preparing them as input to the model.

2. **Custom Dataset**: A PyTorch `Dataset` class is created to handle the tokenized input and sentiment labels, and a `DataLoader` batches the data.

3. **Training Loop**: The model is trained for 3 epochs with the Adam optimizer and a learning rate of `5e-5`. Training is performed on the IMDb dataset to optimize for sentiment classification.

4. **Evaluation**: Accuracy is measured on training, validation, and test datasets.

### Testing
To test the fine-tuned model on new samples, a separate script `test.py` is provided. This script loads the saved model and tokenizer, allowing you to input text reviews for sentiment prediction.

**Example Usage of `test.py`:**
```python
# test.py

# Sample input texts
texts = [
    "This movie was fantastic! The plot was gripping and the characters were well-developed.",
    "I didn't like this movie. It was too slow and the storyline was predictable."
]

# Run the script
python test.py
```

The output will display each review along with its predicted sentiment.
