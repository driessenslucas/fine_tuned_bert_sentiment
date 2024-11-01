import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Set device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(DEVICE)
model.eval()

def classify_texts(texts):
    # Tokenize the input texts
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    input_ids = encodings['input_ids'].to(DEVICE)
    attention_mask = encodings['attention_mask'].to(DEVICE)

    # Run inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    return ["Positive" if pred == 1 else "Negative" for pred in predictions]

if __name__ == "__main__":
    # Example texts to classify
    texts = [
        "This movie was fantastic! The plot was gripping and the characters were well-developed.",
        "I didn't like this movie. It was too slow and the storyline was predictable."
    ]

    # Get predictions
    predictions = classify_texts(texts)

    # Print results
    for text, pred in zip(texts, predictions):
        print(f"Review: '{text}'\nSentiment: {pred}\n")
