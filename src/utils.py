import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import Config 
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

config = Config()

def compute_metrics(eval_pred):
    """
    Calcula las métricas de precisión (accuracy) y F1-score.
    """
    # predictions son los logits (salida cruda del modelo)
    logits, labels = eval_pred 
    
    # Tomar el argmax para obtener la clase predicha (0 o 1)
    predictions = np.argmax(logits, axis=-1)
    
    # Calcular las métricas
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary') 
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def predict_sentiment_threshold(text, model, tokenizer, threshold=config.OPTIMAL_THRESHOLD):
    """
    Realiza la predicción de sentimiento (positivo/negativo) en base a un umbral dado.
    """
    # Tokenizar el texto de entrada
    inputs = tokenizer(
        text,
        max_length=config.MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    inputs = {key: value.to(config.device) for key, value in inputs.items()}
    
    # Modo evaluación
    model.eval()
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        positive_prob = probabilities[0, 1].item()  # Probabilidad de la clase positiva
        negative_prob = probabilities[0, 0].item()  # Probabilidad de la clase negativa
    
    # Aplicar el umbral para decidir la clase
    if positive_prob >= threshold:
        prediction_class = "Positive"
    else:
        prediction_class = "Negative"
    
    return prediction_class, positive_prob, negative_prob

def ProcessingDataframe(df):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Separar clases
    clase_0 = df[df["flag"] == 0]
    clase_1 = df[df["flag"] == 1]
    
    # Submuestreo
    clase_0_subsampled = clase_0.sample(frac=0.5, random_state=config.seed)
    clase_1_subsampled = clase_1.sample(frac=0.5, random_state=config.seed)
    
    # Duplicar clase minoritaria para balancear
    clase_0_duplicated = pd.concat([clase_0_subsampled, clase_0_subsampled])
    
    # Dataset balanceado final
    df_balanced = pd.concat([clase_0_duplicated, clase_1_subsampled]).sample(
        frac=1, random_state=config.seed
    ).reset_index(drop=True)
    
    # Calcular pesos sobre los datos FINALES balanceados
    labels_for_weights = df_balanced['flag'].values
    clase_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels_for_weights),
        y=labels_for_weights
    )
    weights = torch.tensor(clase_weights, dtype=torch.float32).to(config.device)
    
    # Tokenización
    encoding = tokenizer.batch_encode_plus(
        df_balanced['review'].tolist(),
        max_length=config.MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    labels = torch.tensor(df_balanced['flag'].values)
    
    # División train/validation 
    train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
        input_ids, attention_mask, labels,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=labels
    )
    
    # Crear datasets
    train_dataset = Dataset.from_dict({
        'input_ids': train_inputs,
        'attention_mask': train_masks,
        'labels': train_labels
    })
    
    validation_dataset = Dataset.from_dict({
        'input_ids': val_inputs,
        'attention_mask': val_masks,
        'labels': val_labels
    })
    
    return train_dataset, validation_dataset, weights