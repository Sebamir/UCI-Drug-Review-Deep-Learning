import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import Config 
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
<<<<<<< HEAD
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
=======
>>>>>>> db0257491d425664f186c5f8882d56036e37c9e9

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

    inputs = {key: value.to(config.DEVICE) for key, value in inputs.items()}

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
    
    df_clasificado = df.query("rating >= 7 | rating <= 4").copy() # Filtramos solo las filas con rating >= 7 o <= 4
    df_clasificado["flag"] = df_clasificado["rating"].apply(lambda x: 1 if x >= 7 else 0) # Clasificamos los ratings en 1 (positivo) y 0 (negativo)

    df_clasificado = df_clasificado[["review", "flag"]]

    # Separar clases
    clase_0 = df_clasificado[df_clasificado["flag"] == 0]
    clase_1 = df_clasificado[df_clasificado["flag"] == 1]
    
    # Submuestreo
    clase_0_subsampled = clase_0.sample(frac=0.5, random_state=config.SEED)
    clase_1_subsampled = clase_1.sample(frac=0.5, random_state=config.SEED)

    # Duplicar clase minoritaria para balancear
    clase_0_duplicated = pd.concat([clase_0_subsampled, clase_0_subsampled])
    
    # Dataset balanceado final
    df_balanced = pd.concat([clase_0_duplicated, clase_1_subsampled]).sample(
        frac=1, random_state=config.SEED    
    ).reset_index(drop=True)
    
    # Calcular pesos sobre los datos FINALES balanceados
    labels_for_weights = df_balanced['flag'].values
    clase_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels_for_weights),
        y=labels_for_weights
    )
    weights = torch.tensor(clase_weights, dtype=torch.float32).to(config.DEVICE)
    
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
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
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
    
<<<<<<< HEAD
    return train_dataset, validation_dataset, weights



def run_detailed_evaluation(model, val_dataset):
    print("\n--- 🔍 Iniciando Evaluación Detallada ---")
    
    # Configurar el entorno para evaluación
    model.eval()

    all_preds = []
    all_labels = []

    # Extraer etiquetas y predicciones del dataset de validación
    print("Procesando predicciones...")
    for batch in val_dataset:
        # Preparar datos (esto asume que tu dataset devuelve tensores)
        input_ids = torch.tensor(batch['input_ids']).unsqueeze(0).to(config.DEVICE)
        attention_mask = torch.tensor(batch['attention_mask']).unsqueeze(0).to(config.DEVICE)
        label = batch['labels']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            
        all_preds.append(prediction)
        all_labels.append(label)

    # 1. Reporte de Clasificación (Precisión, Recall, F1)
    target_names = ['Negativo', 'Neutral', 'Positivo']
    print("\n📊 Reporte de Clasificación:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # 2. Matriz de Confusión Visual
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Matriz de Confusión: Sentimientos de Medicamentos")
    plt.show()
=======
    return train_dataset, validation_dataset, weights
>>>>>>> db0257491d425664f186c5f8882d56036e37c9e9
