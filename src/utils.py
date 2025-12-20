import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

# 2. Transformers (Hugging Face)
from transformers import AutoTokenizer

# 3. Scikit-learn (Métricas y Utilidades)
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    f1_score, 
    precision_score, 
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src.config import Config


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
    
    return train_dataset, validation_dataset, weights



def run_detailed_evaluation(model, val_dataset):
    """
    Ejecuta una evaluación detallada del modelo con métricas y visualizaciones.
    
    Args:
        model: Modelo de PyTorch a evaluar
        val_dataset: Dataset de validación con input_ids, attention_mask y labels
        config: Objeto de configuración con DEVICE, PER_DEVICE_TEST_BATCH_SIZE, etc.
    
    Returns:
        dict: Diccionario con métricas de evaluación
    """
    print("\n--- 🔍 Iniciando Evaluación Detallada ---")
    
    # Validaciones iniciales
    if len(val_dataset) == 0:
        raise ValueError("El dataset de validación está vacío")
    
    # Asegurar que el modelo esté en el dispositivo correcto y en modo evaluación
    model.to(config.DEVICE)
    model.eval()
    
    # Configurar DataLoader con optimizaciones
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.PER_DEVICE_TEST_BATCH_SIZE,
        pin_memory=True,
        num_workers=0  # Evita problemas de serialización con datasets de HF
    )
    
    # Listas para acumular tensores (más eficiente que numpy en cada iteración)
    all_preds = []
    all_labels = []
    
    print(f"Procesando {len(val_dataset)} muestras en {len(val_loader)} batches...")
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Mover batch a GPU
                input_ids = batch['input_ids'].to(config.DEVICE)
                attention_mask = batch['attention_mask'].to(config.DEVICE)
                labels = batch['labels'].to(config.DEVICE)
                
                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask)
                
                # Obtener predicciones (clase con mayor probabilidad)
                preds = torch.argmax(outputs.logits, dim=1)
                
                # Acumular predicciones y labels (mantener en CPU)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                
                # Progreso cada 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Procesados {batch_idx + 1}/{len(val_loader)} batches")
        
        # Convertir a numpy una sola vez (más eficiente)
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
    except Exception as e:
        print(f"❌ Error durante la evaluación: {str(e)}")
        raise
    
    # Nombres de las clases 
    target_names = ['Negativo', 'Positivo']
    
    # 1. Reporte de Clasificación
    print("\n📊 Reporte de Clasificación:")
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=target_names,
        output_dict=True
    )
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # 2. Matriz de Confusión Visual
    print("\n📈 Generando matriz de confusión...")
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    
    plt.title("Matriz de Confusión: Sentimientos de Medicamentos", fontsize=14, pad=20)
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Etiqueta Real", fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.close()  # Liberar memoria
    
    # 3. Métricas adicionales
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    print(f"\n✅ Accuracy global: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Retornar métricas para logging o análisis posterior
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'true_labels': all_labels.tolist()
    }