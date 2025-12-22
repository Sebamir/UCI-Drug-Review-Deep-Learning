from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import json
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from datetime import datetime

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

def ProcessingTest(df):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    df_proccesado1 = df.query("rating >= 7 | rating <= 4").copy() # Filtramos solo las filas con rating >= 7 o <= 4
    df_proccesado1["flag"] = df_proccesado1["rating"].apply(lambda x: 1 if x >= 7 else 0) # Clasificamos los ratings en 1 (positivo) y 0 (negativo)
    df_proccesado1 = df_proccesado1[["review", "flag"]]
    # Tokenización
    encoding = tokenizer.batch_encode_plus(
        df_proccesado1["review"].tolist(),
        max_length=config.MAX_LEN,
        padding='max_length',
        truncation=True
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    labels = torch.tensor(df_proccesado1['flag'].values)    

    Test_dataset = Dataset.from_dict({
        'input_ids': input_ids, 
        'attention_mask': attention_mask,
        'labels': labels
    }) 

    return Test_dataset

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

def run_detailed_evaluation(model, val_dataset, output_pdf='evaluation_report.pdf'):
    """
    Ejecuta una evaluación detallada del modelo con métricas y visualizaciones.
    Genera un PDF con la matriz de confusión y el classification report.
    
    Args:
        model: Modelo de PyTorch a evaluar
        val_dataset: Dataset de validación con input_ids, attention_mask y labels
        output_pdf: Nombre del archivo PDF de salida
    
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
    
    # 2. Matriz de Confusión
    print("\n📈 Generando matriz de confusión...")
    cm = confusion_matrix(all_labels, all_preds)
    
    # 3. Accuracy
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    print(f"\n✅ Accuracy global: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # ==================== GENERAR PDF ====================
    print(f"\n📄 Generando PDF: {output_pdf}")
    
    with PdfPages(output_pdf) as pdf:
        # --- PÁGINA 1: Información General y Matriz de Confusión ---
        fig = plt.figure(figsize=(11, 14))
        
        # Título principal
        fig.suptitle('Reporte de Evaluación del Modelo', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Información general
        info_text = f"""
Fecha de evaluación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total de muestras: {len(val_dataset)}
Batch size: {config.PER_DEVICE_TEST_BATCH_SIZE}
Dispositivo: {config.DEVICE}
Threshold: {config.OPTIMAL_THRESHOLD}

Accuracy Global: {accuracy:.4f} ({accuracy*100:.2f}%)
        """
        
        ax_info = fig.add_subplot(3, 1, 1)
        ax_info.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                     fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax_info.axis('off')
        
        # Matriz de Confusión
        ax_cm = fig.add_subplot(3, 1, (2, 3))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax_cm, values_format='d', colorbar=False)
        ax_cm.set_title("Matriz de Confusión: Sentimientos de Medicamentos", 
                       fontsize=14, fontweight='bold', pad=15)
        ax_cm.set_xlabel("Predicción", fontsize=12)
        ax_cm.set_ylabel("Etiqueta Real", fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # --- PÁGINA 2: Classification Report ---
        fig = plt.figure(figsize=(11, 8))
        fig.suptitle('Classification Report Detallado', 
                     fontsize=16, fontweight='bold', y=0.96)
        
        # Convertir el reporte a DataFrame para mejor visualización
        report_df = pd.DataFrame(report).transpose()
        
        # Crear tabla con el reporte
        ax_report = fig.add_subplot(111)
        ax_report.axis('tight')
        ax_report.axis('off')
        
        # Formatear los valores numéricos
        report_display = report_df.copy()
        for col in ['precision', 'recall', 'f1-score']:
            if col in report_display.columns:
                report_display[col] = report_display[col].apply(lambda x: f'{x:.4f}' if isinstance(x, float) else x)
        if 'support' in report_display.columns:
            report_display['support'] = report_display['support'].apply(lambda x: f'{int(x)}' if isinstance(x, float) else x)
        
        # Crear la tabla
        table = ax_report.table(
            cellText=report_display.values,
            colLabels=report_display.columns,
            rowLabels=report_display.index,
            cellLoc='center',
            rowLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Estilizar encabezados
        for i in range(len(report_display.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Estilizar filas
        for i in range(len(report_display.index)):
            table[(i+1, -1)].set_facecolor('#D9E2F3')
            table[(i+1, -1)].set_text_props(weight='bold')
            
            # Colorear filas alternas
            if i % 2 == 0:
                for j in range(len(report_display.columns)):
                    table[(i+1, j)].set_facecolor('#F2F2F2')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Metadatos del PDF
        d = pdf.infodict()
        d['Title'] = 'Reporte de Evaluación del Modelo'
        d['Author'] = 'Sistema de Evaluación'
        d['Subject'] = 'Análisis de Sentimientos de Medicamentos'
        d['Keywords'] = 'Machine Learning, NLP, Evaluación'
        d['CreationDate'] = datetime.now()
    
    print(f"✅ PDF generado exitosamente: {output_pdf}")
    
    # Retornar métricas para logging o análisis posterior
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'true_labels': all_labels.tolist(),
        'pdf_path': output_pdf
    }

def find_latest_checkpoint(stage_dir):
    """
    Encuentra el último checkpoint en un directorio de stage.
    
    Args:
        stage_dir: Ruta al directorio del stage (ej: 'results/stage_1_unfrozen')
    
    Returns:
        Ruta completa al último checkpoint o None si no hay checkpoints
    """
    if not os.path.exists(stage_dir):
        print(f"⚠️ Advertencia: El directorio {stage_dir} no existe")
        return None
    
    # Buscar todas las carpetas que empiecen con 'checkpoint-'
    checkpoints = []
    for item in os.listdir(stage_dir):
        item_path = os.path.join(stage_dir, item)
        if os.path.isdir(item_path) and item.startswith('checkpoint-'):
            # Extraer el número del checkpoint
            match = re.search(r'checkpoint-(\d+)', item)
            if match:
                checkpoint_num = int(match.group(1))
                checkpoints.append((checkpoint_num, item_path))
    
    if not checkpoints:
        print(f"⚠️ Advertencia: No se encontraron checkpoints en {stage_dir}")
        return None
    
    # Ordenar por número y devolver el último
    checkpoints.sort(key=lambda x: x[0])
    latest_checkpoint = checkpoints[-1][1]
    
    print(f"✓ Último checkpoint encontrado en {stage_dir}: {os.path.basename(latest_checkpoint)}")
    return latest_checkpoint


def plot_loss_and_lr(*stage_configs):

    """
    Grafica las curvas de Loss y Learning Rate para múltiples stages.
    Automáticamente encuentra el último checkpoint de cada stage.
    
    Args:
        *stage_configs: Tuplas de (directorio_stage, label, color) para cada stage
                       Ejemplo: ('results/stage_1_unfrozen', 'Stage 1', 'blue')
                       También acepta solo (directorio_stage, label) o directorio_stage
    
    Ejemplos de uso:
        # Opción 1: Especificar todo
        plot_loss_and_lr(
            ('results/stage_1_unfrozen', 'Stage 1: Unfrozen', 'blue'),
            ('results/stage_2_frozen', 'Stage 2: Frozen', 'red')
        )
        
        # Opción 2: Sin colores (usa colores por defecto)
        plot_loss_and_lr(
            ('results/stage_1_unfrozen', 'Stage 1'),
            ('results/stage_2_frozen', 'Stage 2')
        )
        
        # Opción 3: Solo directorios (genera labels automáticamente)
        plot_loss_and_lr(
            'results/stage_1_unfrozen',
            'results/stage_2_frozen'
        )
    """
    if len(stage_configs) == 0:
        raise ValueError("Debes proporcionar al menos un directorio de stage")
    
    # Colores por defecto
    default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Procesar cada stage
    for idx, stage_info in enumerate(stage_configs):
        # Desempaquetar información
        if isinstance(stage_info, tuple):
            if len(stage_info) == 3:
                stage_dir, label, color = stage_info
            elif len(stage_info) == 2:
                stage_dir, label = stage_info
                color = default_colors[idx % len(default_colors)]
            else:
                stage_dir = stage_info[0]
                label = f'Stage {idx + 1}'
                color = default_colors[idx % len(default_colors)]
        else:
            # Si solo se pasa un directorio como string
            stage_dir = stage_info
            label = f'Stage {idx + 1}'
            color = default_colors[idx % len(default_colors)]
        
        # Encontrar el último checkpoint
        checkpoint_path = find_latest_checkpoint(stage_dir)
        if checkpoint_path is None:
            continue
        
        # Leer datos del checkpoint
        trainer_state_path = os.path.join(checkpoint_path, 'trainer_state.json')
        try:
            with open(trainer_state_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Advertencia: No se encontró {trainer_state_path}, omitiendo...")
            continue
        except json.JSONDecodeError:
            print(f"⚠️ Advertencia: Error al leer JSON en {trainer_state_path}, omitiendo...")
            continue
        
        history = data['log_history']
        steps = [e['step'] for e in history if 'loss' in e]
        loss = [e['loss'] for e in history if 'loss' in e]
        lrs = [e['learning_rate'] for e in history if 'learning_rate' in e]
        
        if not steps:
            print(f"⚠️ Advertencia: No se encontraron datos de entrenamiento en {checkpoint_path}")
            continue
        
        # Plotear en ambas gráficas
        ax1.plot(steps, loss, label=label, color=color, alpha=0.8, linewidth=2)
        ax2.plot(steps, lrs, label=label, color=color, alpha=0.8, linewidth=2)
    
    # --- Configurar Gráfica 1: Loss Functions ---
    ax1.set_title('Curvas de Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Steps (Pasos de Entrenamiento)', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # --- Configurar Gráfica 2: Learning Rate ---
    ax2.set_title('Learning Rate', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Steps (Pasos de Entrenamiento)', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()