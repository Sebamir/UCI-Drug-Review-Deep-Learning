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

def run_detailed_evaluation_max(model, val_dataset, output_pdf='evaluation_report.pdf'):
    """
    Ejecuta una evaluación detallada del modelo con métricas y visualizaciones.
    Incluye optimización de threshold por F1-score y Umbral de Youden.
    Genera un PDF con todos los resultados.
    
    Args:
        model: Modelo de PyTorch a evaluar
        val_dataset: Dataset de validación con input_ids, attention_mask y labels
        output_pdf: Nombre del archivo PDF de salida
    
    Returns:
        dict: Diccionario con métricas de evaluación y thresholds óptimos
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
        num_workers=0
    )
    
    # Listas para acumular predicciones y probabilidades
    all_preds = []
    all_labels = []
    all_probs = []  # ¡NUEVO! Para guardar probabilidades
    
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
                
                # Obtener probabilidades con softmax
                probs = torch.softmax(outputs.logits, dim=1)
                
                # Obtener predicciones (clase con mayor probabilidad)
                preds = torch.argmax(outputs.logits, dim=1)
                
                # Acumular predicciones, labels y probabilidades
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs[:, 1].cpu())  # Probabilidad de clase positiva
                
                # Progreso cada 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Procesados {batch_idx + 1}/{len(val_loader)} batches")
        
        # Convertir a numpy
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()
        
    except Exception as e:
        print(f"❌ Error durante la evaluación: {str(e)}")
        raise
    
    # ==================== OPTIMIZACIÓN DE THRESHOLDS ====================
    print("\n🎯 Optimizando thresholds...")
    
    # Definir rango de thresholds a probar
    thresholds = np.linspace(0.1, 0.9, 81)  # 81 puntos entre 0.1 y 0.9
    
    # Arrays para guardar métricas
    f1_scores = []
    youden_scores = []
    precisions = []
    recalls = []
    specificities = []
    
    for threshold in thresholds:
        # Aplicar threshold
        preds_thresh = (all_probs >= threshold).astype(int)
        
        # Calcular métricas
        tn, fp, fn, tp = confusion_matrix(all_labels, preds_thresh).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        youden = recall + specificity - 1  # Índice de Youden
        
        f1_scores.append(f1)
        youden_scores.append(youden)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
    
    # Encontrar thresholds óptimos
    best_f1_idx = np.argmax(f1_scores)
    best_youden_idx = np.argmax(youden_scores)
    
    best_f1_threshold = thresholds[best_f1_idx]
    best_youden_threshold = thresholds[best_youden_idx]
    
    print(f"✅ Mejor threshold por F1-Score: {best_f1_threshold:.3f} (F1={f1_scores[best_f1_idx]:.4f})")
    print(f"✅ Mejor threshold por Youden: {best_youden_threshold:.3f} (Youden={youden_scores[best_youden_idx]:.4f})")
    
    # Recalcular métricas con thresholds óptimos
    preds_f1 = (all_probs >= best_f1_threshold).astype(int)
    preds_youden = (all_probs >= best_youden_threshold).astype(int)
    
    # Nombres de las clases 
    target_names = ['Negativo', 'Positivo']
    
    # ==================== MÉTRICAS CON THRESHOLD DEFAULT ====================
    print("\n📊 Reporte con Threshold por Defecto (0.5):")
    preds_default = (all_probs >= 0.5).astype(int)
    report_default = classification_report(all_labels, preds_default, target_names=target_names, output_dict=True)
    print(classification_report(all_labels, preds_default, target_names=target_names))
    cm_default = confusion_matrix(all_labels, preds_default)
    
    # ==================== MÉTRICAS CON THRESHOLD ÓPTIMO F1 ====================
    print(f"\n📊 Reporte con Threshold Óptimo F1 ({best_f1_threshold:.3f}):")
    report_f1 = classification_report(all_labels, preds_f1, target_names=target_names, output_dict=True)
    print(classification_report(all_labels, preds_f1, target_names=target_names))
    cm_f1 = confusion_matrix(all_labels, preds_f1)
    
    # ==================== MÉTRICAS CON THRESHOLD ÓPTIMO YOUDEN ====================
    print(f"\n📊 Reporte con Threshold Óptimo Youden ({best_youden_threshold:.3f}):")
    report_youden = classification_report(all_labels, preds_youden, target_names=target_names, output_dict=True)
    print(classification_report(all_labels, preds_youden, target_names=target_names))
    cm_youden = confusion_matrix(all_labels, preds_youden)
    
    # Accuracy
    accuracy_default = (preds_default == all_labels).sum() / len(all_labels)
    accuracy_f1 = (preds_f1 == all_labels).sum() / len(all_labels)
    accuracy_youden = (preds_youden == all_labels).sum() / len(all_labels)
    
    # ==================== GENERAR PDF ====================
    print(f"\n📄 Generando PDF: {output_pdf}")
    
    with PdfPages(output_pdf) as pdf:
        # --- PÁGINA 1: Información General ---
        fig = plt.figure(figsize=(11, 14))
        fig.suptitle('Reporte de Evaluación del Modelo con Optimización de Threshold', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        info_text = f"""
Fecha de evaluación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total de muestras: {len(val_dataset)}
Batch size: {config.PER_DEVICE_TEST_BATCH_SIZE}
Dispositivo: {config.DEVICE}

═══════════════════════════════════════════════════════════════
                    RESULTADOS DE OPTIMIZACIÓN
═══════════════════════════════════════════════════════════════

Threshold por Defecto:           0.500
  → Accuracy: {accuracy_default:.4f} ({accuracy_default*100:.2f}%)
  → F1-Score: {report_default['weighted avg']['f1-score']:.4f}

Threshold Óptimo F1-Score:       {best_f1_threshold:.3f}
  → Accuracy: {accuracy_f1:.4f} ({accuracy_f1*100:.2f}%)
  → F1-Score: {report_f1['weighted avg']['f1-score']:.4f}
  → Mejora F1: {(report_f1['weighted avg']['f1-score'] - report_default['weighted avg']['f1-score'])*100:+.2f}%

Threshold Óptimo Youden:         {best_youden_threshold:.3f}
  → Accuracy: {accuracy_youden:.4f} ({accuracy_youden*100:.2f}%)
  → F1-Score: {report_youden['weighted avg']['f1-score']:.4f}
  → Índice Youden: {youden_scores[best_youden_idx]:.4f}
        """
        
        ax_info = fig.add_subplot(111)
        ax_info.text(0.05, 0.5, info_text, fontsize=11, verticalalignment='center',
                     fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax_info.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # --- PÁGINA 2: Curvas de Optimización ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Análisis de Optimización de Threshold', fontsize=16, fontweight='bold')
        
        # Subplot 1: F1-Score vs Threshold
        axes[0, 0].plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1-Score')
        axes[0, 0].axvline(best_f1_threshold, color='r', linestyle='--', linewidth=2, label=f'Óptimo: {best_f1_threshold:.3f}')
        axes[0, 0].scatter([best_f1_threshold], [f1_scores[best_f1_idx]], color='r', s=100, zorder=5)
        axes[0, 0].set_xlabel('Threshold', fontsize=11)
        axes[0, 0].set_ylabel('F1-Score', fontsize=11)
        axes[0, 0].set_title('F1-Score vs Threshold', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Subplot 2: Índice de Youden vs Threshold
        axes[0, 1].plot(thresholds, youden_scores, 'g-', linewidth=2, label='Índice de Youden')
        axes[0, 1].axvline(best_youden_threshold, color='r', linestyle='--', linewidth=2, label=f'Óptimo: {best_youden_threshold:.3f}')
        axes[0, 1].scatter([best_youden_threshold], [youden_scores[best_youden_idx]], color='r', s=100, zorder=5)
        axes[0, 1].set_xlabel('Threshold', fontsize=11)
        axes[0, 1].set_ylabel('Índice de Youden', fontsize=11)
        axes[0, 1].set_title('Índice de Youden vs Threshold', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Subplot 3: Precision, Recall, Specificity
        axes[1, 0].plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
        axes[1, 0].plot(thresholds, recalls, 'g-', linewidth=2, label='Recall')
        axes[1, 0].plot(thresholds, specificities, 'orange', linewidth=2, label='Specificity')
        axes[1, 0].axvline(best_f1_threshold, color='r', linestyle='--', alpha=0.5, label=f'F1 Óptimo')
        axes[1, 0].axvline(best_youden_threshold, color='purple', linestyle='--', alpha=0.5, label=f'Youden Óptimo')
        axes[1, 0].set_xlabel('Threshold', fontsize=11)
        axes[1, 0].set_ylabel('Score', fontsize=11)
        axes[1, 0].set_title('Métricas vs Threshold', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Subplot 4: Comparación de Métricas
        metrics_comparison = {
            'Default\n(0.5)': [accuracy_default, report_default['weighted avg']['f1-score'], 
                               report_default['weighted avg']['precision'], report_default['weighted avg']['recall']],
            f'F1 Óptimo\n({best_f1_threshold:.3f})': [accuracy_f1, report_f1['weighted avg']['f1-score'],
                                                        report_f1['weighted avg']['precision'], report_f1['weighted avg']['recall']],
            f'Youden\n({best_youden_threshold:.3f})': [accuracy_youden, report_youden['weighted avg']['f1-score'],
                                                         report_youden['weighted avg']['precision'], report_youden['weighted avg']['recall']]
        }
        
        x = np.arange(len(metrics_comparison))
        width = 0.2
        metrics_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics_names):
            values = [metrics_comparison[k][i] for k in metrics_comparison.keys()]
            axes[1, 1].bar(x + i*width, values, width, label=metric, color=colors[i])
        
        axes[1, 1].set_xlabel('Threshold', fontsize=11)
        axes[1, 1].set_ylabel('Score', fontsize=11)
        axes[1, 1].set_title('Comparación de Métricas por Threshold', fontweight='bold')
        axes[1, 1].set_xticks(x + width * 1.5)
        axes[1, 1].set_xticklabels(metrics_comparison.keys())
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # --- PÁGINA 3: Matrices de Confusión ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Matrices de Confusión - Comparación de Thresholds', fontsize=16, fontweight='bold')
        
        # Matriz 1: Default
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_default, display_labels=target_names)
        disp1.plot(cmap=plt.cm.Blues, ax=axes[0], values_format='d', colorbar=False)
        axes[0].set_title(f'Threshold Default (0.5)\nAcc: {accuracy_default:.4f}', fontweight='bold')
        
        # Matriz 2: F1 Óptimo
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_f1, display_labels=target_names)
        disp2.plot(cmap=plt.cm.Greens, ax=axes[1], values_format='d', colorbar=False)
        axes[1].set_title(f'Threshold F1 Óptimo ({best_f1_threshold:.3f})\nAcc: {accuracy_f1:.4f}', fontweight='bold')
        
        # Matriz 3: Youden Óptimo
        disp3 = ConfusionMatrixDisplay(confusion_matrix=cm_youden, display_labels=target_names)
        disp3.plot(cmap=plt.cm.Oranges, ax=axes[2], values_format='d', colorbar=False)
        axes[2].set_title(f'Threshold Youden ({best_youden_threshold:.3f})\nAcc: {accuracy_youden:.4f}', fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # --- PÁGINA 4: Classification Report - Default ---
        fig = plt.figure(figsize=(11, 8))
        fig.suptitle('Classification Report - Threshold Default (0.5)', fontsize=16, fontweight='bold', y=0.96)
        
        report_df = pd.DataFrame(report_default).transpose()
        ax_report = fig.add_subplot(111)
        ax_report.axis('tight')
        ax_report.axis('off')
        
        report_display = report_df.copy()
        for col in ['precision', 'recall', 'f1-score']:
            if col in report_display.columns:
                report_display[col] = report_display[col].apply(lambda x: f'{x:.4f}' if isinstance(x, float) else x)
        if 'support' in report_display.columns:
            report_display['support'] = report_display['support'].apply(lambda x: f'{int(x)}' if isinstance(x, float) else x)
        
        table = ax_report.table(cellText=report_display.values, colLabels=report_display.columns,
                               rowLabels=report_display.index, cellLoc='center', rowLoc='center',
                               loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(report_display.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # --- PÁGINA 5: Classification Report - F1 Óptimo ---
        fig = plt.figure(figsize=(11, 8))
        fig.suptitle(f'Classification Report - Threshold F1 Óptimo ({best_f1_threshold:.3f})', 
                     fontsize=16, fontweight='bold', y=0.96)
        
        report_df = pd.DataFrame(report_f1).transpose()
        ax_report = fig.add_subplot(111)
        ax_report.axis('tight')
        ax_report.axis('off')
        
        report_display = report_df.copy()
        for col in ['precision', 'recall', 'f1-score']:
            if col in report_display.columns:
                report_display[col] = report_display[col].apply(lambda x: f'{x:.4f}' if isinstance(x, float) else x)
        if 'support' in report_display.columns:
            report_display['support'] = report_display['support'].apply(lambda x: f'{int(x)}' if isinstance(x, float) else x)
        
        table = ax_report.table(cellText=report_display.values, colLabels=report_display.columns,
                               rowLabels=report_display.index, cellLoc='center', rowLoc='center',
                               loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(report_display.columns)):
            table[(0, i)].set_facecolor('#2ca02c')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # --- PÁGINA 6: Classification Report - Youden Óptimo ---
        fig = plt.figure(figsize=(11, 8))
        fig.suptitle(f'Classification Report - Threshold Youden ({best_youden_threshold:.3f})', 
                     fontsize=16, fontweight='bold', y=0.96)
        
        report_df = pd.DataFrame(report_youden).transpose()
        ax_report = fig.add_subplot(111)
        ax_report.axis('tight')
        ax_report.axis('off')
        
        report_display = report_df.copy()
        for col in ['precision', 'recall', 'f1-score']:
            if col in report_display.columns:
                report_display[col] = report_display[col].apply(lambda x: f'{x:.4f}' if isinstance(x, float) else x)
        if 'support' in report_display.columns:
            report_display['support'] = report_display['support'].apply(lambda x: f'{int(x)}' if isinstance(x, float) else x)
        
        table = ax_report.table(cellText=report_display.values, colLabels=report_display.columns,
                               rowLabels=report_display.index, cellLoc='center', rowLoc='center',
                               loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(report_display.columns)):
            table[(0, i)].set_facecolor('#ff7f0e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Metadatos del PDF
        d = pdf.infodict()
        d['Title'] = 'Reporte de Evaluación con Optimización de Threshold'
        d['Author'] = 'Sistema de Evaluación'
        d['Subject'] = 'Análisis de Sentimientos de Medicamentos'
        d['Keywords'] = 'Machine Learning, NLP, Threshold Optimization, Youden'
        d['CreationDate'] = datetime.now()
    
    print(f"✅ PDF generado exitosamente: {output_pdf}")
    
    # Retornar todas las métricas
    return {
        'default_threshold': {
            'threshold': 0.5,
            'accuracy': accuracy_default,
            'classification_report': report_default,
            'confusion_matrix': cm_default.tolist()
        },
        'optimal_f1': {
            'threshold': float(best_f1_threshold),
            'accuracy': accuracy_f1,
            'f1_score': f1_scores[best_f1_idx],
            'classification_report': report_f1,
            'confusion_matrix': cm_f1.tolist()
        },
        'optimal_youden': {
            'threshold': float(best_youden_threshold),
            'accuracy': accuracy_youden,
            'youden_index': youden_scores[best_youden_idx],
            'classification_report': report_youden,
            'confusion_matrix': cm_youden.tolist()
        },
        'threshold_analysis': {
            'thresholds': thresholds.tolist(),
            'f1_scores': f1_scores,
            'youden_scores': youden_scores
        },
        'predictions': all_preds.tolist(),
        'probabilities': all_probs.tolist(),
        'true_labels': all_labels.tolist(),
        'pdf_path': output_pdf
    }

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