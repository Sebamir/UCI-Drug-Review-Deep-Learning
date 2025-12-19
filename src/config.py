import torch
import os

class Config:
    """
    Clase de configuración para el proyecto UCI Drug Review Deep Learning.
    Centraliza rutas, hiperparámetros y configuraciones de hardware.
    """

    def __init__(self):
        # --- 1. Rutas de Archivos ---
        
        # Rutas del modelo
        self.MODEL_NAME = 'distilbert-base-cased' # Modelo base de Hugging Face
<<<<<<< HEAD
        
        self.SAVE_MODEL_PATH = "Modelo_entrenado/modelo_final_consolidado"
        self.CHECKPOINT_PATH = "results/stage_2_frozen/checkpoint-9351" 

        # Rutas de Datos
        self.DATA_PATH = "Dataset/raw/temporal/drugsComTrain_raw.csv" 
        self.TEST_PATH = "Dataset/raw/temporal/drugsComTest_raw.csv"
=======
        # Aunque son iguales, es buena práctica separarlas si planeas hacer ajustes después del entrenamiento
        self.SAVE_MODEL_PATH = "Modelo_entrenado/drug_review_classifier_distilbert_fullbalance" 
        self.CHECKPOINT_PATH = "Modelo_entrenado/drug_review_classifier_distilbert_fullbalance" # Usar para cargar el modelo en modo predict

        # Rutas de Datos
        self.DATA_PATH = "Dataset/raw/temporal/drugsComTrain_raw.csv" 
        self.TRAIN_PATH = "Dataset/processed/train_dataset.csv" 
        self.TEST_PATH = "Dataset/processed/test_dataset.csv" 
>>>>>>> db0257491d425664f186c5f8882d56036e37c9e9
        
        # Directorios de Salida (para el Trainer de Hugging Face)
        self.OUTPUT_DIR_1 = os.path.join("results", "stage_1_unfrozen")
        self.LOGGING_DIR_1 = os.path.join("logs", "stage_1")
        self.OUTPUT_DIR_2 = os.path.join("results", "stage_2_frozen")
        self.LOGGING_DIR_2 = os.path.join("logs", "stage_2")
        
        # --- 2. Hiperparámetros del Modelo ---
        
        self.MAX_LEN = 256        # Longitud máxima de las secuencias de tokens
        self.NUM_LABELS = 2       # Número de etiquetas para clasificación (ej. Positivo, Negativo)
        
        # --- 3. Configuraciones de Entrenamiento por Etapa ---
        
        # Etapa 1: Entrenamiento Completo
        self.EPOCHS_1 = 2
        self.LEARNING_RATE_1 = 2e-5
        
        # Etapa 2: Fine-tuning de Capa de Clasificación
        self.EPOCHS_2 = 1
        self.LEARNING_RATE_2 = 1e-5
        
        # --- 4. Configuraciones Generales y Hardware ---

        self.SEED = 42           # Semilla para reproducibilidad
        self.TEST_SIZE = 0.2     # Tamaño del conjunto de prueba
        self.WARMUP_STEPS = 500  # Pasos de calentamiento
        self.OPTIMAL_THRESHOLD = 0.75 # Umbral (lo estás usando, lo dejamos aquí)
        
        # Corrección: Estas no deben ser tuplas al final
        self.PER_DEVICE_TRAIN_BATCH_SIZE = 8  # Tamaño del batch por dispositivo (GPU/CPU)
        self.PER_DEVICE_EVAL_BATCH_SIZE = 32  # Tamaño del batch para evaluación

        # Determinar el dispositivo (Se define en main.py, pero es útil tener una referencia aquí)
        # NOTA: Esta variable se sobrescribe en main.py para asegurar que el tensor weights y el modelo estén alineados.
        self.USE_GPU = torch.cuda.is_available()
        self.DEVICE = torch.device('cuda' if self.USE_GPU else 'cpu')


# Si accedes a esta clase desde main.py:
# config = Config()
# print(config.MODEL_NAME) # Acceso a la configuración