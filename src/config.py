# --- Rutas de Archivos ---

MODEL_NAME = 'distilbert-base-cased' #Modelo base
MODEL_PATH = "Modelo_entrenado/drug_review_classifier_distilbert_fullbalance" #Ruta para guardar el modelo entrenado
DATA_PATH = "Dataset/raw/temporal/drugsComTrain_raw.csv" #Ruta del conjunto de datos

# --- Hiperparámetros ---
MAX_LEN = 256 #Longitud máxima de las secuencias de tokens
seed = 42 #Semilla para reproducibilidad
test_size=0.2 #Tamaño del conjunto de prueba
BATCH_SIZE=16 #Tamaño del lote para el entrenamiento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Dispositivo para entrenamiento (GPU o CPU)
EPOCHS_1=2 #Número de épocas para el primer ciclo de entrenamiento
EPOCHS_2=1 #Número de épocas para el segundo ciclo de entrenamiento
num_labels = 2 #Número de etiquetas para clasificación
LEARNING_RATE_1=2e-5 #Tasa de aprendizaje para el primer ciclo de entrenamiento
LEARNING_RATE_2=1e-5 #Tasa de aprendizaje para el segundo ciclo
OPTIMAL_THRESHOLD=0.75 #Umbral óptimo para clasificación binaria
warmup_steps=500 #Número de pasos de calentamiento para el programador de tasa de aprendizaje
per_device_train_batch_size=8,  # Tamaño del batch por dispositivo (GPU/CPU)
per_device_eval_batch_size=32,   # Tamaño del batch para evaluación
CLASS_WEIGHTS = torch.tensor([1.1666, 0.8750]).to(device) #Pesos de clase para manejar el desbalanceo