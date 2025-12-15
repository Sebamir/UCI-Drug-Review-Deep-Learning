import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, AdamWeightDecay, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import src.config as config


class BasicTrainer(Trainer):
    """
    Trainer normal sin inyeccion de pesos (CrossEntropyLoss).
    """
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        # Obtener las etiquetas (y_true)
        labels = inputs.pop("labels")
        
        # Propagación hacia adelante (Forward pass)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Crear la función de pérdida CrossEntropy sin pesos de clase
        loss_fct = nn.CrossEntropyLoss() 
        
        # Calcular la pérdida
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class WeightedTrainer(Trainer):
    """
    Trainer personalizado que inyecta pesos de clase en la función de pérdida (CrossEntropyLoss).
    """
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        # Obtener las etiquetas (y_true)
        labels = inputs.pop("labels")
        
        # Propagación hacia adelante (Forward pass)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Crear la función de pérdida CrossEntropy con los pesos de clase
        loss_fct = nn.CrossEntropyLoss(weight=config.CLASS_WEIGHTS) # 
        
        # Calcular la pérdida
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

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

def full_training(train_dataset, validation_dataset):
    # 1. Cargar el modelo DistilBERT preentrenado para clasificación de secuencias
    model = DistilBertForSequenceClassification.from_pretrained(
        config.MODEL_NAME, 
        num_labels = config.num_labels,
        output_attentions = False, # No necesitamos las atenciones
        output_hidden_states = False, # No necesitamos los estados ocultos
    )
    
    model.to(config.device) # Mover el modelo a CPU (o GPU si está disponible)
    
    # 2. Definir los Argumentos de Entrenamiento (Hyperparameters)
    # El 'output_dir' es donde se guardarán los checkpoints y logs

    training_args1 = TrainingArguments(
        output_dir=config.output_dir1,          # Directorio para guardar outputs
        num_train_epochs=config.EPOCHS_1,              # Número total de épocas
        per_device_train_batch_size=config.per_device_train_batch_size,  # Tamaño del batch por dispositivo (GPU/CPU)
        per_device_eval_batch_size=config.per_device_eval_batch_size,   # Tamaño del batch para evaluación
        warmup_steps=config.warmup_steps,                # Número de pasos para el warmup del learning rate
        weight_decay=0.01,               # Descenso de peso (regularización L2)
        logging_dir=config.logging_dir1,            # Directorio para los logs
        logging_steps=50,                # Registrar log cada 50 pasos
        fp16=True,               
        eval_strategy="epoch",     # Evaluar al final de cada época
        save_strategy="epoch",           # Guardar el checkpoint al final de cada época
        load_best_model_at_end=True,     # Cargar el mejor modelo después del entrenamiento
        learning_rate=config.LEARNING_RATE_1,          # Tasa de aprendizaje
        dataloader_num_workers= 4     # Número de trabajadores para cargar datos              
    )

    training_args2 = TrainingArguments(
        output_dir = config.output_dir2,          # Directorio para guardar outputs
        num_train_epochs = config.EPOCHS_2,              # Número total de épocas
        per_device_train_batch_size = config.per_device_train_batch_size,  # Tamaño del batch por dispositivo (GPU/CPU)
        per_device_eval_batch_size = config.per_device_eval_batch_size,   # Tamaño del batch para evaluación
        warmup_steps = config.warmup_steps,                # Número de pasos para el warmup del learning rate
        weight_decay = 0.01,               # Descenso de peso (regularización L2)
        logging_dir = config.logging_dir2,            # Directorio para los logs
        logging_steps=50,                # Registrar log cada 50 pasos
        fp16=True,               
        eval_strategy="epoch",     # Evaluar al final de cada época
        save_strategy="epoch",           # Guardar el checkpoint al final de cada época
        load_best_model_at_end=True,     # Cargar el mejor modelo después del entrenamiento
        learning_rate = config.LEARNING_RATE_2,          # Tasa de aprendizaje
        dataloader_num_workers= 4     # Número de trabajadores para cargar datos              
    )

    trainer_1 = BasicTrainer(
        model=model,                         # El modelo a entrenar
        args=training_args1,                  # Los argumentos de entrenamiento
        train_dataset=train_dataset,         # Conjunto de datos de entrenamiento
        eval_dataset=validation_dataset,     # Conjunto de datos de validación
        compute_metrics=compute_metrics      # Función para calcular métricas
    )
    
    trainer_2 = WeightedTrainer(
        model=model,                         # El modelo a entrenar
        args=training_args2,                  # Los argumentos de entrenamiento
        train_dataset=train_dataset,         # Conjunto de datos de entrenamiento
        eval_dataset=validation_dataset,     # Conjunto de datos de validación
        compute_metrics=compute_metrics      # Función para calcular métricas
    )

    
    
    # 3. Entrenar el modelo
    trainer_1.train()
    # 4. Congelar el modelo y solo entrenar la capa de clasificación
    for param in model.distilbert.parameters():
        param.requires_grad = False # Congelar todas las capas de DistilBERT    

    trainer_2.train()
    
    return trainer_2