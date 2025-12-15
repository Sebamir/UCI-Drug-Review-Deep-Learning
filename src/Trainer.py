import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from src.utils import compute_metrics
from src.config import Config 

config = Config()

class BasicTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inicialización estándar
        self.loss_fct = nn.CrossEntropyLoss()
        
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Simplemente usa la función ya inicializada
        loss = self.loss_fct(
            logits.view(-1, self.model.config.num_labels), 
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Inicializa la función de pérdida UNA SOLA VEZ aquí
        if class_weights is not None:
            self.loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fct = nn.CrossEntropyLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Simplemente USA la función de pérdida ya inicializada
        loss = self.loss_fct(
            logits.view(-1, self.model.config.num_labels), 
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def full_training(train_dataset, validation_dataset, weights):
    """
    Entrena el modelo en dos fases con pesos de clase.
    
    Args:
        train_dataset: Dataset de entrenamiento
        validation_dataset: Dataset de validación
        weights: Tensor con pesos de clase para CrossEntropyLoss
    
    Returns:
        Trainer: El trainer de la fase 2 (con el modelo entrenado)
    """
    
    # Cargar modelo
    model = DistilBertForSequenceClassification.from_pretrained(
        config.MODEL_NAME, 
        num_labels=config.NUM_LABELS,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(config.DEVICE)
    
    # FASE 1: Fine-tuning completo
    print("=" * 60)
    print("FASE 1: Fine-tuning completo del modelo con pesos de clase")
    print("=" * 60)
    
    training_args_phase1 = TrainingArguments(
        output_dir=config.OUTPUT_DIR_1,
        num_train_epochs=config.EPOCHS_1,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=0.01,
        logging_dir=config.LOGGING_DIR_1,
        logging_steps=50,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=config.LEARNING_RATE_1,
        dataloader_num_workers=4,
        metric_for_best_model="eval_loss",  # O "eval_loss" según prefieras
    )
    
    trainer_phase1 = BasicTrainer(
        model=model,
        args=training_args_phase1,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        
    )
    
    trainer_phase1.train()
    
    # FASE 2: Solo capa clasificadora
    print("\n" + "=" * 60)
    print("FASE 2: Fine-tuning solo capa clasificadora")
    print("=" * 60)
    
    # Congelar DistilBERT
    for param in model.distilbert.parameters():
        param.requires_grad = False
    
    # Verificar parámetros entrenables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros entrenables: {trainable_params:,} / {total_params:,}")
    print(f"Porcentaje entrenable: {100 * trainable_params / total_params:.2f}%")
    
    training_args_phase2 = TrainingArguments(
        output_dir=config.OUTPUT_DIR_2,
        num_train_epochs=config.EPOCHS_2,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=0.01,
        logging_dir=config.LOGGING_DIR_2,
        logging_steps=50,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=config.LEARNING_RATE_2,
        dataloader_num_workers=4,
        metric_for_best_model="eval_loss",
    )
    
    trainer_phase2 = WeightedTrainer(
        model=model,
        args=training_args_phase2,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        class_weights=weights  
    )
    
    trainer_phase2.train()
    
    return trainer_phase2