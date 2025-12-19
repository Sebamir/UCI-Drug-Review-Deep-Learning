import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from utils import compute_metrics
from config import Config 

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


def verify_gpu_setup(model, weights, train_dataset, validation_dataset):
    """
    Verifica que todo esté correctamente configurado para GPU.
    """
    print("=" * 60)
    print("VERIFICACIÓN DE RECURSOS GPU")
    print("=" * 60)
    
    # 1. GPU disponible
    gpu_available = torch.cuda.is_available()
    print(f"{'✓' if gpu_available else '✗'} GPU disponible: {gpu_available}")
    
    if gpu_available:
        print(f"  - Dispositivo: {torch.cuda.get_device_name(0)}")
        print(f"  - Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  - Memoria libre: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")
    
  # 2. Verificar modelo
    model_device = next(model.parameters()).device
    print(f"\n✓ Modelo en dispositivo: {model_device}")
    print(f"  - Esperado: {config.DEVICE}")
    
    # Comparar el tipo de dispositivo (cuda vs cpu)
    if model_device.type != config.DEVICE.type:
        raise AssertionError(f"❌ Modelo en {model_device}, esperado {config.DEVICE}")
    
    
    # 3. Verificar pesos
    print(f"\n✓ Pesos en dispositivo: {weights.device}")
    print(f"  - Shape: {weights.shape}")
    print(f"  - Valores: {weights.cpu().numpy()}")
    
    if weights.device.type != config.DEVICE.type:
        raise AssertionError(f"❌ Pesos en {weights.device}, esperado {config.DEVICE}")
    
    # 4. Verificar datasets (están en CPU, se mueven a GPU automáticamente)
    print(f"\n✓ Datasets:")
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Validation: {len(validation_dataset)} samples")
    
    # Verificar estructura del dataset
    sample_train = train_dataset[0]
    print(f"\n✓ Estructura del dataset de entrenamiento:")
    print(f"  - Keys: {list(sample_train.keys())}")
    
    for key, value in sample_train.items():
        if hasattr(value, '__len__') and not isinstance(value, str):
            print(f"  - {key}: tipo={type(value).__name__}, len={len(value)}")
        else:
            print(f"  - {key}: tipo={type(value).__name__}")
    
    # 5. Probar conversión a tensor y mover a GPU
    print(f"\n✓ Test de conversión dataset → GPU:")
    try:
        # Simular lo que hace el DataLoader
        test_input_ids = torch.tensor(sample_train['input_ids']).unsqueeze(0).to(config.DEVICE)
        test_attention = torch.tensor(sample_train['attention_mask']).unsqueeze(0).to(config.DEVICE)
        test_labels = torch.tensor(sample_train['labels']).to(config.DEVICE)
        
        print(f"  - input_ids: shape={test_input_ids.shape}, device={test_input_ids.device}")
        print(f"  - attention_mask: shape={test_attention.shape}, device={test_attention.device}")
        print(f"  - labels: shape={test_labels.shape}, device={test_labels.device}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            test_output = model(input_ids=test_input_ids, attention_mask=test_attention)
        print(f"  - output logits: shape={test_output.logits.shape}, device={test_output.logits.device}")
        print(f"  ✓ Forward pass exitoso en {config.DEVICE}")
        
        # Limpiar memoria
        del test_input_ids, test_attention, test_labels, test_output
        if gpu_available:
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"  ❌ Error en test de GPU: {e}")
        raise
    
    # 6. Parámetros del modelo
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Parámetros del modelo:")
    print(f"  - Total: {total:,}")
    print(f"  - Entrenables: {trainable:,} ({100*trainable/total:.1f}%)")
    
    if gpu_available:
        print(f"\n✓ Memoria GPU después de verificaciones:")
        print(f"  - Asignada: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  - Reservada: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    print("=" * 60)
    print("✓ TODAS LAS VERIFICACIONES PASARON CORRECTAMENTE")
    print("=" * 60 + "\n")




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
    # ============================================================
    # CARGAR Y PREPARAR MODELO
    # ============================================================
    print("Cargando modelo...")
    model = DistilBertForSequenceClassification.from_pretrained(
        config.MODEL_NAME, 
        num_labels=config.NUM_LABELS,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(config.DEVICE)
    
    # Asegurar que los pesos estén en el dispositivo correcto
    if weights.device != config.DEVICE:
        print(f"Moviendo pesos de {weights.device} a {config.DEVICE}...")
        weights = weights.to(config.DEVICE)
    
    # ============================================================
    # VERIFICACIÓN COMPLETA
    # ============================================================
    verify_gpu_setup(model, weights, train_dataset, validation_dataset)

    # ============================================================
    # FASE 1: Fine-tuning completo
    # ============================================================
    
    # FASE 1: Fine-tuning completo
    print("=" * 60)
    print("FASE 1: Fine-tuning completo del modelo base")
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
    
    trainer_phase1.train(resume_from_checkpoint=True)
    
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