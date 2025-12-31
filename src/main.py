import argparse
import os
import pandas as pd
from transformers import DistilBertForSequenceClassification, AutoTokenizer

# Importar tus módulos locales
from src.config import Config  
from src.Trainer import full_training 
from src.utils import ProcessingDataframe, predict_sentiment_threshold, run_detailed_evaluation_max, plot_loss_and_lr, ProcessingTest 
from src.ui import demo

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento y Predicción de Sentimientos con DistilBERT")
    parser.add_argument('--mode', type=str, choices=['train', 'testing', 'predict', 'ui'], required=True, help="Modo de operación: 'train' para entrenar, 'predict' para predecir")
    parser.add_argument('--text', type=str, help="Texto para predecir el sentimiento (solo en modo 'predict')")
    args = parser.parse_args()

    config = Config()  # Instancia tu configuración

    if args.mode == 'train':

        dirs_to_create = [
        config.OUTPUT_DIR_1, 
        config.OUTPUT_DIR_2, 
        config.LOGGING_DIR_1, 
        config.LOGGING_DIR_2
    ]
    
        for d in dirs_to_create:
            if not os.path.exists(d):
                print(f"📁 Creando directorio: {d}")
                os.makedirs(d, exist_ok=True)
        print("Iniciando el entrenamiento del modelo...")
        df = pd.read_csv(config.DATA_PATH)

        print(f"Dataset cargado con {len(df)} registros.")
        train_dataset, validation_dataset, weights = ProcessingDataframe(df)

        print("Datos preprocesados y divididos en conjuntos de entrenamiento y validación.")
        print(f"Pesos de clase calculados: {weights}")
        print("Iniciando el entrenamiento completo del modelo con pesos de clase...")
        model = full_training(train_dataset, validation_dataset, weights)

        print("Modelo entrenado con éxito.")
        print("Entrenamiento completado.")
        print("Calculando métricas en el conjunto de validación...")
        results = model.evaluate()

        print(f"Métricas de validación: {results}")

        print("Imprimiendo grafica")
        plot_loss_and_lr(
            config.OUTPUT_DIR_1,
            config.OUTPUT_DIR_2
        )
        
        print("Proceso de entrenamiento finalizado.")
        print("Modelo guardado en la ruta especificada.")
        model.save_model(config.SAVE_MODEL_PATH)

    elif args.mode == "testing":
        print("Iniciando evaluación del modelo entrenado...")
        print("Cargando datos...")
        df = pd.read_csv(config.TEST_PATH)

        print(f"Dataset de prueba cargado con {len(df)} registros. Procesando...")
        test_dataset = ProcessingTest(df)

        print("Cargando el modelo entrenado...")
        model = DistilBertForSequenceClassification.from_pretrained(config.SAVE_MODEL_PATH)
        model.to(config.DEVICE)

        print("Ejecutando evaluación detallada...")
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        results = run_detailed_evaluation_max(
            model, 
            test_dataset, 
            output_pdf=config.REPORT
        )
 
        print("Evaluación completada.")

    elif args.mode == 'predict':
        if not args.text:
            raise ValueError("Se requiere un texto para predecir en modo 'predict'.")
        print(f"Realizando predicción para el texto: {args.text}")
    
        model = DistilBertForSequenceClassification.from_pretrained(config.SAVE_MODEL_PATH)
        model.to(config.DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME) 
        sentiment, pos_prob, neg_prob = predict_sentiment_threshold(args.text, model, tokenizer)
        print(f"Sentimiento predicho: {sentiment} (Positivo: {pos_prob:.4f}, Negativo: {neg_prob:.4f})")
        print("Predicción completada.")

    elif args.mode == "ui":
        print("Lanzando interfaz web...")
        demo.launch(share=False) # share=True si quieres un link público temporal

if __name__ == "__main__":
    main()