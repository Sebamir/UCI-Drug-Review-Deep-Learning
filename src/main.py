import argparse
import pandas as pd
from transformers import DistilBertForSequenceClassification, AutoTokenizer

# Importar tus módulos locales
from src.config import Config  # Asume que tienes una clase Config en src/config.py
from src.Trainer import full_training # Tu clase Trainer
from src.utils import ProcessingDataframe, predict_sentiment_threshold

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento y Predicción de Sentimientos con DistilBERT")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True, help="Modo de operación: 'train' para entrenar, 'predict' para predecir")
    parser.add_argument('--text', type=str, help="Texto para predecir el sentimiento (solo en modo 'predict')")
    args = parser.parse_args()

    config = Config()  # Instancia tu configuración

    if args.mode == 'train':
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
        print("Proceso de entrenamiento finalizado.")
        print("Modelo guardado en la ruta especificada.")
        model.save_pretrained(config.SAVE_MODEL_PATH)

    elif args.mode == 'predict':
        if not args.text:
            raise ValueError("Se requiere un texto para predecir en modo 'predict'.")
        print(f"Realizando predicción para el texto: {args.text}")
        # Aquí iría la lógica para cargar el modelo entrenado y hacer la predicción
        # Por ejemplo:
        # model = load_trained_model(config)
        # sentiment = preprocess_and_predict(args.text, model)
        # print(f"Sentimiento predicho: {sentiment}")

        model = DistilBertForSequenceClassification.from_pretrained(config.SAVE_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME) 
        sentiment, pos_prob, neg_prob = predict_sentiment_threshold(args.text, model, tokenizer)
        print(f"Sentimiento predicho: {sentiment} (Positivo: {pos_prob:.4f}, Negativo: {neg_prob:.4f})")
        print("Predicción completada.")

if __name__ == "__main__":
    main()