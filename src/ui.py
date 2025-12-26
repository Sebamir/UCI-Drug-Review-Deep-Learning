import gradio as gr
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from src.config import Config
from src.utils import predict_sentiment_threshold # Usaremos tu lógica de predict

config = Config()
# CARGAMOS UNA SOLA VEZ AL INICIAR EL SCRIPT
print("Cargando modelo en memoria...")
model = DistilBertForSequenceClassification.from_pretrained(config.SAVE_MODEL_PATH)
model.to(config.DEVICE)
model.eval() # Ponemos el modelo en modo evaluación
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

def predict_interface(texto):
    if not texto.strip():
        return "Por favor, ingresa una reseña."
    
    # Llamamos a tu función de predicción existente
    
    _ , positive_prob, negative_prob = predict_sentiment_threshold(texto, model, tokenizer)

    return {
        "Positivo": positive_prob, 
        "Negativo": negative_prob
    }

# Configuración de la interfaz
demo = gr.Interface(
    fn=predict_interface,
    inputs=gr.Textbox(lines=5, placeholder="Escribe aquí la experiencia con el medicamento..."),
    outputs= gr.Label(num_top_classes=2, label="Resultado del Análisis"),
    title="Análisis de Sentimiento de Medicamentos",
    description="Esta IA analiza si la reseña de un fármaco es positiva o negativa basándose en DistilBERT.",
    examples=[
        ["The medication worked wonders for my migraine, I feel great!"],
        ["I had a terrible skin rash and the pain didn't go away."]
    ]
)

if __name__ == "__main__":
    demo.launch()