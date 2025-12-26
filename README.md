# 💊 Drug Review Sentiment Analysis with DistilBERT

Este proyecto desarrolla un clasificador de sentimientos de última generación para reseñas de medicamentos, utilizando **DistilBERT**. El objetivo es identificar experiencias positivas y negativas de pacientes para ayudar en la farmacovigilancia y el análisis de satisfacción del usuario.

---

## 📊 Origen del Dataset
Los datos provienen del **UCI Drug Review Dataset**, disponible en el repositorio de Machine Learning de la UCI. Contiene más de 200,000 reseñas de medicamentos, nombres de fármacos, condiciones médicas y una calificación del 1 al 10 proporcionada por los usuarios.
link: https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018

Para este proyecto:
- **Positivos:** Ratings ≥ 7 (Efectividad alta).
- **Negativos:** Ratings ≤ 4 (Presencia de efectos secundarios o baja eficacia).
- **Neutros:** Los ratings de 5 y 6 fueron excluidos para forzar una clasificación binaria clara.

---

## 🛠️ El Desafío: Superando el Sesgo Predictivo

### El Problema Inicial
Durante las primeras fases del entrenamiento, el modelo presentaba una **Accuracy superior al 85%**, lo que sugería un éxito rotundo. Sin embargo, al probarlo con casos reales, el modelo **siempre predecía "Positivo"**, incluso ante críticas claramente negativas.

**¿Por qué ocurrió esto?**
El dataset original está altamente desbalanceado: hay muchas más reseñas positivas que negativas. El modelo aprendió que, para maximizar la exactitud (Accuracy), la estrategia más "segura" era clasificar todo como positivo.

### La Solución: Ingeniería de Datos y Umbrales
Para corregir este comportamiento, implementamos un proceso de procesamiento de datos en dos pasos:
1. **Sobremuestreo (Oversampling):** Balanceamos las clases en el set de entrenamiento duplicando la cantidad de muestras de la clase minoritaria para que el modelo viera una cantidad equitativa de ejemplos positivos y negativos.
2. **Balance de peso (wieght balance):** Luego de balancear las clases se realizo un balance de pesos adicional para garantizar la mayor igualdad posible.
3. **Entrenamiento en dos fases:** El entrenamiento se realizo en dos fases. La primera con el modelo completo con el fin de captar las caracteristicas generales y la segunda solo entrenando la capa de clasificación para volver al modelo más especifico. 
4. **Optimización del Umbral de Clasificación:** En lugar de usar el estándar de 0.5 para decidir si algo es positivo, implementamos la **Estadística J de Youden**. Esto nos permitió encontrar el punto de corte óptimo en la curva ROC que maximiza tanto la Sensibilidad como la Especificidad.

---

## 📈 ¿Por qué usamos estas métricas?

En este proyecto, la **Accuracy** fue descartada como métrica principal debido al desbalance inicial. En su lugar, utilizamos:

* **Curva ROC y AUC (Area Under Curve):** Fundamental para entender la capacidad del modelo de distinguir entre las dos clases, independientemente del umbral.
* **Matriz de Confusión:** Para visualizar específicamente los Falsos Positivos (un paciente que odió el medicamento pero la IA dice que le gustó), lo cual es crítico en contextos de salud.
* **J de Youden:** Elegida específicamente para "castigar" el sesgo del modelo y forzarlo a ser justo con la clase minoritaria (las reseñas negativas).

---

## Modos de uso
**Entrenamiento del modelo:**: python -m src.main --mode train.
**Evaluación tecnica (Testing):** python -m src.main --mode testing.
**A través de la interface para prediciones directas (Ejecutar Interfaz Web por Gradio):** python -m src.main --mode ui.

---

## 📁 Estructura del Proyecto

A continuación se detalla la organización del repositorio, siguiendo una arquitectura modular para facilitar el mantenimiento y la escalabilidad del modelo:
├── Dataset/
│   ├── raw/
│   │   ├── mimic-iii-clinical-database-demo-1.4/ # Próxima fase: Datos clínicos para integración
│   │   └── postgres/                             # Próxima fase: Almacenamiento persistente de datos clínicos
│   └── temporal/
│       ├── drugsComTest_raw.csv                  # Dataset de prueba original (UCI)
│       └── drugsComTrain_raw.csv                 # Dataset de entrenamiento original (UCI)
├── Modelo_entrenado/                             # Exportaciones del modelo en diferentes etapas
│   ├── drug_review_classifier_distilbert/
│   ├── drug_review_classifier_distilbert_FINAL/
│   └── modelo_final_consolidado/                 # Pesos finales listos para producción
├── results/                                      # Logs y artefactos del entrenamiento
│   ├── checkpoint-7347/                          # Puntos de control del entrenamiento
│   ├── Evaluation/                               # Gráficas ROC, Matrices de Confusión y Reportes
│   ├── stage_1_unfrozen/                         # Resultados del ajuste fino inicial
│   └── stage_2_frozen/                           # Resultados del entrenamiento con capas congeladas
├── src/                                          # Código fuente del proyecto
│   ├── config.py                                 # Configuración de rutas, hiperparámetros y constantes
│   ├── main.py                                   # Orquestador principal (Modos: train, testing, ui)
│   ├── Trainer.py                                # Lógica de entrenamiento y bucles de optimización
│   ├── ui.py                                     # Interfaz gráfica interactiva (Gradio)
│   └── utils.py                                  # Funciones de procesamiento, métricas y Youden's J
├── venv/                                         # Entorno virtual de Python
├── Drug Reviews.ipynb                            # Notebook de experimentación y análisis exploratorio
├── requirements.txt                              # Dependencias del proyecto (Transformers, Torch, Gradio)
└── .gitignore                                    # Archivos excluidos de control de versiones

---

## 🚀 Hoja de Ruta y Planes a Futuro (Roadmap)

El proyecto actual con el dataset de UCI es la base para un sistema de análisis de salud mucho más complejo. La arquitectura de carpetas ya está preparada para integrar las siguientes fases:

### 1. Integración con MIMIC-III (Datos Clínicos Reales)
Actualmente, la carpeta `Dataset/raw/mimic-iii-clinical-database-demo-1.4/` está reservada para la incorporación de registros electrónicos de salud (EHR).
* **Objetivo:** Cruzar el análisis de sentimiento de las reseñas con datos clínicos objetivos (signos vitales, resultados de laboratorio y códigos de diagnóstico).
* **Análisis Multimodal:** Entrenar un modelo que no solo lea el texto, sino que entienda el contexto clínico del paciente que escribe la reseña.

### 2. Implementación de Infraestructura SQL (PostgreSQL)
Uso del directorio `Dataset/raw/postgres/` para la persistencia de datos masivos.
* **Escalabilidad:** Migrar de archivos CSV/TSV planos a una base de datos relacional robusta.
* **Consultas Complejas:** Permitir que el modelo consulte rápidamente patrones entre condiciones médicas específicas y la efectividad percibida de los fármacos.

### 3. Mejora del Modelo de Lenguaje (NLP Avanzado)
* **Reconocimiento de Entidades Nombradas (NER):** Implementar una capa para identificar automáticamente nombres de medicamentos y síntomas específicos dentro de las reseñas, más allá del sentimiento general.
* **Modelos Médicos Especializados:** Realizar pruebas de *fine-tuning* con modelos como **BioBERT** o **PubMedBERT** para comparar si mejoran la precisión de DistilBERT en términos técnicos médicos.

### 4. Despliegue y API (Producción)
* **Contenerización:** Crear un `Dockerfile` para empaquetar la aplicación de Gradio y el modelo, facilitando su despliegue en la nube (AWS, Azure o Google Cloud).
* **API REST:** Desarrollar un endpoint con **FastAPI** para que otros servicios de salud puedan enviar reseñas y recibir la predicción de sentimiento y confianza de forma programática.


## 💻 Instalación y Uso

1. **Clonar e instalar:**
   ```bash
   git clone [https://github.com/tu-usuario/UCI-Drug-Review-Deep-Learning.git](https://github.com/tu-usuario/UCI-Drug-Review-Deep-Learning.git)
   cd UCI-Drug-Review-Deep-Learning
   pip install -r requirements.txt
