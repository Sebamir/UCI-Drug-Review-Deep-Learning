import pandas as pd
import torch
import psycopg2
from psycopg2.extras import execute_batch
from transformers import pipeline
import time
from config import Config

config = Config()

# Configuración
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'drug_reviews_db',
    'user': 'postgres',
    'password': 'S3bast1an.97'
}

MODEL_PATH = config.SAVE_MODEL_PATH
THRESHOLD = config.OPTIMAL_THRESHOLD
BATCH_SIZE = 5000  # Procesar de 5000 en 5000

def inference_pipeline():
    conn = None
    cursor = None
    
    try:
        print("--- 🧠 Iniciando Proceso de Inferencia ---")
        global_start_time = time.time()
        
        # Conectar a la base de datos
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Cargar el pipeline UNA SOLA VEZ
        device = 0 if torch.cuda.is_available() else -1
        print(f"🔧 Cargando modelo en: {'GPU' if device == 0 else 'CPU'}")
        classifier = pipeline("text-classification", model=MODEL_PATH, device=device)
        
        # Contador global
        total_processed = 0
        batch_number = 1
        
        # 🔄 BUCLE: Procesar hasta que no queden reseñas
        while True:
            print(f"\n{'='*60}")
            print(f"📦 LOTE #{batch_number} - Procesando hasta {BATCH_SIZE} reseñas...")
            print(f"{'='*60}")
            
            batch_start_time = time.time()
            
            # EXTRACT: Obtener el siguiente lote
            query = """
                SELECT r.unique_id, r.review 
                FROM raw_reviews r
                LEFT JOIN processed_reviews p ON r.unique_id = p.unique_id
                WHERE p.unique_id IS NULL
                LIMIT %s;
            """
            cursor.execute(query, (BATCH_SIZE,))
            rows = cursor.fetchall()
            
            # Si no hay más reseñas, terminar
            if not rows:
                print("\n✅ ¡No hay más reseñas pendientes!")
                break
            
            # Convertir a DataFrame
            df_tasks = pd.DataFrame(rows, columns=['unique_id', 'review'])
            print(f"📊 Reseñas en este lote: {len(df_tasks)}")
            
            # TRANSFORM: Predicción con la IA
            results = []
            print("🤖 Procesando con el modelo...")
            
            for index, row in df_tasks.iterrows():
                if (index + 1) % 500 == 0:
                    print(f"   Procesadas: {index + 1}/{len(df_tasks)}")
                
                try:
                    review_text = str(row['review']) if row['review'] else ""
                    
                    if not review_text.strip():
                        results.append({
                            'unique_id': row['unique_id'],
                            'sentiment_label': 'Negativo',
                            'sentiment_score': 0.5
                        })
                        continue
                    
                    pred = classifier(review_text, truncation=True, max_length=512)[0]
                    score = pred['score']
                    
                    if pred['label'] == 'LABEL_1' and score >= THRESHOLD:
                        final_label = 'Positivo'
                    else:
                        final_label = 'Negativo'
                    
                    results.append({
                        'unique_id': row['unique_id'],
                        'sentiment_label': final_label,
                        'sentiment_score': float(score)
                    })
                    
                except Exception as e:
                    print(f"⚠️ Error procesando reseña {row['unique_id']}: {str(e)}")
                    results.append({
                        'unique_id': row['unique_id'],
                        'sentiment_label': 'Negativo',
                        'sentiment_score': 0.5
                    })
            
            # LOAD: Guardar resultados
            print("💾 Guardando resultados...")
            
            insert_query = """
                INSERT INTO processed_reviews (unique_id, sentiment_label, sentiment_score)
                VALUES (%s, %s, %s)
                ON CONFLICT (unique_id) DO NOTHING;
            """
            
            data = [(r['unique_id'], r['sentiment_label'], r['sentiment_score']) for r in results]
            execute_batch(cursor, insert_query, data, page_size=100)
            conn.commit()  # ✅ Commit después de cada lote
            
            # Estadísticas del lote
            batch_end_time = time.time()
            batch_elapsed = batch_end_time - batch_start_time
            total_processed += len(results)
            
            df_results = pd.DataFrame(results)
            sentiment_counts = df_results['sentiment_label'].value_counts()
            
            print(f"\n✅ Lote #{batch_number} completado:")
            print(f"   ⏱️  Tiempo: {batch_elapsed:.2f}s")
            print(f"   ⚡ Velocidad: {len(results)/batch_elapsed:.2f} reseñas/seg")
            print(f"   📈 Positivos: {sentiment_counts.get('Positivo', 0)}")
            print(f"   📉 Negativos: {sentiment_counts.get('Negativo', 0)}")
            print(f"   📊 Total acumulado: {total_processed} reseñas")
            
            batch_number += 1
        
        # Estadísticas finales
        global_end_time = time.time()
        total_elapsed = global_end_time - global_start_time
        
        print(f"\n{'='*60}")
        print(f"🎉 ¡PROCESO COMPLETO!")
        print(f"{'='*60}")
        print(f"📊 Total procesado: {total_processed} reseñas")
        print(f"⏱️  Tiempo total: {total_elapsed:.2f} segundos ({total_elapsed/60:.2f} minutos)")
        print(f"⚡ Velocidad promedio: {total_processed/total_elapsed:.2f} reseñas/segundo")
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ Error en el pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("\n🔌 Conexión a la base de datos cerrada.")

if __name__ == "__main__":
    inference_pipeline()