import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
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

def ingest_data():
    conn = None
    cursor = None
    
    try:
        print("--- 📥 Iniciando Ingesta de Datos ---")
        start_time = time.time()

        file_path = config.TEST_PATH
        
        # Lectura del CSV
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            df = pd.read_csv(f)
        
        # Renombrar columnas
        df = df.rename(columns={
            'uniqueID': 'unique_id',
            'drugName': 'drug_name',
            'condition': 'condition',
            'review': 'review',
            'rating': 'rating',
            'date': 'review_date',
            'usefulCount': 'useful_count'
        })
        
        # Conversión de fecha
        df['review_date'] = pd.to_datetime(df['review_date'], format='mixed').dt.date
        
        # Convertir NaN a None para PostgreSQL
        df = df.where(pd.notna(df), None)

        print(f"Subiendo {len(df)} registros a PostgreSQL...")
        
        # Conectar a PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Preparar query de inserción
        insert_query = """
            INSERT INTO raw_reviews 
            (unique_id, drug_name, condition, review, rating, review_date, useful_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        # Convertir DataFrame a lista de tuplas
        data = [tuple(row) for row in df.values]
        
        # Inserción por lotes (más rápida)
        execute_batch(cursor, insert_query, data, page_size=1000)
        
        conn.commit()
        
        end_time = time.time()
        print(f"✅ ¡Éxito! Se cargaron {len(df)} registros en {end_time - start_time:.2f} segundos.")

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ Error durante la carga: {str(e)}")
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    ingest_data()