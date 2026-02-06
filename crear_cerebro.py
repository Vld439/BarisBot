import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os

# --- CONFIGURACION ---
csv_path = "base_conocimiento_baris_POTENCIADA.csv"
db_path = "./cerebro_baris_db"

print("Iniciando proceso de indexacion vectorial...")

# 1. Verificacion de archivo
if not os.path.exists(csv_path):
    print(f"Error: No se encuentra el archivo '{csv_path}'")
    exit()

# 2. Carga de datos
try:
    # Leemos el CSV y rellenamos valores vacios para evitar errores
    df = pd.read_csv(csv_path)
    df = df.fillna("")
    print(f"CSV cargado correctamente. Registros encontrados: {len(df)}")
except Exception as e:
    print(f"Error al leer el CSV: {e}")
    exit()

# 3. Configuracion de ChromaDB
print("Configurando motor de base de datos vectorial...")

# Cliente persistente guarda los datos en disco
client = chromadb.PersistentClient(path=db_path)

# Funcion de embedding: Convierte texto a numeros
# all-MiniLM-L6-v2 es un modelo eficiente y gratuito que corre localmente
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# Reiniciar la coleccion si ya existe para asegurar datos frescos
try:
    client.delete_collection(name="manual_baris")
except:
    pass

# Crear nueva coleccion
collection = client.create_collection(name="manual_baris", embedding_function=emb_fn)

# 4. Procesamiento e Indexacion
ids = []
documents = []
metadatas = []

print("Procesando registros y generando vectores...")

for index, row in df.iterrows():
    # ID unico para la base de datos
    ids.append(str(index))
    
    # Contexto Semantico:
    # Combinamos la pregunta y la respuesta. Esto permite que el buscador
    # encuentre coincidencias tanto en el problema (pregunta) como en la solucion (respuesta).
    texto_para_vectorizar = f"Pregunta: {row['Pregunta']}\nRespuesta: {row['Respuesta']}"
    documents.append(texto_para_vectorizar)
    
    # Metadatos:
    # Informacion que recuperaremos despues de la busqueda (para mostrar en pantalla)
    # Es importante convertir todo a string para evitar errores de tipo en ChromaDB
    metadatas.append({
        "id_original": str(row['ID']),
        "pregunta": str(row['Pregunta']),
        # Cortamos la respuesta en metadatos si es muy larga para ahorrar espacio, 
        # pero el vector (documents) si usa el texto completo.
        "respuesta": str(row['Respuesta'])[:500], 
        "video": str(row['Video'])
    })

# Carga por lotes en la base de datos
try:
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Exito: Se han indexado {len(documents)} documentos en '{db_path}'.")
    print("El sistema de busqueda semantica esta listo.")
except Exception as e:
    print(f"Error al guardar en ChromaDB: {e}")