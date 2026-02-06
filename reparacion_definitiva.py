import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import shutil

# --- CONFIGURACION ---
input_csv = "base_conocimiento_baris_limpia.csv"
db_path = "./cerebro_baris_db"

# --- IDs CRITICOS ---
# Estos son los IDs que vamos a modificar quirurgicamente
ids_anular = [3540, 40500, 118698, 118853, 118854, 70074, 214722]

print("--- INICIANDO PROTOCOLO NUCLEAR V3 ---")

if not os.path.exists(input_csv):
    print("ERROR: No encuentro el CSV.")
    exit()

df = pd.read_csv(input_csv)
df = df.fillna("")

# 1. ELIMINAR DUPLICADOS (Limpieza)
total_antes = len(df)
df = df.drop_duplicates(subset=['ID'])
print(f"1. Limpieza: Se eliminaron {total_antes - len(df)} duplicados.")

# 2. INYECCION FRONTAL AGRESIVA
print("2. Aplicando inyeccion frontal de palabras clave...")

def inyeccion_nuclear(row):
    texto = str(row['Pregunta'])
    id_actual = row['ID']
    
    # Si es uno de los IDs de anular, reescribimos la pregunta violentamente
    if id_actual in ids_anular:
        # TRUCO: Ponemos las claves AL PRINCIPIO y REPETIDAS
        prefijo = "ECHAR PARA ATRAS REVERTIR CANCELAR ANULAR. "
        return prefijo + texto
        
    return texto

df['Pregunta'] = df.apply(inyeccion_nuclear, axis=1)

# VERIFICACION: Imprimir como quedo la pregunta 3540
print("\n--- VERIFICACION VISUAL ---")
fila_test = df[df['ID'] == 3540]
if not fila_test.empty:
    print(f"La pregunta 3540 ahora es:\n'{fila_test.iloc[0]['Pregunta']}'")
print("---------------------------\n")

# 3. RECONSTRUCCION DE BASE DE DATOS
print("3. Generando nuevo cerebro...")
if os.path.exists(db_path):
    try:
        shutil.rmtree(db_path)
    except:
        pass

client = chromadb.PersistentClient(path=db_path)
# Usamos el modelo multilingue que ya tienes descargado
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
collection = client.create_collection(name="manual_baris", embedding_function=emb_fn)

ids = [str(i) for i in df.index]
documents = []
metadatas = []

for index, row in df.iterrows():
    # El documento vectorizado tendra las palabras clave al inicio
    doc = f"Pregunta: {row['Pregunta']}\nRespuesta: {row['Respuesta']}"
    documents.append(doc)
    metadatas.append({
        "id_original": str(row['ID']),
        "pregunta": str(row['Pregunta']), 
        "respuesta": str(row['Respuesta'])[:500],
        "video": str(row['Video'])
    })

collection.add(ids=ids, documents=documents, metadatas=metadatas)
print("4. Indexacion finalizada.")

# 4. PRUEBA FINAL
print("\n--- PRUEBA FINAL DE FUEGO ---")
query = "echar para atras una venta"
print(f"Buscando: '{query}'")

results = collection.query(query_texts=[query], n_results=3)

if results['metadatas']:
    print("\nRESULTADOS:")
    for i, meta in enumerate(results['metadatas'][0]):
        print(f"#{i+1}: {meta['pregunta']} (ID: {meta['id_original']})")