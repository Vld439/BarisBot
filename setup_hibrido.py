import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import shutil

# --- CONFIGURACION ---
input_csv = "base_conocimiento_baris_limpia.csv"
output_csv = "base_conocimiento_HIBRIDA.csv" # Archivo nuevo para la App
db_path = "./cerebro_baris_db"

# --- SINONIMOS Y JERGA ---
sinonimos = {
    "cancelar": ["anular", "revertir", "echar para atras", "deshacer", "borrar"],
    "impresion": ["ticket", "papel", "factura", "impresora", "trabada", "no sale"],
    "lento": ["pesado", "se arrastra", "tarda", "lenteja"],
    "error": ["fallo", "mensaje", "cartel", "aviso", "no funciona"],
    "anular": ["echar para atras", "baja", "cancelar"] 
}

# IDs criticos para forzar inyeccion
ids_anular = [3540, 40500, 118698, 118853, 118854, 70074, 214722]

print("--- INICIANDO CONFIGURACION HIBRIDA ---")

if not os.path.exists(input_csv):
    print("ERROR: No encuentro el CSV original.")
    exit()

try:
    df = pd.read_csv(input_csv)
    df = df.fillna("")
except:
    print("Error leyendo CSV.")
    exit()

# 1. ENRIQUECIMIENTO DE DATOS
print("1. Generando archivo de datos maestro...")

def enriquecer(row):
    texto = str(row['Pregunta'])
    id_actual = row['ID']
    extras = []
    
    # Inyeccion por ID (Prioridad Maxima)
    if id_actual in ids_anular:
        extras.extend(["ECHAR PARA ATRAS", "REVERTIR", "CANCELAR", "ANULAR"])
    
    # Inyeccion por Diccionario
    texto_lower = texto.lower()
    for k, v in sinonimos.items():
        if k in texto_lower:
            extras.extend(v)
            
    # Unir todo
    if extras:
        palabras_unicas = list(set(extras))
        # Ponemos las claves AL PRINCIPIO para el CSV y la busqueda de texto
        return f"{' '.join(palabras_unicas)} | {texto}"
    return texto

df['Pregunta_Hibrida'] = df.apply(enriquecer, axis=1)

# Guardamos este CSV porque la APP lo va a leer directamente para buscar texto
df.to_csv(output_csv, index=False)
print(f"   -> Archivo '{output_csv}' creado con exito.")

# 2. CREACION DE VECTORES (Para busquedas difusas)
print("2. Generando base vectorial...")
if os.path.exists(db_path):
    try:
        shutil.rmtree(db_path)
    except:
        pass

client = chromadb.PersistentClient(path=db_path)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
collection = client.create_collection(name="manual_baris", embedding_function=emb_fn)

ids = [str(i) for i in df.index]
documents = []
metadatas = []

for index, row in df.iterrows():
    # Vectorizamos la pregunta enriquecida
    doc = f"Pregunta: {row['Pregunta_Hibrida']}\nRespuesta: {row['Respuesta']}"
    documents.append(doc)
    metadatas.append({
        "id_original": str(row['ID']),
        "pregunta": str(row['Pregunta_Hibrida']), 
        "respuesta": str(row['Respuesta'])[:500],
        "video": str(row['Video'])
    })

collection.add(ids=ids, documents=documents, metadatas=metadatas)
print("3. Sistema Hibrido Listo.")
print("Ejecuta ahora el nuevo app.py")