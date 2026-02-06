import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import shutil

# --- CONFIGURACION ---
input_csv = "base_conocimiento_baris_limpia.csv"
db_path = "./cerebro_baris_db"

# --- DICCIONARIO DE JERGA ---
sinonimos = {
    "cancelar": ["anular", "revertir", "echar para atras", "deshacer", "borrar", "quitar", "me equivoque", "baja"],
    "impresion": ["ticket", "papel", "factura", "comprobante", "impresora", "cola", "trabada", "atascada", "no sale", "imprime mal"],
    "lento": ["pesado", "se arrastra", "cuelga", "tarda", "lag", "lenteja", "no responde", "lentasos"],
    "venta": ["cobro", "facturacion", "despacho", "caja", "vendido"],
    "anular": ["echar para atras", "cancelar", "revertir", "dar de baja", "eliminar", "restaurar", "devolver"],
    "cliente": ["comprador", "persona", "señor", "tipo"],
    "stock": ["mercaderia", "inventario", "existencia", "articulos", "productos"],
    "caja": ["dinero", "plata", "sencillo", "cobranza", "turno", "cierre"],
    "error": ["fallo", "mensaje", "cartel", "aviso", "no funciona", "roto", "bug"]
}

print("--- INICIANDO REPARACION TOTAL ---")

# 1. Cargar Datos Originales
if not os.path.exists(input_csv):
    print(f"Error critico: No encuentro {input_csv}")
    exit()

df = pd.read_csv(input_csv)
df = df.fillna("")
print(f"1. Datos cargados: {len(df)} registros.")

# 2. Inyectar Jerga (En memoria)
print("2. Inyectando vocabulario tecnico y coloquial...")
def enriquecer(texto):
    t = str(texto).lower()
    extras = []
    for k, v in sinonimos.items():
        if k in t:
            extras.extend(v)
    if extras:
        # Agregar palabras ocultas al final
        return f"{texto} [{' '.join(list(set(extras)))}]"
    return texto

# Aplicamos la inyeccion a la columna Pregunta
df['Pregunta'] = df['Pregunta'].apply(enriquecer)

# 3. Reiniciar Base de Datos Vectorial
print("3. Reconstruyendo cerebro digital...")
if os.path.exists(db_path):
    try:
        shutil.rmtree(db_path) # Borrado fisico de la carpeta
        print("   - Memoria antigua eliminada.")
    except:
        pass

client = chromadb.PersistentClient(path=db_path)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
collection = client.create_collection(name="manual_baris", embedding_function=emb_fn)

# 4. Indexar Datos Inyectados
ids = [str(i) for i in df.index]
documents = []
metadatas = []

for index, row in df.iterrows():
    # El documento contiene la pregunta ENRIQUECIDA + la respuesta
    doc = f"Pregunta: {row['Pregunta']}\nRespuesta: {row['Respuesta']}"
    documents.append(doc)
    metadatas.append({
        "id_original": str(row['ID']),
        "pregunta": str(row['Pregunta']), # Guardamos la pregunta con los trucos ocultos
        "respuesta": str(row['Respuesta'])[:500],
        "video": str(row['Video'])
    })

collection.add(ids=ids, documents=documents, metadatas=metadatas)
print(f"4. Indexacion completada: {len(documents)} conocimientos guardados.")

# 5. PRUEBA DE FUEGO AUTOMATICA
print("\n--- PRUEBA DE DIAGNOSTICO INTERNO ---")
test_query = "echar para atras una venta"
print(f"Buscando: '{test_query}'...")

results = collection.query(query_texts=[test_query], n_results=1)

if results['metadatas'] and results['metadatas'][0]:
    mejor_resultado = results['metadatas'][0][0]
    print(f"RESULTADO ENCONTRADO: {mejor_resultado['pregunta']}")
    print("Estado: EXITOSO (Si ves la palabra 'Anular' o 'Venta' arriba, funciono).")
else:
    print("Estado: FALLIDO (No se encontro nada).")

print("\nAhora ejecuta: streamlit run app.py")