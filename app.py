import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import os

# --- CONFIGURACION ---
st.set_page_config(page_title="Soporte Baris", layout="centered", page_icon="🤖")

# Ocultar menu de hamburguesa y footer de Streamlit para que se vea mas profesional
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("Error: Falta el archivo secrets.toml")
    st.stop()

genai.configure(api_key=api_key)

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    path = "base_conocimiento_HIBRIDA.csv"
    if os.path.exists(path):
        return pd.read_csv(path).fillna("")
    return None

df = load_data()

@st.cache_resource
def get_vector_store():
    try:
        client = chromadb.PersistentClient(path="./cerebro_baris_db")
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        return client.get_collection(name="manual_baris", embedding_function=emb_fn)
    except:
        return None

collection = get_vector_store()
model = genai.GenerativeModel("gemini-flash-latest")

# --- MOTOR DE PUNTUACION ---
def buscar_por_puntos(query, dataframe):
    if dataframe is None: return []
    palabras_usuario = query.lower().split()
    palabras_clave = [p for p in palabras_usuario if p not in ["una", "el", "la", "de", "para", "en", "como", "quiero", "que"]]
    
    if not palabras_clave: return []

    resultados = []
    for index, row in dataframe.iterrows():
        texto_fila = str(row['Pregunta_Hibrida']).lower()
        puntos = 0
        for palabra in palabras_clave:
            if palabra in texto_fila:
                puntos += 1
                if palabra in ["echar", "atras", "revertir", "cancelar", "trabada", "lento", "error", "no sale"]:
                    puntos += 2 # Bonificacion por palabras urgentes

        if puntos > 0:
            resultados.append({
                "puntos": puntos,
                "pregunta": row['Pregunta_Hibrida'],
                "respuesta": row['Respuesta'],
                "video": row['Video']
            })
    
    resultados = sorted(resultados, key=lambda x: x['puntos'], reverse=True)
    return resultados[:3]

# --- INTERFAZ GRAFICA ---
st.title(" Asistente Técnico Baris")
st.markdown("Escribe tu consulta abajo y te ayudaré paso a paso.")

query = st.text_input("¿En qué puedo ayudarte hoy?", placeholder="Ej: No imprime la factura, quiero anular una venta...")

if st.button("Buscar Solución", type="primary"):
    if not query:
        st.warning("Por favor escribe una consulta.")
    else:
        with st.spinner("Analizando manual técnico..."):
            fuentes = []
            
            # 1. Busqueda Ranking (Texto)
            ganadores = buscar_por_puntos(query, df)
            if ganadores:
                fuentes.extend(ganadores)
            
            # 2. Busqueda Vectorial (IA) - Solo si el ranking dio pocos resultados
            if len(fuentes) < 2 and collection:
                vector_results = collection.query(query_texts=[query], n_results=3)
                if vector_results['metadatas']:
                    for meta in vector_results['metadatas'][0]:
                        # Evitar duplicados
                        if not any(f['pregunta'] == meta['pregunta'] for f in fuentes):
                            fuentes.append(meta)

            # 3. Generar Respuesta
            if fuentes:
                contexto = ""
                for f in fuentes:
                    contexto += f"Tema: {f['pregunta']}\nProcedimiento: {f['respuesta']}\nVideo: {f['video']}\n---\n"
                
                prompt = f"""
                Actúa como un técnico experto de la empresa JHF.
                NO saludes con frases largas tipo "Hola, soy el asistente...".
                Ve directo a la solución del problema.
                Sé conciso, usa listas numeradas y lenguaje profesional.
                
                PREGUNTA: {query}
                
                DATOS TÉCNICOS:
                {contexto}
                """
                
                try:
                    response = model.generate_content(prompt)
                    
                    # Mostrar respuesta limpia en una tarjeta
                    st.markdown("### Solución Encontrada")
                    st.markdown("---")
                    st.markdown(response.text)
                    
                    except Exception as e:
                        st.error(f"Error REAL: {e}")
            else:
                st.error("Lo siento, no encontré información sobre ese tema específico en el manual.")
                st.info("Intenta reformular la pregunta con otras palabras.")