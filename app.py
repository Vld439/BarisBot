import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import os

# --- CONFIGURACION ---
st.set_page_config(page_title="Soporte Baris", layout="centered", page_icon="🤖")

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
    st.error("Error: Falta el archivo secrets.toml o la configuración de secretos en la nube.")
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
        # Aseguramos que exista el directorio
        if not os.path.exists("./cerebro_baris_db"):
            os.makedirs("./cerebro_baris_db")
            
        client = chromadb.PersistentClient(path="./cerebro_baris_db")
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        # Intentamos obtener o crear la coleccion para evitar errores si esta vacia
        return client.get_or_create_collection(name="manual_baris", embedding_function=emb_fn)
    except Exception as e:
        st.error(f"Error cargando base de datos vectorial: {e}")
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
                    puntos += 2 

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
st.title("🤖 Soporte Técnico JHF")
st.markdown("Sistema experto de consultas.")

query = st.text_input("Describa el problema:", placeholder="Ej: No imprime la factura...")

if st.button("Buscar Solución", type="primary"):
    if not query:
        st.warning("Por favor escriba una consulta.")
    else:
        with st.spinner("Procesando solicitud..."):
            fuentes = []
            
            # 1. Busqueda Ranking
            ganadores = buscar_por_puntos(query, df)
            if ganadores:
                fuentes.extend(ganadores)
            
            # 2. Busqueda Vectorial
            if len(fuentes) < 2 and collection:
                try:
                    vector_results = collection.query(query_texts=[query], n_results=3)
                    if vector_results['metadatas']:
                        for meta in vector_results['metadatas'][0]:
                            if not any(f['pregunta'] == meta['pregunta'] for f in fuentes):
                                fuentes.append(meta)
                except Exception as e:
                    print(f"Alerta Vector: {e}")

            # 3. Generar Respuesta
            if fuentes:
                contexto = ""
                for f in fuentes:
                    contexto += f"Tema: {f['pregunta']}\nProcedimiento: {f['respuesta']}\nVideo: {f['video']}\n---\n"
                
                prompt = f"""
                Actúa como un técnico experto de la empresa JHF.
                NO saludes con frases largas ni genéricas.
                Ve directo a la solución del problema paso a paso.
                Sé conciso, usa listas numeradas y lenguaje profesional.
                
                PREGUNTA DEL USUARIO: {query}
                
                DATOS TÉCNICOS:
                {contexto}
                """
                
                try:
                    response = model.generate_content(prompt)
                    
                    st.markdown("### Solución:")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Error conectando con la IA: {e}")
            else:
                st.error("No se encontró información en el manual para esa consulta específica.")