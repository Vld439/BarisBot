import streamlit as st
from groq import Groq
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

# --- CONEXIÓN GROQ (NUEVO CEREBRO) ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except:
    st.error("Error: Falta la GROQ_API_KEY en los secrets.")
    st.stop()

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    path = "base_conocimiento_HIBRIDA.csv"
    if os.path.exists(path):
        return pd.read_csv(path).fillna("")
    return None

df = load_data()

# --- BASE DE DATOS VECTORIAL ---
@st.cache_resource
def get_vector_store():
    try:
        if not os.path.exists("./cerebro_baris_db"):
            os.makedirs("./cerebro_baris_db")
        
        db_client = chromadb.PersistentClient(path="./cerebro_baris_db")
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        return db_client.get_or_create_collection(name="manual_baris", embedding_function=emb_fn)
    except Exception as e:
        return None

collection = get_vector_store()

# --- BUSQUEDA MEJORADA (Lee preguntas Y respuestas) ---
def buscar_por_puntos(query, dataframe):
    if dataframe is None: return []
    palabras = query.lower().split()
    resultados = []
    
    for index, row in dataframe.iterrows():
        puntos = 0
        # TRUCO: Juntamos pregunta y respuesta en un solo texto para buscar
        texto_completo = (str(row['Pregunta_Hibrida']) + " " + str(row['Respuesta'])).lower()
        
        for p in palabras:
            if p in texto_completo: 
                puntos += 1
        
        # Si encuentra palabras clave, guardamos el resultado
        if puntos > 0:
            # Damos un empujoncito extra si la palabra está en el título
            if query.lower() in str(row['Pregunta_Hibrida']).lower():
                puntos += 5
            resultados.append({'fila': row.to_dict(), 'puntos': puntos})
            
    # Ordenamos: primero los que tengan más coincidencia (mayor puntaje)
    resultados = sorted(resultados, key=lambda x: x['puntos'], reverse=True)[:3]
    
    # Devolvemos solo los datos limpios
    return [r['fila'] for r in resultados]
# --- INTERFAZ ---
st.title("🤖 Soporte Baris (Motor Llama-3)")

query = st.text_input("Describa el problema:")

if st.button("Buscar Solución", type="primary"):
    if not query:
        st.warning("Escribe algo primero.")
    else:
        with st.spinner("Consultando a Llama-3..."):
            # 1. Recuperar contexto
            contexto_texto = ""
            fuentes = buscar_por_puntos(query, df)
            
            # Si no hay resultados por palabras, intentar vector (opcional)
            if not fuentes and collection:
                try:
                    res = collection.query(query_texts=[query], n_results=2)
                    if res['metadatas'][0]:
                        for meta in res['metadatas'][0]:
                            contexto_texto += f"- {meta['pregunta']}: {meta['respuesta']}\n"
                except:
                    pass

            for f in fuentes:
                contexto_texto += f"PREGUNTA: {f['Pregunta_Hibrida']}\nSOLUCIÓN: {f['Respuesta']}\nVIDEO: {f['Video']}\n---\n"
            
            # 2. Prompt para Groq
            prompt_sistema = """
            Eres un experto en soporte técnico de JHF Ingeniería.
            Tu trabajo es leer la INFORMACIÓN TÉCNICA provista y responder la duda del usuario.
            - Sé directo y profesional.
            - Si la información incluye un link de video, dámelo.
            - Si la información NO está en el texto provisto, di: "No tengo información sobre eso en mis manuales actuales."
            """

            mensaje_usuario = f"""
            INFORMACIÓN TÉCNICA:
            {contexto_texto}
            
            CONSULTA DEL USUARIO:
            {query}
            """
            
            try:
                # LLAMADA A LA API DE GROQ
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": mensaje_usuario}
                    ],
                    model="llama-3.3-70b-versatile", # Modelo muy potente y rápido
                    temperature=0.2, # Baja temperatura para ser preciso
                )
                
                respuesta = chat_completion.choices[0].message.content
                st.markdown("### Solución:")
                st.write(respuesta)
                
            except Exception as e:
                st.error(f"Error conectando con Groq: {e}")

                # --- DEBUGGING: HERRAMIENTA DE RAYOS X ---
# Pega esto al final de tu app.py para ver qué tiene el bot en la panza

with st.sidebar:
    st.divider()
    st.header("🕵️‍♂️ Rayos X (Debug)")
    st.write("Escribe una palabra para ver si REALMENTE está en el CSV:")
    
    palabra_clave = st.text_input("Buscar palabra exacta:")
    
    if palabra_clave:
        if df is not None:
            # Busca la palabra en CUALQUIER columna del CSV (sin IA, búsqueda bruta)
            mask = df.apply(lambda row: row.astype(str).str.contains(palabra_clave, case=False).any(), axis=1)
            df_resultado = df[mask]
            
            st.write(f"🔍 Resultados encontrados: **{len(df_resultado)}**")
            
            if len(df_resultado) > 0:
                # Muestra la tabla tal cual la ve el bot
                st.dataframe(df_resultado)
            else:
                st.error(f"❌ La palabra '{palabra_clave}' NO existe en el archivo CSV cargado.")
        else:
            st.error("⚠️ El archivo CSV no se ha cargado todavía.")