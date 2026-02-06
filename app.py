import streamlit as st
import pandas as pd
import google.generativeai as genai

# --- 1. CONFIGURACIÓN ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("⚠️ No se encontró el archivo .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=api_key)

# --- 2. CARGAR DATOS ---
@st.cache_data
def load_data():
    try:
    
        df = pd.read_csv("base_conocimiento_baris_limpia.csv", encoding="utf-8")
        df = df.fillna("")
        return df
    except:
        return None

df = load_data()

if df is None:
    st.error("❌ Falta el archivo 'base_conocimiento_baris_limpia.csv'")
    st.stop()

# --- 3. MOTOR DE BÚSQUEDA ---
def buscar_contexto(query, dataframe, top_k=8):
    """
    Buscamos las 8 filas más parecidas. 
    para que la IA tenga margen de eleccion
    """
    query = query.lower()
    palabras = query.split()
    
    def score(row):
        txt = (str(row['Pregunta']) + " " + str(row['Respuesta'])).lower()
        return sum(1 for p in palabras if p in txt)

    temp = dataframe.copy()
    temp['score'] = temp.apply(score, axis=1)
    return temp[temp['score'] > 0].sort_values('score', ascending=False).head(top_k)

# --- 4. CEREBRO INTELIGENTE ---
# Instrucciones de Discernimiento para que no mezcle temas.
model = genai.GenerativeModel(
    model_name="models/gemini-flash-latest", 
    system_instruction="""
    Eres el Asistente Experto de Baris. Recibirás una lista de posibles respuestas del manual (Contexto).
    Tu trabajo es SELECCIONAR la mejor opción para la pregunta del usuario.

    REGLAS DE SELECCIÓN (CRÍTICO):
    1. **Distingue la Intención:**
       - Si el usuario pregunta "Cómo hacer X" (Proceso), DALE los pasos operativos. IGNORA soluciones de errores técnicos (Regedit, errores 0x...).
       - Solo da soluciones técnicas complejas si el usuario pregunta explícitamente por un "Error" o "Fallo".
    
    2. **Fidelidad:**
       - Copia los pasos numerados EXACTAMENTE como están en el texto seleccionado.
       - No resumas ni cambies las palabras técnicas.

    3. **Video:**
       - Si la fila seleccionada tiene un link en la columna 'Video', ponlo al final.
       - Si la columna 'Video' está vacía, NO inventes links.

    4. **Honestidad:**
       - Si ninguna de las opciones en el contexto responde realmente a la pregunta, di: "No tengo esa información en el manual".
    """
)

# --- 5. INTERFAZ ---
st.title("Soporte Baris")
st.caption("Sistema de soporte de información del Baris")

query = st.text_input("Pregunta del funcionario:", placeholder="Ej: Registrar funcionario...")

if st.button("Buscar") or query:
    if query:
        with st.spinner("Analizando manual..."):
            # 1. Python busca candidatos
            resultados = buscar_contexto(query, df)
            
            if resultados.empty:
                st.warning("No encontré coincidencias con esas palabras.")
            else:
                # 2. Preparamos el contexto de forma clara para que la IA elija
                # Formato: ID - Pregunta - Respuesta - Video
                contexto_texto = ""
                for index, row in resultados.iterrows():
                    contexto_texto += f"""
                    --- OPCIÓN {index} ---
                    PREGUNTA DEL MANUAL: {row['Pregunta']}
                    RESPUESTA DEL MANUAL: {row['Respuesta']}
                    VIDEO DISPONIBLE: {row['Video']}
                    -----------------------
                    """
                
                # 3. Prompt Final
                prompt = f"""
                PREGUNTA DEL USUARIO: "{query}"

                CANDIDATOS ENCONTRADOS EN EL MANUAL:
                {contexto_texto}

                INSTRUCCIÓN: Analiza los candidatos. ¿Cuál responde mejor a la intención del usuario? 
                Si el usuario pide un proceso simple, evita las respuestas sobre "Errores de sistema" o "Regedit" a menos que sea necesario.
                Genera la respuesta usando solo el candidato ganador.
                """
                
                try:
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                    
                    # Ver qué opciones consideró la IA
                    with st.expander("Ver qué encontró el sistema internamente"):
                        st.dataframe(resultados[['Pregunta', 'Respuesta']])
                        
                except Exception as e:
                    st.error(f"Error: {e}")