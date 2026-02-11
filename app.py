import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from github import Github
import pdfplumber
import io
import os
import time
import shutil

#Configuracion de la pagina
st.set_page_config(page_title="Soporte Baris", layout="wide")

# --- FUNCIONES DE MANTENIMIENTO (BARRA LATERAL) ---

def procesar_pdf_y_generar_csv(file_obj):
    """Lee el PDF subido en memoria y devuelve el DataFrame y el texto CSV"""
    
    # Configurar Groq
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except:
        st.error("[ERROR] Falta la GROQ_API_KEY en secrets.")
        return None, None

    # Leer PDF desde memoria
    texto_completo = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            # Limpieza basica para reporte WinJes
            t = t.replace("Preguntas Frecuentes", "").replace("ORDEN:Hora", "")
            # Eliminar encabezados de pagina repetitivos si es necesario
            if "Pág.:" in t:
                lines = t.split('\n')
                t = '\n'.join([l for l in lines if "Pág.:" not in l and "Fecha:[-]" not in l])
            texto_completo += t + "\n"

    # Procesar con IA por bloques
    chunk_size = 3000
    chunks = [texto_completo[i:i+chunk_size] for i in range(0, len(texto_completo), chunk_size)]
    datos = []
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"Procesando bloque {i+1} de {len(chunks)}...")
        
        # Prompt para Llama-3 (Con generacion de sinonimos)
        prompt = f"""
        Analiza el siguiente texto de un manual tecnico (WinJes) y extrae datos estructurados.
        
        FORMATO DE ENTRADA TIPICO:
        Texto de la pregunta
        1. Paso uno
        2. Paso dos
        (ID_NUMERICO)

        TU TAREA:
        Genera una lista CSV separada por pipes (|).
        Formato: ID|Pregunta Hibrida|Respuesta

        REGLAS CLAVE:
        1. ID: Es el numero entre parentesis al final del bloque.
        2. Pregunta Hibrida: Toma la pregunta original y AGREGALE sinonimos entre parentesis.
           Ejemplo: "Anular Venta" -> "Anular Venta (borrar, cancelar, eliminar factura, echar para atras)"
        3. Respuesta: Incluye todos los pasos numerados y enlaces (links) si existen.
        4. Solo genera las lineas de datos, sin introduccion.

        TEXTO: {chunk}
        """
        
        try:
            chat = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            res = chat.choices[0].message.content
            
            # Procesar respuesta de la IA
            for linea in res.split('\n'):
                if "|" in linea:
                    parts = linea.split("|")
                    if len(parts) >= 3:
                        # Limpieza de ID
                        id_clean = parts[0].strip().replace("(", "").replace(")", "")
                        preg = parts[1].strip()
                        resp = parts[2].strip()
                        datos.append([id_clean, preg, resp, ""])
        except Exception as e:
            st.error(f"Error en bloque {i}: {e}")
        
        progress_bar.progress((i + 1) / len(chunks))
        time.sleep(0.5) # Respetar limites de API

    progress_bar.empty()
    status_text.empty()
    
    # Crear DataFrame y eliminar duplicados
    df = pd.DataFrame(datos, columns=["ID", "Pregunta_Hibrida", "Respuesta", "Video"])
    df = df.drop_duplicates(subset=["ID"])
    
    # Convertir a CSV String
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue(), df

def actualizar_github(nuevo_csv_str, nombre_repo):
    """Sube el archivo directamente a GitHub usando la API"""
    try:
        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(nombre_repo)
        
        file_path = "base_conocimiento_HIBRIDA.csv"
        mensaje_commit = "Actualizacion automatica desde Panel de Control"
        
        try:
            # Intentar obtener el archivo existente para actualizarlo
            contents = repo.get_contents(file_path)
            repo.update_file(file_path, mensaje_commit, nuevo_csv_str, contents.sha)
            return True, "[EXITO] Archivo actualizado correctamente en GitHub."
        except:
            # Si no existe, crearlo
            repo.create_file(file_path, mensaje_commit, nuevo_csv_str)
            return True, "[EXITO] Archivo creado en GitHub."
            
    except Exception as e:
        return False, f"[ERROR] Fallo la conexion con GitHub: {e}"

# --- BARRA LATERAL: PANEL DE CONTROL ---
with st.sidebar:
    st.title("Panel de Control")
    st.divider()
    st.subheader("Actualizar Base de Conocimientos")
    st.info("Sube el PDF exportado de WinJes (rp_mref.frx.pdf) para actualizar el bot.")
    
    uploaded_file = st.file_uploader("Seleccionar PDF", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Procesar y Subir a GitHub"):
            with st.status("Iniciando proceso...", expanded=True) as status:
                
                status.write("Leyendo PDF y generando sinonimos...")
                csv_content, df_preview = procesar_pdf_y_generar_csv(uploaded_file)
                
                if csv_content:
                    status.write(f"Se extrajeron {len(df_preview)} registros.")
                    st.dataframe(df_preview.head(3))
                    
                    status.write("Conectando con GitHub...")
                    # REPOSITORIO DE DESTINO
                    REPO_NAME = "Vld439/BarisBot" 
                    
                    exito, mensaje = actualizar_github(csv_content, REPO_NAME)
                    
                    if exito:
                        status.update(label="Proceso Finalizado", state="complete", expanded=False)
                        st.success(mensaje)
                        st.info("Por favor espera unos minutos a que Streamlit detecte el cambio y recargue la aplicacion.")
                    else:
                        status.update(label="Error", state="error")
                        st.error(mensaje)
                else:
                    st.error("No se pudo procesar el PDF.")
    
    st.divider()
    st.write("Estado del Sistema: ACTIVO")

# --- ZONA PRINCIPAL: CHATBOT ---

st.title("Asistente de Soporte Baris")
st.write("Escribe tu consulta sobre el sistema.")

# Funcion para cargar/crear la base de datos vectorial
@st.cache_resource
def get_vector_store():
    db_path = "./cerebro_baris_db"
    csv_path = "base_conocimiento_HIBRIDA.csv"
    
    # Si existe el CSV, intentamos cargar/crear la DB
    if os.path.exists(csv_path):
        # Limpieza preventiva: Si el CSV es mas nuevo que la DB, reconstruir
        # (Aqui simplificamos borrando siempre para asegurar frescura al reiniciar app)
        if os.path.exists(db_path):
            try:
                shutil.rmtree(db_path)
            except:
                pass
        
        client = chromadb.PersistentClient(path=db_path)
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        collection = client.get_or_create_collection(name="manual_baris", embedding_function=emb_fn)
        
        # Leer CSV y poblar DB
        df = pd.read_csv(csv_path).fillna("")
        
        ids = []
        docs = []
        metas = []
        
        for _, row in df.iterrows():
            # ID unico
            doc_id = str(row['ID'])
            # Contenido para busqueda semantica (Pregunta + Respuesta)
            texto = f"{row['Pregunta_Hibrida']} \n {row['Respuesta']}"
            
            ids.append(doc_id)
            docs.append(texto)
            metas.append({
                "pregunta": str(row['Pregunta_Hibrida']),
                "respuesta": str(row['Respuesta']),
                "video": str(row['Video'])
            })
            
        # Agregar en lotes
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.add(ids=ids[i:end], documents=docs[i:end], metadatas=metas[i:end])
            
        return collection, df
    else:
        return None, None

# Cargar cerebro
collection, df_global = get_vector_store()

if collection is None:
    st.warning("No se encontro la base de conocimientos. Por favor sube un PDF en el panel lateral para iniciar.")
    st.stop()

# Logica de busqueda hibrida (Puntos)
def buscar_por_puntos(query, dataframe):
    if dataframe is None: return []
    palabras = query.lower().split()
    resultados = []
    
    for index, row in dataframe.iterrows():
        puntos = 0
        texto_completo = (str(row['Pregunta_Hibrida']) + " " + str(row['Respuesta'])).lower()
        
        for p in palabras:
            if p in texto_completo: 
                puntos += 1
        
        if puntos > 0:
            # Bonus si esta en el titulo
            if query.lower() in str(row['Pregunta_Hibrida']).lower():
                puntos += 5
            resultados.append({'fila': row.to_dict(), 'puntos': puntos})
            
    resultados = sorted(resultados, key=lambda x: x['puntos'], reverse=True)[:3]
    return [r['fila'] for r in resultados]

# Interfaz de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿En que puedo ayudarte hoy?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 1. Busqueda Vectorial (Semantica)
        results = collection.query(query_texts=[prompt], n_results=3)
        contexto_vectorial = ""
        fuentes = []
        
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                contexto_vectorial += f"Opcion {i+1}:\nPregunta: {meta['pregunta']}\nRespuesta: {meta['respuesta']}\n\n"
                fuentes.append(meta) # Guardamos para mostrar luego

        # 2. Busqueda por Palabras Clave (Puntos) - Respaldo
        resultados_puntos = buscar_por_puntos(prompt, df_global)
        contexto_puntos = ""
        for row in resultados_puntos:
            contexto_puntos += f"Coincidencia exacta: {row['Pregunta_Hibrida']} - {row['Respuesta']}\n"
            # Agregar a fuentes si no estaba
            if not any(f['pregunta'] == row['Pregunta_Hibrida'] for f in fuentes):
                fuentes.append({
                    "pregunta": row['Pregunta_Hibrida'],
                    "respuesta": row['Respuesta'],
                    "video": row['Video']
                })

        # 3. Generar Respuesta con Groq
        full_prompt = f"""
        Eres un asistente de soporte tecnico experto en el sistema Baris.
        Usa la siguiente informacion recuperada de la base de conocimientos para responder al usuario.
        
        INFORMACION RECUPERADA (VECTORIAL):
        {contexto_vectorial}
        
        INFORMACION ADICIONAL (PALABRAS CLAVE):
        {contexto_puntos}
        
        PREGUNTA DEL USUARIO: {prompt}
        
        INSTRUCCIONES:
        - Responde de forma directa y amable.
        - Si la informacion contiene pasos numerados, usalos.
        - Si hay un ID de nota al final (ej: 3528), mencionalo discretamente.
        - Si la informacion no es suficiente, di que no lo sabes, no inventes.
        """
        
        try:
            client_groq = Groq(api_key=st.secrets["GROQ_API_KEY"])
            chat_completion = client_groq.chat.completions.create(
                messages=[{"role": "user", "content": full_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
            )
            respuesta = chat_completion.choices[0].message.content
            message_placeholder.markdown(respuesta)
            
            # Mostrar fuentes en desplegable
            with st.expander("Ver fuentes y detalles tecnicos"):
                if not fuentes:
                    st.write("No se encontraron coincidencias exactas en la base de datos.")
                else:
                    st.write(f"Se consultaron {len(fuentes)} registros:")
                    for f in fuentes:
                        st.divider()
                        st.markdown(f"**Pregunta:** {f['pregunta']}")
                        st.info(f"**Solucion:** {f['respuesta']}")
                        if f['video']:
                            st.markdown(f"**Video:** {f['video']}")

            st.session_state.messages.append({"role": "assistant", "content": respuesta})

        except Exception as e:
            st.error(f"Error generando respuesta: {e}")