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
import re

# Configuracion de la pagina
st.set_page_config(page_title="Soporte Baris", layout="wide")

# --- FUNCIONES DE MANTENIMIENTO (BARRA LATERAL) ---

def procesar_pdf_y_generar_csv(file_obj):
    """
    Lee el PDF subido en memoria y devuelve el DataFrame y el texto CSV.
    Usa REGEX para extraccion estructural y IA para sinonimos (con manejo de errores).
    """
    
    # 1. Configurar Groq (Opcional, si falla seguimos sin sinonimos)
    client = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except:
        pass # Se procesara en modo texto plano

    # 2. Leer PDF desde memoria
    texto_completo = ""
    try:
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                # Limpieza basica de encabezados de WinJes
                lines = t.split('\n')
                # Filtramos lineas de encabezado y pie de pagina
                cleaned_lines = [
                    l for l in lines 
                    if "Pág.:" not in l 
                    and "Fecha:[-]" not in l 
                    and "Preguntas Frecuentes" not in l
                    and "ORDEN:Hora" not in l
                ]
                texto_completo += "\n".join(cleaned_lines) + "\n"
    except Exception as e:
        return None, None

    # 3. EXTRACCION POR PATRONES (REGEX)
    # Busca: Texto previo -> "1." -> Texto respuesta -> "(ID)"
    # (?s) permite que el punto coincida con saltos de linea
    patron = r"(?s)(.*?)\n\s*1\.\s+(.*?)\s*\((\d+)\)"
    
    coincidencias = re.findall(patron, texto_completo)
    
    datos = []
    total = len(coincidencias)
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    if total == 0:
        st.error("[ERROR] No se encontraron preguntas con el formato estandar (Pregunta -> 1. Respuesta -> (ID)).")
        return None, None

    for i, (pregunta_raw, respuesta_raw, id_nota) in enumerate(coincidencias):
        status_text.text(f"Procesando registro {i+1} de {total} (ID: {id_nota})...")
        
        # Limpieza de texto
        # Tomamos la ultima linea de la pregunta si viene con basura anterior
        lineas_pregunta = pregunta_raw.strip().split('\n')
        pregunta_limpia = lineas_pregunta[-1].strip() if lineas_pregunta else "Sin titulo"
        # Si la ultima linea es muy corta (menos de 3 chars), tomamos todo el bloque
        if len(pregunta_limpia) < 3:
            pregunta_limpia = pregunta_raw.strip()

        respuesta_final = "1. " + respuesta_raw.strip() # Reagregamos el 1.
        pregunta_hibrida = pregunta_limpia

        # 4. ENRIQUECIMIENTO CON IA (INTENTO)
        # Solo pedimos sinonimos si tenemos cliente. Si falla, seguimos.
        if client:
            try:
                # Prompt muy corto para gastar pocos tokens
                prompt = f"Dame 3 palabras clave o sinonimos para buscar esta duda tecnica: '{pregunta_limpia}'. Formato: palabra1, palabra2, palabra3. Sin explicaciones."
                
                chat = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=60 # Limite estricto para ahorrar
                )
                sinonimos = chat.choices[0].message.content.replace("\n", " ").strip()
                pregunta_hibrida = f"{pregunta_limpia} ({sinonimos})"
                
            except Exception as e:
                # Si da error 429 (Rate Limit) u otro, ignoramos y usamos la pregunta original
                # No imprimimos error para no ensuciar la interfaz
                pass 

        datos.append([id_nota, pregunta_hibrida, respuesta_final, ""])
        
        progress_bar.progress((i + 1) / total)
        
        # Pausa breve para no saturar API si funciona
        if i % 10 == 0: time.sleep(0.2)

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
                
                status.write("Analizando estructura del PDF...")
                csv_content, df_preview = procesar_pdf_y_generar_csv(uploaded_file)
                
                if csv_content:
                    status.write(f"Se extrajeron {len(df_preview)} registros correctamente.")
                    st.dataframe(df_preview.head(3))
                    
                    status.write("Conectando con GitHub...")
                    
                    # --- CONFIGURACION DEL REPOSITORIO ---
                    REPO_NAME = "Vld439/BarisBot" 
                    # -------------------------------------
                    
                    exito, mensaje = actualizar_github(csv_content, REPO_NAME)
                    
                    if exito:
                        status.update(label="Proceso Finalizado", state="complete", expanded=False)
                        st.success(mensaje)
                        st.info("Por favor espera unos minutos a que Streamlit detecte el cambio y recarga la pagina.")
                    else:
                        status.update(label="Error", state="error")
                        st.error(mensaje)
                else:
                    status.update(label="Error", state="error")
                    st.error("No se pudo procesar el PDF. Verifica que sea el reporte correcto.")
    
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
    
    # Verificar si existe el CSV generado
    if not os.path.exists(csv_path):
        return None, None

    # Limpieza preventiva si el CSV cambio
    if os.path.exists(db_path):
        try:
            # En entorno local podria dar error de permisos, en nube no
            shutil.rmtree(db_path) 
        except:
            pass
    
    try:
        client = chromadb.PersistentClient(path=db_path)
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        collection = client.get_or_create_collection(name="manual_baris", embedding_function=emb_fn)
        
        # Leer CSV y poblar DB
        df = pd.read_csv(csv_path).fillna("")
        
        if collection.count() == 0:
            ids = []
            docs = []
            metas = []
            
            for _, row in df.iterrows():
                doc_id = str(row['ID'])
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
    except Exception as e:
        st.error(f"Error cargando base de datos: {e}")
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

# --- INTERFAZ DE CHAT (OPTIMIZADA A PRUEBA DE FALLOS) ---

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu consulta aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 1. Recuperacion de informacion (No gasta cuota)
        contexto_vectorial = ""
        fuentes = []
        
        # Busqueda Vectorial
        try:
            results = collection.query(query_texts=[prompt], n_results=3)
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    if i < len(results['metadatas'][0]):
                        meta = results['metadatas'][0][i]
                        contexto_vectorial += f"Pregunta: {meta['pregunta']}\nRespuesta: {meta['respuesta']}\n\n"
                        fuentes.append(meta)
        except:
            pass # Si falla vector, seguimos

        # Busqueda por Palabras Clave (Respaldo)
        resultados_puntos = buscar_por_puntos(prompt, df_global)
        contexto_puntos = ""
        for row in resultados_puntos:
            contexto_puntos += f"Coincidencia: {row['Pregunta_Hibrida']} - {row['Respuesta']}\n"
            # Agregar si no esta repetido
            if not any(f['pregunta'] == row['Pregunta_Hibrida'] for f in fuentes):
                fuentes.append({
                    "pregunta": row['Pregunta_Hibrida'],
                    "respuesta": row['Respuesta'],
                    "video": row['Video']
                })

        # 2. Generacion de Respuesta
        if not fuentes:
            respuesta_final = "No encontre informacion relacionada en el manual."
            message_placeholder.markdown(respuesta_final)
        else:
            # Intentamos usar la IA para resumir
            try:
                full_prompt = f"""
                Eres un asistente tecnico. Responde la duda del usuario usando esta informacion del manual.
                
                INFORMACION DEL MANUAL:
                {contexto_vectorial}
                {contexto_puntos}
                
                PREGUNTA USUARIO: {prompt}
                
                Responde directo y conciso.
                """
                
                client_groq = Groq(api_key=st.secrets["GROQ_API_KEY"])
                chat_completion = client_groq.chat.completions.create(
                    messages=[{"role": "user", "content": full_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.1,
                )
                respuesta_final = chat_completion.choices[0].message.content
                message_placeholder.markdown(respuesta_final)
            
            except Exception as e:
                # SI LA IA FALLA (Error 429), entra aqui y muestra los datos crudos
                if "429" in str(e):
                    st.warning("Alerta: Limite de IA alcanzado. Mostrando resultados directos del manual:")
                else:
                    st.error(f"Error de conexion: {e}")
                
                respuesta_final = ""
                for f in fuentes:
                    st.success(f"**Tema encontrado:** {f['pregunta']}")
                    st.markdown(f"{f['respuesta']}")
                    if f['video']:
                        st.markdown(f"**Video:** {f['video']}")
                    st.divider()
                    respuesta_final += f"**{f['pregunta']}**\n{f['respuesta']}\n\n"

        st.session_state.messages.append({"role": "assistant", "content": respuesta_final})