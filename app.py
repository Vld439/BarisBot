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

# --- CONFIGURACION DE LA PAGINA ---
st.set_page_config(page_title="BarisBot soporte interno", layout="wide")

# --- FUNCION DE RESPALDO  ---
def consultar_ia_blindada(cliente_groq, prompt, max_tokens=500):
    """
    Intenta con varios modelos para asegurar la respuesta.
    """
    modelos = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
    
    for modelo in modelos:
        try:
            chat = cliente_groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=modelo,
                temperature=0.2,
                max_tokens=max_tokens
            )
            return chat.choices[0].message.content
        except:
            continue     
    return None

# --- FUNCIONES DE MANTENIMIENTO ---

def obtener_csv_actual_github(repo_name):
    try:
        if "GITHUB_TOKEN" not in st.secrets:
            return pd.DataFrame(columns=["ID", "Pregunta_Hibrida", "Respuesta", "Video"])
        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(repo_name)
        file_content = repo.get_contents("base_conocimiento_HIBRIDA.csv")
        csv_data = file_content.decoded_content.decode("utf-8")
        return pd.read_csv(io.StringIO(csv_data))
    except:
        return pd.DataFrame(columns=["ID", "Pregunta_Hibrida", "Respuesta", "Video"])

def procesar_pdf_reporte_limpio(file_obj, df_actual):
    # 1. Configurar Cliente Groq
    client = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except:
        pass

    # Lista de IDs existentes (limpieza de formato)
    ids_existentes = set()
    if not df_actual.empty and "ID" in df_actual.columns:
        # Convertimos a string y quitamos decimales .0 si existen
        ids_existentes = set(str(x).replace('.0', '').strip() for x in df_actual["ID"].tolist())
        
        # Mantenemos datos viejos
        cols = ["ID", "Pregunta_Hibrida", "Respuesta", "Video"]
        valid_cols = [c for c in cols if c in df_actual.columns]
        datos_finales = df_actual[valid_cols].values.tolist()
    else:
        datos_finales = []

    buffer_texto_completo = ""
    registros_encontrados = 0

    # Leemos todo el PDF a texto
    try:
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                # Usamos extract_text simple para capturar el flujo crudo (con comillas y comas)
                texto_pag = page.extract_text() 
                if texto_pag:
                    buffer_texto_completo += texto_pag + "\n"
    except Exception as e:
        st.error(f"Error leyendo PDF: {e}")
        return None, None, 0

    # --- LÓGICA ESPECÍFICA PARA REPORTE JHF (FORMATO CSV EN PDF) ---
    lineas = buffer_texto_completo.split('\n')
    
    bloque_actual_id = None
    bloque_actual_texto = ""
    
    # ELEMENTOS VISUALES DE PROGRESO
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_lineas = len(lineas)

    for i, linea in enumerate(lineas):
        linea_raw = linea.strip()
        if not linea_raw: continue

        # REGEX: Detectar inicio de registro: "3434" o 3434
        match_id = re.search(r'^"?(\d{3,6})"?', linea_raw)
        
        # ¿Es una línea de continuación? (Empieza con ,,,)
        es_continuacion = linea_raw.startswith(",,,")
        
        es_nuevo_registro = False
        if match_id and not es_continuacion:
            # Filtro extra: asegurar que no sea una fecha disfrazada
            if not re.search(r'^"?\d{2}/\d{2}/\d{2}', linea_raw):
                es_nuevo_registro = True

        if es_nuevo_registro:
            # -- PROCESAR EL BLOQUE ANTERIOR --
            if bloque_actual_id:
                # Actualizamos el texto en pantalla para ver qué hace
                status_text.text(f"Procesando ID: {bloque_actual_id}...")
                guardar_bloque(bloque_actual_id, bloque_actual_texto, ids_existentes, datos_finales, client)
            
            # Iniciar nuevo bloque
            bloque_actual_id = match_id.group(1)
            bloque_actual_texto = linea_raw 
            registros_encontrados += 1
            
        else:
            # Es continuación del bloque actual
            if bloque_actual_id:
                # Limpiar las comas de continuación ",,," del inicio
                linea_limpia = re.sub(r'^,+', '', linea_raw)
                bloque_actual_texto += " " + linea_limpia

        # Actualizar barra cada 20 lineas
        if i % 20 == 0: progress_bar.progress(min(i / total_lineas, 1.0))

    # Procesar el último bloque pendiente
    if bloque_actual_id:
        status_text.text(f"Finalizando ID: {bloque_actual_id}...")
        guardar_bloque(bloque_actual_id, bloque_actual_texto, ids_existentes, datos_finales, client)

    progress_bar.progress(1.0)
    status_text.empty()
    
    # DEBUG INFO VISUAL
    st.sidebar.info(f" Registros leídos del PDF: {registros_encontrados}")
    
    # Crear DataFrame
    df_nuevo = pd.DataFrame(datos_finales, columns=["ID", "Pregunta_Hibrida", "Respuesta", "Video"])
    # Eliminar duplicados por ID (mantener el último)
    df_nuevo = df_nuevo.drop_duplicates(subset=["ID"], keep="last")
    
    csv_buffer = io.StringIO()
    df_nuevo.to_csv(csv_buffer, index=False)
    
    cont_real = len(df_nuevo) - len(ids_existentes)
    # Si la base estaba vacía, todos son nuevos
    if len(ids_existentes) == 0: cont_real = len(df_nuevo)
    
    st.sidebar.info(f"Registros NUEVOS a guardar: {cont_real}")

    return csv_buffer.getvalue(), df_nuevo, cont_real

def guardar_bloque(id_nota, texto, ids_existentes, datos_finales, client):
    """Limpia el texto sucio tipo CSV y separa pregunta/respuesta"""
    
    # Si ya existe, IGNORAR (Logica Incremental)
    if str(id_nota) in ids_existentes:
        return

    # 1. Limpieza masiva de caracteres basura del CSV
    # Reemplazamos comillas dobles y comas residuales
    texto_limpio = texto.replace('"', '').strip()
    
    # 2. SEPARACION PREGUNTA / RESPUESTA
    # Buscamos "OBS:" o "Obs:"
    partes = re.split(r'OBS:?', texto_limpio, maxsplit=1, flags=re.IGNORECASE)
    
    pregunta_sucia = partes[0]
    
    # 3. LIMPIEZA DE LA CABECERA (Donde está el ID, Fecha y Codigo)
    # Formato tipico: 3434 05/01/07 0663 Procedimiento...
    
    # a) Quitar ID del inicio
    pregunta = re.sub(r'^\d+\s*', '', pregunta_sucia)
    # b) Quitar fechas dd/mm/yy
    pregunta = re.sub(r'\d{2}/\d{2}/\d{2}\s*', '', pregunta)
    # c) Quitar codigos de 3-4 letras/numeros (JHF, 0663)
    pregunta = re.sub(r'^[A-Z0-9]{3,4}\s+', '', pregunta)
    # d) Quitar comas iniciales sueltas
    pregunta = re.sub(r'^,\s*', '', pregunta).strip()
    
    if len(partes) > 1:
        respuesta = "OBS: " + partes[1].strip()
    else:
        # Fallback si no hay OBS: Buscar "1."
        partes_num = re.split(r'(?=\s1\.)', pregunta, maxsplit=1)
        if len(partes_num) > 1:
            pregunta = partes_num[0].strip()
            respuesta = partes_num[1].strip()
        else:
            respuesta = "Ver detalle en el manual."

    # Validacion minima
    if len(pregunta) < 3: return

    # --- IA GENERADORA DE SINONIMOS ---
    preg_hibrida = pregunta
    if client:
        try:
            # Limitamos el texto para no confundir a la IA
            texto_ia = pregunta[:200]
            prompt = f"Genera 3 sinónimos técnicos breves para buscar: '{texto_ia}'. Solo palabras separadas por coma."
            sinonimos = consultar_ia_blindada(client, prompt, max_tokens=40)
            if sinonimos:
                preg_hibrida = f"{pregunta} ({sinonimos})"
                time.sleep(0.1) # Pausa muy breve para velocidad
        except:
            pass 

    # Detectar Video
    video = ""
    match_vid = re.search(r'(https?://youtu\.?be\S+)', texto)
    if match_vid: 
        # Limpiar posible basura al final del link
        video = match_vid.group(1).replace(',', '').strip()

    datos_finales.append([str(id_nota), preg_hibrida, respuesta, video])


def actualizar_github(content, repo_name):
    try:
        if "GITHUB_TOKEN" not in st.secrets: return False, "Falta Token"
        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(repo_name)
        path = "base_conocimiento_HIBRIDA.csv"
        msg = "Actualizacion base limpia"
        try:
            contents = repo.get_contents(path)
            repo.update_file(path, msg, content, contents.sha)
        except:
            repo.create_file(path, msg, content)
        return True, "OK"
    except Exception as e:
        return False, str(e)

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Panel de Control")
    st.info("Sistema listo para archivo JHF (FRX)")
    uploaded = st.file_uploader("Sube el PDF Limpio", type="pdf")
    
    if uploaded and st.button("Actualizar Base"):
        REPO = "Vld439/BarisBot" 
        
        # Usamos un contenedor vacío para ir mostrando el progreso sin borrar lo anterior
        status_box = st.empty()
        log_box = st.empty()
        
        with st.status("Procesando...", expanded=True) as status:
            status.write("Descargando base actual...")
            df_actual = obtener_csv_actual_github(REPO)
            
            status.write("Analizando PDF y consultando IA...")
            # Llamamos a la función
            csv_str, df_final, cont = procesar_pdf_reporte_limpio(uploaded, df_actual)
            
            if len(df_final) > 0:
                status.write(f"Guardando {cont} registros nuevos en GitHub...")
                ok, msg = actualizar_github(csv_str, REPO)
                
                if ok: 
                    status.update(label="¡Completado!", state="complete", expanded=False)
                    # AQUÍ EL CAMBIO: Quitamos el st.rerun() y ponemos globos
                    st.balloons()
                    st.success(f"¡ÉXITO TOTAL!\n\nSe han guardado {cont} registros nuevos.\n\nLa base ahora tiene {len(df_final)} preguntas.")
                    st.warning(" Nota: No cierres ni recargues la página hasta ver este mensaje.")
                else: 
                    status.update(label="Error", state="error")
                    st.error(f" Error al subir a GitHub: {msg}")
            else:
                status.update(label="Sin datos", state="error")
                st.error(" Error: El PDF no generó registros válidos. Revisa si el formato cambió.")
# --- CHATBOT ---
st.title("BarisBot soporte interno")

@st.cache_resource
def load_db():
    if not os.path.exists("base_conocimiento_HIBRIDA.csv"): return None, None
    if os.path.exists("./cerebro_db"): shutil.rmtree("./cerebro_db")
    try:
        client = chromadb.PersistentClient(path="./cerebro_db")
        emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        coll = client.get_or_create_collection("baris_manual", embedding_function=emb)
        df = pd.read_csv("base_conocimiento_HIBRIDA.csv").fillna("")
        ids, docs, metas = [], [], []
        for _, r in df.iterrows():
            ids.append(str(r['ID']))
            docs.append(f"{r['Pregunta_Hibrida']} \n {r['Respuesta']}")
            metas.append({"p": r['Pregunta_Hibrida'], "r": r['Respuesta'], "v": r['Video']})
        batch = 100
        for i in range(0, len(ids), batch):
            coll.add(ids=ids[i:i+batch], documents=docs[i:i+batch], metadatas=metas[i:i+batch])
        return coll, df
    except: return None, None

collection, df_global = load_db()

if not collection:
    st.warning("Esperando base de datos...")
    st.stop()

if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Consulta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        contexto, fuentes = "", []
        try:
            res = collection.query(query_texts=[prompt], n_results=3)
            if res['documents']:
                for i, meta in enumerate(res['metadatas'][0]):
                    contexto += f"- {meta['p']}: {meta['r']}\n\n"
                    fuentes.append(meta)
        except: pass
        
        # Logica Fallback keywords
        palabras = prompt.lower().split()
        for _, row in df_global.iterrows():
            txt = (str(row['Pregunta_Hibrida']) + " " + str(row['Respuesta'])).lower()
            if all(p in txt for p in palabras) and not any(f['p'] == row['Pregunta_Hibrida'] for f in fuentes):
                contexto += f"- {row['Pregunta_Hibrida']}: {row['Respuesta']}\n"
                fuentes.append({"p": row['Pregunta_Hibrida'], "r": row['Respuesta'], "v": row['Video']})
        
        if not fuentes:
            resp_final = "No encontré información en el manual."
        else:
            client = None
            try: 
                if "GROQ_API_KEY" in st.secrets: client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            except: pass
            
            prompt_ia = f"Eres experto en Baris. Responde SOLO con esto:\n{contexto}\nPregunta: {prompt}"
            resp_ia = consultar_ia_blindada(client, prompt_ia) if client else None
            
            if resp_ia: resp_final = resp_ia
            else:
                resp_final = "**Resultados:**\n\n"
                for f in fuentes: resp_final += f"**{f['p']}**\n{f['r']}\n\n"
        
        st.markdown(resp_final)
        st.session_state.messages.append({"role": "assistant", "content": resp_final})