import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from github import Github, Auth
import pdfplumber
import io
import os
import time
import shutil
import re

# --- CONFIGURACION DE LA PAGINA ---
st.set_page_config(page_title="BarisBot Soporte Interno", layout="wide")

# --- FUNCION 1: EMBELLECER RESPUESTA ---
def embellecer_respuesta(texto):
    if not texto: return ""
    texto = str(texto).replace("OBS:", "**OBSERVACIÓN:**")
    patron = r'(\s\d+[\-\.\)]\s?)'
    return re.sub(patron, r'\n\n\1', texto)

# --- FUNCION 2: IA BLINDADA (ESTRICTA) ---
def consultar_ia_blindada(cliente_groq, prompt, max_tokens=500, temperatura=0.0): # <--- TEMPERATURA 0 (ROBOT)
    modelos = ["llama-3.1-8b-instant", "mixtral-8x7b-32768"] # Modelos más obedientes
    for modelo in modelos:
        try:
            chat = cliente_groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=modelo, temperature=temperatura, max_tokens=max_tokens
            )
            return chat.choices[0].message.content
        except: continue     
    return None

# --- FUNCIONES GITHUB Y PDF (Iguales que antes) ---
def obtener_csv_actual_github(repo_name):
    try:
        if "GITHUB_TOKEN" not in st.secrets:
            return pd.DataFrame(columns=["ID", "Pregunta_Hibrida", "Respuesta", "Video"])
        auth = Auth.Token(st.secrets["GITHUB_TOKEN"])
        g = Github(auth=auth)
        repo = g.get_repo(repo_name)
        file_content = repo.get_contents("base_conocimiento_HIBRIDA.csv")
        csv_data = file_content.decoded_content.decode("utf-8")
        return pd.read_csv(io.StringIO(csv_data))
    except:
        return pd.DataFrame(columns=["ID", "Pregunta_Hibrida", "Respuesta", "Video"])

def subir_csv_a_github(df, repo_name):
    try:
        if "GITHUB_TOKEN" not in st.secrets: return False, "Falta Token"
        auth = Auth.Token(st.secrets["GITHUB_TOKEN"])
        g = Github(auth=auth)
        repo = g.get_repo(repo_name)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        content = csv_buffer.getvalue()
        path = "base_conocimiento_HIBRIDA.csv"
        msg = "Actualización automática desde BarisBot"
        try:
            contents = repo.get_contents(path)
            repo.update_file(path, msg, content, contents.sha)
        except:
            repo.create_file(path, msg, content)
        return True, "OK"
    except Exception as e: return False, str(e)

def procesar_pdf_completo(file_obj, df_actual):
    client = None
    try:
        if "GROQ_API_KEY" in st.secrets: client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except: pass

    buffer_texto = ""
    try:
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text: buffer_texto += text + "\n"
    except Exception as e:
        return None, 0, f"Error leyendo PDF: {e}"

    lineas = buffer_texto.split('\n')
    bloque_id = None
    bloque_texto = ""
    registros_crudos = []

    for linea in lineas:
        linea = linea.strip()
        if not linea: continue
        match_id = re.search(r'^"?(\d{3,6})"?', linea)
        es_nuevo = False
        if match_id:
            if not re.search(r'^"?\d{2}/\d{2}/\d{2}', linea) and not linea.startswith(",,,"):
                es_nuevo = True
        if es_nuevo:
            if bloque_id: registros_crudos.append((bloque_id, bloque_texto))
            bloque_id = match_id.group(1)
            bloque_texto = linea
        else:
            if bloque_id:
                linea_clean = re.sub(r'^,+', '', linea)
                bloque_texto += " " + linea_clean
    if bloque_id: registros_crudos.append((bloque_id, bloque_texto))

    datos_finales = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(registros_crudos)
    count_nuevos = 0

    for i, (id_nota, texto_raw) in enumerate(registros_crudos):
        if i % 5 == 0: 
            progress_bar.progress(min(i/total, 1.0))
            status_text.text(f"Procesando registro {i}/{total}...")

        texto_limpio = texto_raw.replace('"', '').strip()
        partes = re.split(r'OBS:?', texto_limpio, maxsplit=1, flags=re.IGNORECASE)
        pregunta_sucia = partes[0]
        pregunta = re.sub(r'^\d+\s*', '', pregunta_sucia)
        pregunta = re.sub(r'\d{2}/\d{2}/\d{2}\s*', '', pregunta)
        pregunta = re.sub(r'^[A-Z0-9]{3,4}\s+', '', pregunta)
        pregunta = re.sub(r'^,\s*', '', pregunta).strip()
        
        if len(partes) > 1: respuesta = "OBS: " + partes[1].strip()
        else:
            partes_num = re.split(r'(?=\s1\.)', pregunta, maxsplit=1)
            if len(partes_num) > 1: pregunta = partes_num[0].strip(); respuesta = partes_num[1].strip()
            else: respuesta = "Ver detalle en el manual."

        if len(pregunta) < 3: continue

        video = ""
        match_vid = re.search(r'(https?://youtu\.?be\S+)', texto_raw)
        if match_vid: video = match_vid.group(1).replace(',', '').strip()

        preg_hibrida = pregunta
        if client:
            try:
                # Prompt de sinónimos
                prompt = f"Genera 3 sinónimos técnicos o palabras coloquiales (ej: borrar, anular, cancelar) para buscar: '{pregunta}'. Solo devuelve las palabras separadas por coma."
                sinonimos = consultar_ia_blindada(client, prompt, max_tokens=40)
                if sinonimos: preg_hibrida = f"{pregunta} ({sinonimos})"
                time.sleep(0.1)
            except: pass

        datos_finales.append([id_nota, preg_hibrida, respuesta, video])
        count_nuevos += 1

    progress_bar.progress(1.0)
    status_text.empty()
    df_nuevo = pd.DataFrame(datos_finales, columns=["ID", "Pregunta_Hibrida", "Respuesta", "Video"])
    df_nuevo = df_nuevo.drop_duplicates(subset=["ID"], keep="last")
    return df_nuevo, count_nuevos, "OK"

# --- INIT Y MAIN ---
REPO = "Vld439/BarisBot" 
FILE_PATH = "base_conocimiento_HIBRIDA.csv"

if not os.path.exists(FILE_PATH):
    df_gh = obtener_csv_actual_github(REPO)
    if not df_gh.empty: df_gh.to_csv(FILE_PATH, index=False)

@st.cache_resource
def init_db():
    if not os.path.exists(FILE_PATH): return None, None
    if os.path.exists("./cerebro_db"): shutil.rmtree("./cerebro_db")
    try:
        client = chromadb.PersistentClient(path="./cerebro_db")
        emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        coll = client.get_or_create_collection("baris_manual", embedding_function=emb)
        df = pd.read_csv(FILE_PATH).fillna("")
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

collection, df_global = init_db()

with st.sidebar:
    st.header("Panel de Control")
    uploaded_file = st.file_uploader("Actualizar Manual (PDF)", type="pdf")
    if uploaded_file and st.button("Procesar y Actualizar"):
        with st.status("Iniciando actualización...", expanded=True) as status:
            status.write("Descargando versión actual...")
            df_actual = obtener_csv_actual_github(REPO)
            status.write("Leyendo PDF y generando sinónimos...")
            df_nuevo, cant, msg = procesar_pdf_completo(uploaded_file, df_actual)
            if df_nuevo is not None and cant > 0:
                status.write(f"Subiendo {cant} registros a GitHub...")
                ok, gh_msg = subir_csv_a_github(df_nuevo, REPO)
                if ok:
                    df_nuevo.to_csv(FILE_PATH, index=False)
                    status.update(label="¡Éxito!", state="complete", expanded=False)
                    st.balloons()
                    st.success(f"Base actualizada con {len(df_nuevo)} registros.")
                    st.warning("Recarga la página para usar los nuevos datos.")
                else:
                    status.update(label="Error GitHub", state="error"); st.error(f"Error: {gh_msg}")
            else:
                status.update(label="Error Lectura", state="error"); st.error(f"Error: {msg}")

st.title("BarisBot soporte interno")
if not collection: st.warning("Base vacía. Sube el PDF."); st.stop()

if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        contexto, fuentes = "", []
        
        # 1. BUSQUEDA
        try:
            res = collection.query(query_texts=[prompt], n_results=3) # Trae top 3
            if res['documents']:
                for i, meta in enumerate(res['metadatas'][0]):
                    r_bonita = embellecer_respuesta(meta['r'])
                    contexto += f"- {meta['p']}: {r_bonita}\n\n"
                    fuentes.append({"p": meta['p'], "r": r_bonita, "v": meta['v']})
        except: pass
        
        palabras = prompt.lower().split()
        for _, row in df_global.iterrows():
            txt = (str(row['Pregunta_Hibrida']) + " " + str(row['Respuesta'])).lower()
            if all(p in txt for p in palabras) and not any(f['p'] == row['Pregunta_Hibrida'] for f in fuentes):
                r_bonita = embellecer_respuesta(row['Respuesta'])
                contexto += f"- {row['Pregunta_Hibrida']}: {r_bonita}\n"
                fuentes.append({"p": row['Pregunta_Hibrida'], "r": r_bonita, "v": row['Video']})
        if not fuentes:
            resp_final = "No encontré información en el manual sobre ese tema. Por favor intenta con otras palabras o revisa si el tema existe en el PDF."
        else:
            client = None
            try: 
                if "GROQ_API_KEY" in st.secrets: client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            except: pass
            
            if client:
                prompt_ia = f"""
                ERES UN ASISTENTE DE SOPORTE TÉCNICO ESTRICTO.
                
                TU FUENTE DE VERDAD ES ÚNICAMENTE ESTE CONTEXTO:
                {contexto}
                
                PREGUNTA DEL USUARIO: {prompt}
                
                REGLAS ABSOLUTAS:
                1. SOLO responde usando la información del CONTEXTO de arriba.
                2. SI LA RESPUESTA NO ESTÁ EN EL CONTEXTO, DI: "La información encontrada no responde exactamente a tu pregunta".
                3. NO INVENTES PASOS. NO INVENTES LINKS DE YOUTUBE.
                4. Si hay pasos numerados en el contexto, úsalos tal cual.
                5. Sé conciso.
                """
                resp_ia = consultar_ia_blindada(client, prompt_ia)
                resp_final = resp_ia if resp_ia else "Error en IA."
            else:
                resp_final = "**Resultados encontrados:**\n\n"
                for f in fuentes:
                    vid = f"\n📺 [Ver Video]({f['v']})" if f['v'] else ""
                    resp_final += f"### {f['p']}\n{f['r']}{vid}\n\n"
        
        st.markdown(resp_final)
        st.session_state.messages.append({"role": "assistant", "content": resp_final})