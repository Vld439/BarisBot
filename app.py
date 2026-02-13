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

# --- CONFIGURACION ---
st.set_page_config(page_title="BarisBot Soporte Interno", layout="wide")

# --- FUNCION IA (Vuelve a ser flexible para entender el texto sucio) ---
def consultar_ia(cliente_groq, prompt):
    try:
        chat = cliente_groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", # Modelo rapido y bueno resumiendo
            temperature=0.3, # Un poco de creatividad para entender el texto sucio
            max_tokens=800
        )
        return chat.choices[0].message.content
    except: return None

# --- FUNCIONES GITHUB ---
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
        msg = "Actualizacion BarisBot"
        try:
            contents = repo.get_contents(path)
            repo.update_file(path, msg, content, contents.sha)
        except:
            repo.create_file(path, msg, content)
        return True, "OK"
    except Exception as e: return False, str(e)

# --- PROCESAMIENTO PDF ---
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
        if i % 10 == 0: 
            progress_bar.progress(min(i/total, 1.0))
            status_text.text(f"Procesando {i}/{total}...")

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
                # Prompt para sinónimos
                prompt = f"Dame 3 palabras clave o sinonimos para: '{pregunta}'. Solo palabras separadas por coma."
                sinonimos = consultar_ia(client, prompt)
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

# --- INIT ---
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
        with st.status("Procesando...", expanded=True) as status:
            status.write("Descargando base...")
            df_actual = obtener_csv_actual_github(REPO)
            status.write("Procesando PDF...")
            df_nuevo, cant, msg = procesar_pdf_completo(uploaded_file, df_actual)
            if df_nuevo is not None and cant > 0:
                status.write("Subiendo a GitHub...")
                ok, gh_msg = subir_csv_a_github(df_nuevo, REPO)
                if ok:
                    df_nuevo.to_csv(FILE_PATH, index=False)
                    status.update(label="Listo", state="complete", expanded=False)
                    st.success("Actualizado.")
                else:
                    status.error(f"Error GitHub: {gh_msg}")
            else:
                status.error(f"Error: {msg}")

st.title("BarisBot soporte interno")
if not collection: st.warning("Base vacia. Sube el PDF."); st.stop()

if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        
        # 1. Recuperar contexto (Inteligencia de busqueda)
        contexto = ""
        fuentes_videos = []
        
        # Busqueda Vectorial
        try:
            res = collection.query(query_texts=[prompt], n_results=4) # Traemos mas contexto (4)
            if res['documents']:
                for i, meta in enumerate(res['metadatas'][0]):
                    # Pasamos el texto sucio tal cual a la IA, ella sabra limpiarlo
                    contexto += f"--- OPCION {i+1} ---\nTITULO: {meta['p']}\nCONTENIDO: {meta['r']}\n\n"
                    if meta['v']: fuentes_videos.append(meta['v'])
        except: pass
        
        # Busqueda Keyword (Respaldo)
        palabras = prompt.lower().split()
        if len(palabras) > 1:
            for _, row in df_global.iterrows():
                txt_full = (str(row['Pregunta_Hibrida']) + " " + str(row['Respuesta'])).lower()
                if all(p in txt_full for p in palabras):
                    contexto += f"--- OPCION EXTRA ---\nTITULO: {row['Pregunta_Hibrida']}\nCONTENIDO: {row['Respuesta']}\n\n"
                    if row['Video']: fuentes_videos.append(row['Video'])

        # 2. Generar respuesta
        if not contexto:
            resp_final = "No encontre informacion en el manual."
        else:
            # AQUI ESTA LA MAGIA: Le pedimos a la IA que ACTUE como un limpiador
            prompt_ia = f"""
            Eres un experto en soporte tecnico. Tienes la siguiente informacion extraida de una base de datos sucia:
            
            {contexto}
            
            TU TAREA:
            1. Responde a la pregunta del usuario: "{prompt}"
            2. USA SOLO la informacion de arriba.
            3. El texto de arriba esta sucio (tiene codigos, fechas, formatos raros). LIMPIALO.
            4. Si hay pasos numerados, ponlos en una lista ordenada clara.
            5. No menciones "segun el contexto", solo da la respuesta directa.
            """
            
            client = None
            try: 
                if "GROQ_API_KEY" in st.secrets: client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            except: pass

            if client:
                resp_ia = consultar_ia(client, prompt_ia)
                if resp_ia:
                    resp_final = resp_ia
                    # Agregamos los videos al final si existen
                    videos_unicos = list(set(fuentes_videos))
                    if videos_unicos:
                        resp_final += "\n\n**Videos Relacionados:**"
                        for v in videos_unicos:
                            resp_final += f"\n- {v}"
                else:
                    resp_final = "Error conectando con la IA, mostrando datos crudos:\n" + contexto
            else:
                resp_final = "Falta API Key de Groq.\n" + contexto

        st.markdown(resp_final)
        st.session_state.messages.append({"role": "assistant", "content": resp_final})