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

# --- FUNCION 1: MAQUILLAJE DE TEXTO ---
def embellecer_respuesta(texto):
    if not texto: return ""
    texto = str(texto).replace("OBS:", "**OBSERVACION:**")
    # Detecta pasos numerados (1-, 1., 2)) y agrega saltos de linea
    patron = r'(\s\d+[\-\.\)]\s?)'
    return re.sub(patron, r'\n\n\1', texto)

# --- FUNCION 2: IA BLINDADA ---
def consultar_ia_blindada(cliente_groq, prompt, max_tokens=600, temperatura=0.1):
    modelos = ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama-3.3-70b-versatile"]
    for modelo in modelos:
        try:
            chat = cliente_groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=modelo, temperature=temperatura, max_tokens=max_tokens
            )
            return chat.choices[0].message.content
        except: 
            time.sleep(0.5)
            continue     
    return None

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
        msg = "Actualizacion automatica BarisBot"
        try:
            contents = repo.get_contents(path)
            repo.update_file(path, msg, content, contents.sha)
        except:
            repo.create_file(path, msg, content)
        return True, "OK"
    except Exception as e: return False, str(e)

# --- FUNCION 4: PROCESAR PDF ---
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
                # Prompt mas simple para evitar errores
                prompt = f"Dame 3 palabras clave o sinonimos para: '{pregunta}'. Solo palabras separadas por coma."
                sinonimos = consultar_ia_blindada(client, prompt, max_tokens=30)
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
            status.write("Descargando base actual...")
            df_actual = obtener_csv_actual_github(REPO)
            status.write("Analizando PDF...")
            df_nuevo, cant, msg = procesar_pdf_completo(uploaded_file, df_actual)
            if df_nuevo is not None and cant > 0:
                status.write(f"Guardando {cant} registros...")
                ok, gh_msg = subir_csv_a_github(df_nuevo, REPO)
                if ok:
                    df_nuevo.to_csv(FILE_PATH, index=False)
                    status.update(label="Listo", state="complete", expanded=False)
                    st.success("Base actualizada.")
                    st.warning("Recarga la pagina.")
                else:
                    status.update(label="Error", state="error"); st.error(f"Error GitHub: {gh_msg}")
            else:
                status.update(label="Error", state="error"); st.error(f"Error: {msg}")

st.title("BarisBot soporte interno")
if not collection: st.warning("Esperando base de datos..."); st.stop()

if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        
        # Logica de memoria corta para preguntas tipo "y el video?"
        query_busqueda = prompt
        if len(prompt.split()) < 4 and len(st.session_state.messages) > 1:
            ultimo_user = [m['content'] for m in st.session_state.messages if m['role'] == 'user'][-2]
            query_busqueda = f"{ultimo_user} {prompt}"
        
        contexto, fuentes = "", []
        
        # 1. Busqueda Vectorial
        try:
            res = collection.query(query_texts=[query_busqueda], n_results=4) # Traemos mas resultados (4)
            if res['documents']:
                for i, meta in enumerate(res['metadatas'][0]):
                    r_bonita = embellecer_respuesta(meta['r'])
                    contexto += f"- {meta['p']}: {r_bonita}\n\n"
                    fuentes.append({"p": meta['p'], "r": r_bonita, "v": meta['v']})
        except: pass
        
        # 2. Busqueda Keyword (Respaldo por si vector falla)
        palabras = prompt.lower().split()
        if len(palabras) > 1:
            for _, row in df_global.iterrows():
                txt = (str(row['Pregunta_Hibrida']) + " " + str(row['Respuesta'])).lower()
                if all(p in txt for p in palabras) and not any(f['p'] == row['Pregunta_Hibrida'] for f in fuentes):
                    r_bonita = embellecer_respuesta(row['Respuesta'])
                    contexto += f"- {row['Pregunta_Hibrida']}: {r_bonita}\n"
                    fuentes.append({"p": row['Pregunta_Hibrida'], "r": r_bonita, "v": row['Video']})
        
        if not fuentes:
            resp_final = "No encontre informacion especifica en el manual."
        else:
            client = None
            try: 
                if "GROQ_API_KEY" in st.secrets: client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            except: pass
            
            resp_ia = None
            if client:
                # Prompt estricto pero con orden de limpiar
                prompt_ia = f"""
                Eres un asistente de soporte.
                CONTEXTO (Puede estar sucio/desordenado):
                {contexto}
                PREGUNTA: {prompt}
                
                REGLAS:
                1. Responde basandote SOLAMENTE en el CONTEXTO.
                2. Si el texto esta desordenado, limpialo y ordenalo en pasos.
                3. NO digas 'Lo siento' ni 'No hay informacion' si ves datos utiles en el contexto.
                4. Si realmente no hay nada util, di "NO_DATA".
                """
                resp_ia = consultar_ia_blindada(client, prompt_ia)
            
            # --- LOGICA DE FALLBACK (VALVULA DE SEGURIDAD) ---
            
            # 1. Recolectar videos siempre
            videos_str = ""
            for f in fuentes:
                if f['v'] and len(f['v']) > 10:
                    videos_str += f"\n\n**Video Tutorial:** {f['v']}"
            
            # 2. Decidir si usamos IA o Texto Crudo
            frases_rechazo = ["lo siento", "no tengo informacion", "no_data", "no aparece", "no se menciona"]
            
            usar_ia = False
            if resp_ia:
                usar_ia = True
                for rechazo in frases_rechazo:
                    if rechazo in resp_ia.lower():
                        usar_ia = False # La IA se rindio, forzamos texto crudo
                        break
            
            if usar_ia:
                resp_final = resp_ia + videos_str
            else:
                # Fallback: Mostrar lo que encontro la base de datos sin filtrar
                resp_final = "**Informacion del Manual (Sin procesar):**\n\n"
                for f in fuentes:
                    resp_final += f"### {f['p']}\n{f['r']}\n\n"
                resp_final += videos_str

        st.markdown(resp_final)
        st.session_state.messages.append({"role": "assistant", "content": resp_final})