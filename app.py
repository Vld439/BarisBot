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

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Soporte Baris", layout="wide")

# --- FUNCIÓN "CAJA DE CAMBIOS" ---
def consultar_ia_blindada(cliente_groq, prompt, max_tokens=500):
    """
    Intenta con varios modelos en orden. Si uno falla, salta al siguiente.
    Si todos fallan, devuelve None pero NO rompe la app.
    """
    modelos = [
        "llama-3.3-70b-versatile",  # 1. El Potente (Ideal)
        "llama-3.1-8b-instant",     # 2. El Rápido (Respaldo)
        "mixtral-8x7b-32768",       # 3. La Alternativa
        "gemma2-9b-it"              # 4. El Último recurso
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
        except Exception as e:
            # Si falla (Error 429 o lo que sea), probamos el siguiente silenciosamente
            continue
            
    return None # Si llegamos aquí, es que no hay IA disponible hoy.

# --- FUNCIONES DE MANTENIMIENTO ---

def procesar_pdf_y_generar_csv(file_obj):
    # 1. Configurar Cliente
    client = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except:
        pass

    # 2. Leer PDF (Método seguro)
    texto_completo = ""
    try:
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                lines = t.split('\n')
                # Limpieza agresiva de encabezados basura
                cleaned = [l for l in lines if "Pág.:" not in l and "Fecha:[-]" not in l and "ORDEN:Hora" not in l]
                texto_completo += "\n".join(cleaned) + "\n"
    except:
        return None, None

    # 3. Extracción con REGEX (Gratis e Infalible)
    # Patrón: Texto... -> 1. Pasos... -> (ID)
    patron = r"(?s)(.*?)\n\s*1\.\s+(.*?)\s*\((\d+)\)"
    coincidencias = re.findall(patron, texto_completo)
    
    datos = []
    total = len(coincidencias)
    progress_bar = st.progress(0)
    status_text = st.empty()

    if total == 0:
        st.error("No encontré el formato estándar en el PDF. Revisa que sea el reporte correcto.")
        return None, None

    for i, (preg_raw, resp_raw, id_nota) in enumerate(coincidencias):
        status_text.text(f"Procesando {i+1}/{total} (ID: {id_nota})...")
        
        # Limpieza
        lines_p = preg_raw.strip().split('\n')
        preg_clean = lines_p[-1].strip() if lines_p else "Pregunta General"
        if len(preg_clean) < 5: preg_clean = preg_raw.strip() # Recuperar si cortamos de más
        
        resp_final = "1. " + resp_raw.strip()
        preg_hibrida = preg_clean

        # 4. Enriquecer con IA (Solo si hay cupo, si no, seguimos igual)
        if client:
            prompt_sinonimos = f"Dame 3 sinónimos técnicos breves para buscar: '{preg_clean}'. Solo las palabras, separadas por coma."
            # Usamos la función blindada con pocos tokens
            sinonimos = consultar_ia_blindada(client, prompt_sinonimos, max_tokens=30)
            if sinonimos:
                preg_hibrida = f"{preg_clean} ({sinonimos})"
        
        datos.append([id_nota, preg_hibrida, resp_final, ""])
        progress_bar.progress((i + 1) / total)
        
        # Pausa táctica para no saturar
        if i % 10 == 0: time.sleep(0.1)

    progress_bar.empty()
    status_text.empty()
    
    # Crear CSV
    df = pd.DataFrame(datos, columns=["ID", "Pregunta_Hibrida", "Respuesta", "Video"])
    df = df.drop_duplicates(subset=["ID"])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    return csv_buffer.getvalue(), df

def actualizar_github(content, repo_name):
    try:
        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(repo_name)
        path = "base_conocimiento_HIBRIDA.csv"
        msg = "Actualizacion automatica (Failover Mode)"
        
        try:
            contents = repo.get_contents(path)
            repo.update_file(path, msg, content, contents.sha)
        except:
            repo.create_file(path, msg, content)
        return True, "Actualizado en GitHub correctamente."
    except Exception as e:
        return False, f"Error GitHub: {e}"

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Panel de Control")
    st.info("Sube el PDF de WinJes para actualizar.")
    uploaded = st.file_uploader("Archivo PDF", type="pdf")
    
    if uploaded and st.button("Actualizar Bot"):
        with st.status("Procesando...", expanded=True) as status:
            csv, df = procesar_pdf_y_generar_csv(uploaded)
            if csv:
                st.write(f"✅ {len(df)} registros extraídos.")
                REPO = "Vld439/BarisBot" 
                ok, msg = actualizar_github(csv, REPO)
                if ok: 
                    st.success(msg)
                    time.sleep(2)
                    st.rerun()
                else: st.error(msg)
                status.update(label="Listo", state="complete")

# --- CHATBOT ---
st.title("🤖 Soporte Baris")

# Cargar Base de Datos
@st.cache_resource
def load_db():
    if not os.path.exists("base_conocimiento_HIBRIDA.csv"): return None, None
    # Limpieza preventiva
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
            
        # Carga por lotes
        batch = 100
        for i in range(0, len(ids), batch):
            coll.add(ids=ids[i:i+batch], documents=docs[i:i+batch], metadatas=metas[i:i+batch])
            
        return coll, df
    except: return None, None

collection, df_global = load_db()

if not collection:
    st.warning("ALERTA: Sube el PDF en el panel lateral para iniciar el cerebro del bot.")
    st.stop()

# Chat Logic
if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Consulta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # 1. Buscar Info (Esto nunca falla)
        contexto = ""
        fuentes = []
        
        # Vectorial
        try:
            res = collection.query(query_texts=[prompt], n_results=3)
            if res['documents']:
                for i, meta in enumerate(res['metadatas'][0]):
                    contexto += f"- {meta['p']}: {meta['r']}\n\n"
                    fuentes.append(meta)
        except: pass
        
        # Palabras Clave (Respaldo)
        palabras = prompt.lower().split()
        for _, row in df_global.iterrows():
            score = 0
            txt = (str(row['Pregunta_Hibrida']) + " " + str(row['Respuesta'])).lower()
            if all(p in txt for p in palabras): # Coincidencia estricta de palabras
                if not any(f['p'] == row['Pregunta_Hibrida'] for f in fuentes):
                    contexto += f"- {row['Pregunta_Hibrida']}: {row['Respuesta']}\n"
                    fuentes.append({"p": row['Pregunta_Hibrida'], "r": row['Respuesta'], "v": row['Video']})

        # 2. Generar Respuesta
        if not fuentes:
            resp_final = "No encontré información exacta en el manual sobre eso."
        else:
            client = None
            try: client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            except: pass
            
            prompt_ia = f"""
            Eres experto en Baris. Responde al usuario usando SOLO esta información:
            {contexto}
            Pregunta: {prompt}
            Responde amable y directo.
            """
            
            # Usamos la "Caja de Cambios"
            resp_ia = None
            if client:
                resp_ia = consultar_ia_blindada(client, prompt_ia)
            
            if resp_ia:
                resp_final = resp_ia
            else:
                # FALLBACK TOTAL: Si no hay IA, mostramos los datos crudos
                resp_final = "ALERTA **Modo sin conexión a IA** (Mostrando datos directos):\n\n"
                for f in fuentes:
                    resp_final += f"**{f['p']}**\n{f['r']}\n\n---\n"

        st.markdown(resp_final)
        st.session_state.messages.append({"role": "assistant", "content": resp_final})