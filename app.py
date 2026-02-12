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
st.set_page_config(page_title="Soporte Baris", layout="wide")

# --- FUNCION DE RESPALDO ---
def consultar_ia_blindada(cliente_groq, prompt, max_tokens=500):
    """
    Intenta con varios modelos para evitar bloqueos.
    """
    modelos = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
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
    """
    Descarga el CSV actual de GitHub para saber que IDs ya tenemos.
    Si no existe, devuelve un DataFrame vacio.
    """
    try:
        if "GITHUB_TOKEN" not in st.secrets:
            return pd.DataFrame(columns=["ID", "Pregunta_Hibrida", "Respuesta", "Video"])
            
        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(repo_name)
        
        try:
            file_content = repo.get_contents("base_conocimiento_HIBRIDA.csv")
            csv_data = file_content.decoded_content.decode("utf-8")
            return pd.read_csv(io.StringIO(csv_data))
        except:
            # El archivo no existe aun en el repo
            return pd.DataFrame(columns=["ID", "Pregunta_Hibrida", "Respuesta", "Video"])
    except Exception:
        return pd.DataFrame(columns=["ID", "Pregunta_Hibrida", "Respuesta", "Video"])

def procesar_pdf_incremental(file_obj, df_actual):
    # 1. Configurar Cliente Groq (si existe)
    client = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except:
        pass

    # 2. Leer PDF
    texto_completo = ""
    try:
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                lines = t.split('\n')
                # Limpieza de encabezados WinJes
                cleaned = [l for l in lines if "Pág.:" not in l and "Fecha:[-]" not in l and "ORDEN:Hora" not in l]
                texto_completo += "\n".join(cleaned) + "\n"
    except:
        return None, None, 0

    # 3. Extraccion con REGEX
    patron = r"(?s)(.*?)\n\s*1\.\s+(.*?)\s*\((\d+)\)"
    coincidencias = re.findall(patron, texto_completo)
    
    # Lista de IDs que ya tenemos
    ids_existentes = set(df_actual["ID"].astype(str).tolist())
    
    # Convertimos el PDF actual a lista para ir agregando lo nuevo
    # Aseguramos que las columnas sean las correctas
    if df_actual.empty:
        datos_finales = []
    else:
        # Reordenar columnas por si acaso
        df_actual = df_actual[["ID", "Pregunta_Hibrida", "Respuesta", "Video"]]
        datos_finales = df_actual.values.tolist()

    nuevos_cont = 0
    total = len(coincidencias)
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (preg_raw, resp_raw, id_nota) in enumerate(coincidencias):
        id_nota = str(id_nota)
        
        # --- LOGICA INCREMENTAL ---
        if id_nota in ids_existentes:
            status_text.text(f"Registro {i+1}/{total} (ID: {id_nota}) -> YA EXISTE. Saltando...")
            # No hacemos nada, mantenemos el dato viejo que ya esta en datos_finales
        else:
            status_text.text(f"Registro {i+1}/{total} (ID: {id_nota}) -> NUEVO. Procesando...")
            
            # Limpieza de la pregunta nueva
            lines_p = preg_raw.strip().split('\n')
            preg_clean = lines_p[-1].strip() if lines_p else "Pregunta General"
            if len(preg_clean) < 5: preg_clean = preg_raw.strip() 
            
            resp_final = "1. " + resp_raw.strip()
            preg_hibrida = preg_clean

            # Solo usamos IA para este registro nuevo
            if client:
                prompt_sinonimos = f"Dame 3 sinonimos tecnicos para: '{preg_clean}'. Solo palabras separadas por coma."
            
                sinonimos = consultar_ia_blindada(client, prompt_sinonimos, max_tokens=30)
                if sinonimos:
                    preg_hibrida = f"{preg_clean} ({sinonimos})"
            
            datos_finales.append([id_nota, preg_hibrida, resp_final, ""])
            nuevos_cont += 1
            
            # Pequeña pausa solo si procesamos IA
            if client: time.sleep(0.5)
        
        progress_bar.progress((i + 1) / total)

    progress_bar.empty()
    status_text.empty()
    
    # Crear DataFrame final combinado
    df_nuevo = pd.DataFrame(datos_finales, columns=["ID", "Pregunta_Hibrida", "Respuesta", "Video"])
    # Asegurar que no hay duplicados (priorizando el ultimo por si acaso)
    df_nuevo = df_nuevo.drop_duplicates(subset=["ID"], keep="last")
    
    csv_buffer = io.StringIO()
    df_nuevo.to_csv(csv_buffer, index=False)
    
    return csv_buffer.getvalue(), df_nuevo, nuevos_cont

def actualizar_github(content, repo_name):
    try:
        if "GITHUB_TOKEN" not in st.secrets:
            return False, "Falta GITHUB_TOKEN en secrets."

        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(repo_name)
        path = "base_conocimiento_HIBRIDA.csv"
        msg = "Actualizacion incremental automatica"
        
        try:
            contents = repo.get_contents(path)
            repo.update_file(path, msg, content, contents.sha)
        except:
            repo.create_file(path, msg, content)
            
        return True, "Base de datos actualizada en GitHub correctamente."
    except Exception as e:
        return False, f"Error conectando con GitHub: {e}"

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Panel de Control")
    st.info("Sube el PDF de WinJes. Solo se procesaran las preguntas nuevas.")
    
    uploaded = st.file_uploader("Archivo PDF", type="pdf")
    
    if uploaded and st.button("Actualizar Base de Datos"):
        REPO = "Vld439/BarisBot"
        
        with st.status("Iniciando modo incremental...", expanded=True) as status:
            
            status.write("1. Descargando base de datos actual de GitHub...")
            df_actual = obtener_csv_actual_github(REPO)
            st.write(f"   - Registros actuales en la nube: {len(df_actual)}")
            
            status.write("2. Analizando PDF en busca de novedades...")
            csv_str, df_final, cont_nuevos = procesar_pdf_incremental(uploaded, df_actual)
            
            if csv_str:
                if cont_nuevos > 0:
                    status.write(f"3. Se encontraron {cont_nuevos} preguntas nuevas. Subiendo...")
                    ok, msg = actualizar_github(csv_str, REPO)
                    if ok:
                        status.update(label="Actualizacion Exitosa", state="complete")
                        st.success(f"Listo. Se agregaron {cont_nuevos} conocimientos nuevos. Total: {len(df_final)}")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    status.update(label="Sin cambios", state="complete")
                    st.info("El PDF no contiene ninguna pregunta nueva. La base de datos esta al dia.")
            else:
                st.error("Error leyendo el PDF.")

# --- CHATBOT ---
st.title("Soporte Baris")

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
    st.warning("No hay base de datos local. Si acabas de actualizar, espera un momento o recarga.")
    st.stop()

# Logica del Chat
if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Consulta aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # 1. Busqueda
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
        
        # Palabras Clave
        palabras = prompt.lower().split()
        for _, row in df_global.iterrows():
            txt = (str(row['Pregunta_Hibrida']) + " " + str(row['Respuesta'])).lower()
            if all(p in txt for p in palabras): 
                if not any(f['p'] == row['Pregunta_Hibrida'] for f in fuentes):
                    contexto += f"- {row['Pregunta_Hibrida']}: {row['Respuesta']}\n"
                    fuentes.append({"p": row['Pregunta_Hibrida'], "r": row['Respuesta'], "v": row['Video']})

        # 2. Generar Respuesta
        if not fuentes:
            resp_final = "No encontre informacion sobre eso en el manual."
        else:
            client = None
            try: 
                if "GROQ_API_KEY" in st.secrets:
                    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            except: pass
            
            prompt_ia = f"""
            Eres experto en Baris. Responde usando SOLO esta informacion:
            {contexto}
            Pregunta: {prompt}
            """
            
            resp_ia = None
            if client:
                resp_ia = consultar_ia_blindada(client, prompt_ia)
            
            if resp_ia:
                resp_final = resp_ia
            else:
                resp_final = "**Resultados del Manual:**\n\n"
                for f in fuentes:
                    resp_final += f"**{f['p']}**\n{f['r']}\n\n---\n"

        st.markdown(resp_final)
        st.session_state.messages.append({"role": "assistant", "content": resp_final})