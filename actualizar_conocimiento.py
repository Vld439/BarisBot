import pdfplumber
import pandas as pd
from groq import Groq
import os
import re
import time

# --- CONFIGURACION ---
# Nombre exacto del archivo que exporta WinJes
PDF_PATH = "rp_mref.frx.pdf" 
CSV_OUTPUT = "base_conocimiento_HIBRIDA.csv"

# Intenta obtener la API KEY de las variables de entorno o secrets.toml
API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    try:
        import toml
        secrets = toml.load(".streamlit/secrets.toml")
        API_KEY = secrets["GROQ_API_KEY"]
    except:
        print("[ERROR] No encuentro la GROQ_API_KEY en variables de entorno ni en secrets.toml")
        exit()

client = Groq(api_key=API_KEY)

def limpiar_texto_winjes(texto):
    """Limpia encabezados y pies de pagina tipicos del reporte de WinJes"""
    # Eliminar lineas de encabezado repetitivas y numeros de pagina
    texto = re.sub(r"Pág\.:\s*\d+", "", texto)
    texto = re.sub(r"Preguntas Frecuentes", "", texto)
    # Elimina la cabecera de la tabla que se repite
    texto = re.sub(r"Fecha:\[-\].*?ORDEN:Hora", "", texto, flags=re.DOTALL)
    return texto

def extraer_texto_pdf(path):
    print(f"[INFO] Leyendo reporte WinJes: {path}...")
    texto_completo = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                texto_pagina = page.extract_text()
                if texto_pagina:
                    texto_completo += limpiar_texto_winjes(texto_pagina) + "\n"
    except Exception as e:
        print(f"[ERROR] Leyendo PDF: {e}")
        return None
    return texto_completo

def procesar_con_ia(texto_chunk):
    # Prompt: Genera sinonimos automaticamente
    prompt = f"""
    Eres un experto en soporte tecnico de sistemas en Paraguay/Latinoamerica.
    Tu tarea es leer el texto de un manual tecnico y extraer pares de PREGUNTA y RESPUESTA.
    
    IMPORTANTE - ENRIQUECIMIENTO DE TITULOS (HIBRIDACION):
    La "Pregunta" no debe ser solo el titulo del manual. DEBES agregarle sinonimos comunes y frases coloquiales entre parentesis para facilitar la busqueda del usuario final.
    
    Ejemplos de conversion:
    - Manual: "Anular Venta" -> Tu salida: "Como anular una venta (borrar, cancelar, eliminar, echar para atras)"
    - Manual: "Ingresar Articulo" -> Tu salida: "Registrar entrada de articulo (cargar stock, meter mercaderia, compras)"
    - Manual: "ABM Clientes" -> Tu salida: "Alta Baja y Modificacion de Clientes (crear cliente, editar datos, borrar cliente)"

    REGLAS DE FORMATO DE SALIDA:
    1. Salida SOLO en formato compatible CSV: ID|Pregunta Hibrida|Respuesta
    2. Usa el separador pipe (|) estrictamente.
    3. Una pareja por linea.
    4. El ID es el numero entre parentesis al final del bloque, ej: (3528).
    5. En la RESPUESTA: Incluye los pasos numerados completos y cualquier Link de Youtube si aparece.
    6. NO pongas introduccion ni explicaciones extra, solo los datos.

    TEXTO DEL MANUAL A PROCESAR:
    {texto_chunk}
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3, 
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] Procesando bloque con IA: {e}")
        return ""

def main():
    if not os.path.exists(PDF_PATH):
        print(f"[ERROR] No encuentro {PDF_PATH}. Exporta el reporte de WinJes y ponlo aqui.")
        return

    raw_text = extraer_texto_pdf(PDF_PATH)
    if not raw_text: return

    # Dividimos en trozos para no saturar a la IA (3000 caracteres aprox)
    chunk_size = 3000 
    chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
    
    datos_nuevos = []
    print(f"[INFO] Procesando {len(chunks)} bloques con Inteligencia Artificial...")

    for i, chunk in enumerate(chunks):
        print(f"   - Analizando bloque {i+1}/{len(chunks)}...")
        resultado = procesar_con_ia(chunk)
        
        # Procesar respuesta de la IA linea por linea
        if resultado:
            lineas = resultado.strip().split('\n')
            for linea in lineas:
                if "|" in linea:
                    partes = linea.split("|")
                    # Buscamos que tenga al menos 3 partes (ID, Pregunta, Respuesta)
                    if len(partes) >= 3:
                        try:
                            # Limpiar ID de parentesis
                            id_nota = partes[0].strip().replace("(", "").replace(")", "")
                            preg = partes[1].strip()
                            resp = partes[2].strip()
                            
                            # Deteccion basica de video si existe un link http en la respuesta
                            video = ""
                            # Opcional: Logica para extraer link si se desea en columna aparte
                            
                            datos_nuevos.append([id_nota, preg, resp, video])
                        except:
                            continue
        
        time.sleep(1) # Pequeña pausa para respetar limites de la API

    # Guardar a CSV
    if datos_nuevos:
        columnas = ["ID", "Pregunta_Hibrida", "Respuesta", "Video"]
        df = pd.DataFrame(datos_nuevos, columns=columnas)
        
        # Eliminar duplicados por ID (si el PDF tiene repetidos o solapamiento de bloques)
        df = df.drop_duplicates(subset=["ID"])
        
        df.to_csv(CSV_OUTPUT, index=False)
        print(f"[EXITO] Se genero {CSV_OUTPUT} con {len(df)} conocimientos procesados.")
        print("[INFO] Ahora ejecuta: git add . && git commit -m 'Actualizar manual' && git push")
    else:
        print("[ALERTA] No se pudieron extraer datos validos. Revisa el formato del PDF.")

if __name__ == "__main__":
    main()