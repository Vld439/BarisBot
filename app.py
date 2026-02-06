import streamlit as st
import google.generativeai as genai
import os

st.set_page_config(page_title="Diagnóstico BarisBot", page_icon="🩺")

st.title("🩺 Diagnóstico de Conexión")

# 1. VERIFICACIÓN DE CLAVE (Sin mostrarla completa por seguridad)
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    # Mostramos los primeros y últimos 4 caracteres para que VERIFIQUES si es la nueva
    clave_visible = f"{api_key[:4]}...{api_key[-4:]}"
    st.info(f"🔑 Clave detectada: {clave_visible}")
except:
    st.error(" No se detectó ninguna API Key en los Secrets.")
    st.stop()

# 2. CONFIGURACIÓN
genai.configure(api_key=api_key)

# 3. PRUEBA DE FUEGO
# Usamos el modelo más estándar y estable del mundo.
# Si este falla, el problema es 100% la cuenta/clave.
nombre_modelo = "gemini-2.0-flash-lite-001" 

st.write(f" Intentando conectar con: `{nombre_modelo}`...")

try:
    model = genai.GenerativeModel(nombre_modelo)
    response = model.generate_content("Responde solo con la palabra: ¡CONECTADO!")
    
    st.success(f" ÉXITO: {response.text}")
    st.balloons()
    
except Exception as e:
    # AQUI ES DONDE VEREMOS LA VERDAD
    st.error(" ERROR FATAL DE GOOGLE:")
    st.code(str(e)) # Muestra el error técnico crudo
    
    st.markdown("""
    **Guía de Errores Comunes:**
    * **403 / API key not valid:** La clave en secrets está mal escrita o es la vieja.
    * **429 / Quota exceeded:** Estás usando la cuenta vieja o la nueva no tiene facturación habilitada (aunque sea gratis, a veces pide verificar tarjeta).
    * **404 / Not Found:** El nombre del modelo está mal (raro con 1.5-flash).
    """)