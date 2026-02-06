import re
import pandas as pd
import PyPDF2

pdf_filename = "Preguntas Frecuentes - Base de conocimiento CHATFUEL.pdf"

def limpiar_y_reparar():
    print("⏳ Leyendo PDF...")
    reader = PyPDF2.PdfReader(pdf_filename)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Dividimos por los códigos numéricos (ej: 3434)
    sections = re.split(r'\((\d+)\)', text)
    
    data = []

    intro_text = sections[0]
    # Buscamos si hay una pregunta oculta en la intro
    # Limpiamos cabeceras basura primero
    intro_clean = re.sub(r'Fecha:\[-].*?ORDEN:Numero', '', intro_text, flags=re.DOTALL)
    intro_clean = re.sub(r'--- PAGE \d+ ---', '', intro_clean)
    
    # Si detectamos pasos numerados (1. 2. 3.), asumimos que es una respuesta válida
    if "1." in intro_clean and "2." in intro_clean:
        lines = [l.strip() for l in intro_clean.split('\n') if l.strip()]
        # Asumimos que la primera línea relevante es el Título
        # Buscamos la línea que dice "Procedimiento..." o similar
        pregunta_intro = "Procedimiento General" # Default
        respuesta_intro = intro_clean
        
        for line in lines:
            if "Procedimiento" in line or "?" in line:
                pregunta_intro = line
                break
        
        data.append({
            "ID": "0000", # ID falso para la primera
            "Pregunta": pregunta_intro, 
            "Respuesta": intro_clean, 
            "Video": ""
        })

    # --- PROCESAR EL RESTO DEL ARCHIVO ---
    for i in range(1, len(sections), 2):
        code = sections[i].strip()
        content = sections[i+1].strip()
        
        # Extracción estándar
        lines = content.split('\n')
        question = lines[0].strip()
        
        # Extracción de Video
        video_match = re.search(r'(https?://youtu\.be/\S+)', content)
        video = video_match.group(0) if video_match else ""
        
        # Limpieza de basura visual
        body_lines = [line for line in lines[1:] if "Fecha:[-]" not in line and "ORDEN:Numero" not in line]
        body = "\n".join(body_lines).replace(video, "").strip()
        
        data.append({"ID": code, "Pregunta": question, "Respuesta": body, "Video": video})

    # Guardar
    df = pd.DataFrame(data)
    df.to_csv("base_conocimiento_baris_limpia.csv", index=False, encoding='utf-8')
    print(f" Base de datos reparada con {len(df)} preguntas.")

if __name__ == "__main__":
    limpiar_y_reparar()