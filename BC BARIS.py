import re
import pandas as pd
import PyPDF2

# Colocar PDF en la misma carpeta:
pdf_filename = "Preguntas Frecuentes - Base de conocimiento CHATFUEL.pdf"

def clean_baris_pdf():
    reader = PyPDF2.PdfReader(pdf_filename)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    # 1. Separar por los códigos de pregunta ej: (3466)
    # Esto divide el PDF en bloques lógicos
    sections = re.split(r'\((\d+)\)', text)
    
    data = []
    for i in range(1, len(sections), 2):
        code = sections[i].strip()     # El código (ej: 3466)
        content = sections[i+1].strip() # El texto sucio
        
        # 2. Extraer Título (Primera línea)
        lines = content.split('\n')
        question = lines[0].strip()
        
        # 3. Extraer Video (SOLO si está explícito en el bloque)
        # Busca enlaces de YouTube
        video_match = re.search(r'(https?://youtu\.be/\S+)', content)
        video = video_match.group(0) if video_match else ""
        
        # 4. Limpiar el cuerpo de la respuesta
        # Quitamos basura visual que tenga "Fecha:", "Sis:", etc.
        body_lines = [line for line in lines[1:] if "Fecha:[-]" not in line and "ORDEN:Numero" not in line]
        body = "\n".join(body_lines).replace(video, "").strip() # Quitamos el link del texto para ponerlo en su columna
        
        data.append({"ID": code, "Pregunta": question, "Respuesta": body, "Video": video})

    # Guardar en CSV
    df = pd.DataFrame(data)
    df.to_csv("base_conocimiento_baris_limpia.csv", index=False)
    print("¡Archivo 'base_conocimiento_baris_limpia.csv' creado con éxito!")

if __name__ == "__main__":
    try:
        clean_baris_pdf()
    except Exception as e:
        print(f"Error: {e}. Asegúrate de tener instalado: pip install pandas pypdf2")