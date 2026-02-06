import google.generativeai as genai

# --- PEGA TU API KEY AQUÍ ---
api_key = "AIzaSyBPjo43jJOj-NM-si-MppQHqqYTxhbYRf8"  
genai.configure(api_key=api_key)

print("🔍 Buscando modelos disponibles para tu cuenta...")
print("-" * 40)

try:
    # Listamos todos los modelos que sirven para generar texto (chat)
    found = False
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Disponible: {m.name}")
            found = True
            
    if not found:
        print("❌ No se encontraron modelos. Verifica tu API Key.")
        
except Exception as e:
    print(f"❌ Error al conectar: {e}")

print("-" * 40)
print("Usa uno de los nombres EXACTOS de la lista en tu archivo app.py")