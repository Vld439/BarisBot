import pandas as pd
import os

# --- CONFIGURACION ---
input_csv = "base_conocimiento_baris_limpia.csv"
output_csv = "base_conocimiento_baris_POTENCIADA.csv"

# --- DICCIONARIO DE SINONIMOS ---
# Formato: "palabra_clave": ["sinonimo1", "sinonimo2", "frase coloquial"]
sinonimos = {
    "cancelar": ["anular", "revertir", "echar para atras", "deshacer", "borrar", "quitar", "me equivoque", "baja"],
    "impresion": ["ticket", "papel", "factura", "comprobante", "impresora", "cola", "trabada", "atascada", "no sale", "imprime mal"],
    "lento": ["pesado", "se arrastra", "cuelga", "tarda", "lag", "lenteja", "no responde", "lentasos"],
    "venta": ["cobro", "facturacion", "despacho", "caja", "vendido"],
    "anular": ["echar para atras", "cancelar", "revertir", "dar de baja", "eliminar", "restaurar", "devolver"],
    "cliente": ["comprador", "persona", "señor", "tipo"],
    "stock": ["mercaderia", "inventario", "existencia", "articulos", "productos"],
    "caja": ["dinero", "plata", "sencillo", "cobranza", "turno", "cierre"],
    "error": ["fallo", "mensaje", "cartel", "aviso", "no funciona", "roto", "bug"]
}

print("Iniciando proceso de enriquecimiento de datos...")

if not os.path.exists(input_csv):
    print(f"Error: No se encuentra el archivo {input_csv}")
    exit()

try:
    df = pd.read_csv(input_csv)
    # Rellenar valores nulos para evitar errores
    df = df.fillna("")
    print(f"Archivo cargado. Procesando {len(df)} registros...")
except Exception as e:
    print(f"Error leyendo CSV: {e}")
    exit()

def enriquecer_pregunta(texto):
    texto_str = str(texto)
    texto_lower = texto_str.lower()
    palabras_extra = []
    
    for clave, lista in sinonimos.items():
        if clave in texto_lower:
            palabras_extra.extend(lista)
            
    if palabras_extra:
        # Eliminar duplicados y palabras que ya esten en el texto original
        nuevas = [p for p in set(palabras_extra) if p not in texto_lower]
        if nuevas:
            # Agregamos las palabras ocultas entre corchetes al final
            return f"{texto_str} [{' '.join(nuevas)}]"
    
    return texto_str

# Crear copia y aplicar cambios
df['Pregunta'] = df['Pregunta'].apply(enriquecer_pregunta)

# Guardar nuevo archivo
df.to_csv(output_csv, index=False)

print("Proceso finalizado correctamente.")
print(f"Nuevo archivo generado: {output_csv}")