# llm/respuestas_estaticas.py
"""
Mensajes estáticos para flujos que no necesitan Gemini.
Flujos SANO y NO_BANANA no requieren LLM — son respuestas fijas.
"""

# ⚙️ PARÁMETRO — Ajustar el tono y contenido según feedback de usuarios
RESPUESTAS = {
    "SANO": (
        "¡Buenas noticias! Tu planta de banano luce saludable. "
        "No se detectaron enfermedades ni deficiencias nutricionales. "
        "Continúa con tu programa de mantenimiento preventivo y "
        "realiza la próxima revisión en 7 días."
    ),
    "NO_BANANA": (
        "La imagen no parece ser una hoja de banano. "
        "Por favor toma la foto más cerca de la hoja, con buena iluminación "
        "y asegúrate de que la hoja ocupe la mayor parte de la imagen. "
        "Intenta de nuevo."
    ),
}

def obtener_respuesta_estatica(flujo: str) -> str:
    return RESPUESTAS.get(flujo, "No se pudo procesar la imagen. Intenta de nuevo.")
