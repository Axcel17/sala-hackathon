# llm/__init__.py
from llm.gemini_client import generar_recomendacion, generar_recomendacion_con_imagen
from llm.respuestas_estaticas import obtener_respuesta_estatica

__all__ = [
    "generar_recomendacion",
    "generar_recomendacion_con_imagen",
    "obtener_respuesta_estatica",
]
