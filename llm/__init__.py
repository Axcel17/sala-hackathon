# llm/__init__.py
from llm.respuestas_fijas import obtener_recomendacion_fija, obtener_detalle_fijo
from llm.respuestas_estaticas import obtener_respuesta_estatica

# gemini_client y base_conocimiento quedan disponibles en el repo
# pero no se usan en el flujo principal — requieren OPENAI_API_KEY

__all__ = [
    "obtener_recomendacion_fija",
    "obtener_detalle_fijo",
    "obtener_respuesta_estatica",
]
