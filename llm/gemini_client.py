# llm/gemini_client.py
"""
Cliente Gemini para BananaVision.

Recibe un PipelineResult ya procesado y genera una recomendación
en lenguaje natural para el agricultor.

NO hace diagnóstico — eso lo hacen los modelos .keras.
Solo traduce el diagnóstico técnico a lenguaje útil y accionable.
"""

import os
from google import genai
from google.genai import types
from pipeline.schemas import PipelineResult
from llm.base_conocimiento import cargar_documento_rag, BASE_COMPLETA

# ⚙️ PARÁMETRO — Configurar via variable de entorno GEMINI_API_KEY
# Obtener key gratis en: https://aistudio.google.com/apikey
_client = None

def _setup():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY no configurada. "
                "Exporta: export GEMINI_API_KEY='tu_key_aqui'"
            )
        _client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1"},
        )
    return _client


# ⚙️ PARÁMETRO — Cambiar modelo si se necesita más capacidad:
# "gemini-2.0-flash"  → rápido y gratis (recomendado para hackathon)
# "gemini-2.0-pro"    → más preciso, límite de requests menor
MODELO_GEMINI = "gemini-1.5-flash"


def _construir_prompt(ctx: dict) -> str:
    """
    Construye el prompt para Gemini basado en el contexto del pipeline.

    ctx viene de PipelineResult.to_llm_context():
    {
        "flujo": "ENFERMO" | "DEFICIT",
        "diagnostico": "Fusarium" | "Potassium" | etc.,
        "confianza_pct": 87.3,
        "baja_confianza": False,
        "categoria": "enfermedad" | "deficiencia"
    }
    """

    advertencia_confianza = ""
    if ctx.get("baja_confianza"):
        advertencia_confianza = (
            f"\nNOTA IMPORTANTE: El modelo de IA tiene BAJA CONFIANZA "
            f"({ctx['confianza_pct']}%) en este diagnóstico. "
            "Debes mencionar esto al agricultor y recomendar que tome otra foto "
            "o consulte con un técnico presencialmente."
        )

    conocimiento = cargar_documento_rag(ctx.get("diagnostico")) or BASE_COMPLETA

    return f"""Eres un agrónomo experto en banano ecuatoriano que habla directamente con agricultores.
Tu tarea: explicar el diagnóstico de IA de forma clara y dar acciones concretas.

BASE DE CONOCIMIENTO TÉCNICA:
{conocimiento}

DIAGNÓSTICO DEL SISTEMA DE IA:
- Categoría: {ctx['categoria']}
- Diagnóstico: {ctx['diagnostico']}
- Confianza del modelo: {ctx['confianza_pct']}%
{advertencia_confianza}

INSTRUCCIONES PARA TU RESPUESTA:
1. Usa español claro y simple — el agricultor puede tener poca escolaridad
2. Máximo 5 oraciones en total
3. La PRIMERA oración debe ser la acción más urgente e importante
4. Menciona el nombre del problema en lenguaje simple (no solo el nombre científico)
5. Si es Fusarium: SIEMPRE mencionar que debe contactar a Agrocalidad
6. Si es deficiencia nutricional: menciona el nutriente y qué producto aplicar
7. NO uses listas con bullets — escribe en párrafo continuo
8. NO repitas el porcentaje de confianza en tu respuesta
9. Termina con una frase de aliento breve

Responde SOLO con el mensaje para el agricultor, sin encabezados ni explicaciones adicionales."""


def generar_recomendacion(resultado: PipelineResult) -> str:
    """
    Función principal: recibe PipelineResult → devuelve string con recomendación.

    Solo se llama cuando resultado.llm_needed == True.
    Para flujos SANO y NO_BANANA usar respuestas_estaticas.py
    """
    client = _setup()

    ctx = resultado.to_llm_context()
    prompt = _construir_prompt(ctx)

    # ⚙️ PARÁMETRO — Ajustar temperature si las respuestas salen muy variables
    # 0.3 = más consistente, 0.7 = más creativo
    response = client.models.generate_content(
        model=MODELO_GEMINI,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=300,
        )
    )

    return response.text.strip()


def generar_recomendacion_con_imagen(resultado: PipelineResult, imagen_bytes: bytes) -> str:
    """
    Versión multimodal: pasa también la imagen a Gemini.
    Gemini puede ver la foto además del diagnóstico del modelo.

    ⚙️ PARÁMETRO — Activar esta versión si se quiere que Gemini
    confirme visualmente el diagnóstico del modelo .keras
    """
    client = _setup()

    ctx = resultado.to_llm_context()
    prompt = _construir_prompt(ctx)

    response = client.models.generate_content(
        model=MODELO_GEMINI,
        contents=[
            prompt,
            types.Part.from_bytes(data=imagen_bytes, mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=300,
        )
    )

    return response.text.strip()
