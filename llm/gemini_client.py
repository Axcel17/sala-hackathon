# llm/gemini_client.py
"""
Cliente OpenAI para BananaVision.

Recibe un PipelineResult ya procesado y genera una recomendación
en lenguaje natural para el agricultor usando GPT-4o mini.

NO hace diagnóstico — eso lo hacen los modelos .keras.
Solo traduce el diagnóstico técnico a lenguaje útil y accionable.
"""

import os
import base64
from openai import OpenAI
from pipeline.schemas import PipelineResult
from llm.base_conocimiento import cargar_documento_rag, BASE_COMPLETA

# ⚙️ PARÁMETRO — Configurar via variable de entorno OPENAI_API_KEY
_client = None

def _setup():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY no configurada. "
                "Exporta: export OPENAI_API_KEY='tu_key_aqui'"
            )
        _client = OpenAI(api_key=api_key)
    return _client


# ⚙️ PARÁMETRO — Cambiar modelo si se necesita más capacidad:
# "gpt-4o-mini"  → rápido y económico (recomendado)
# "gpt-4o"       → más preciso, más costo
MODELO = "gpt-4o-mini"


def _construir_prompt(ctx: dict) -> str:
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
    """
    client = _setup()
    ctx = resultado.to_llm_context()
    prompt = _construir_prompt(ctx)

    response = client.chat.completions.create(
        model=MODELO,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


def generar_recomendacion_con_imagen(resultado: PipelineResult, imagen_bytes: bytes) -> str:
    """
    Versión multimodal: pasa también la imagen a GPT-4o mini.
    """
    client = _setup()
    ctx = resultado.to_llm_context()
    prompt = _construir_prompt(ctx)

    imagen_b64 = base64.b64encode(imagen_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=MODELO,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imagen_b64}"}},
            ],
        }],
        temperature=0.3,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()
