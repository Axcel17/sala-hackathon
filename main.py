# main.py
"""
Servidor FastAPI que expone el pipeline de diagnóstico de hojas de banano.

Endpoints:
    GET  /health   → estado del servidor y modelos
    POST /predict  → recibe imagen como archivo (multipart), devuelve diagnóstico + recomendación RAG
"""

import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import run_pipeline
from llm import generar_recomendacion, obtener_respuesta_estatica

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bananavision")

app = FastAPI(title="BananaVision API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    flujo: str
    llm_needed: bool
    diagnostico: str | None = None
    confianza: float | None = None
    confianza_pct: float | None = None
    baja_confianza: bool = False
    categoria: str | None = None
    recomendacion: str = ""


# ─────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Verifica que el servidor y los modelos están listos."""
    return {"status": "ok", "modelos": ["m1", "m3", "m4"]}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen como archivo y devuelve el diagnóstico del pipeline
    más una recomendación generada por el RAG (OpenAI GPT-4o mini + documentos agrícolas).
    """
    image_bytes = await file.read()
    log.info(f"📥 Imagen recibida — {file.filename} ({len(image_bytes) / 1024:.1f} KB)")

    try:
        resultado = run_pipeline(image_bytes)
    except Exception as e:
        log.error(f"❌ Error en pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Error en el pipeline: {str(e)}")

    confianza_pct = round(resultado.confianza * 100, 1) if resultado.confianza else None
    baja = " ⚠️  baja confianza" if resultado.baja_confianza else ""
    diag = f"→ {resultado.diagnostico} ({confianza_pct}%){baja}" if resultado.diagnostico else ""
    log.info(f"✅ Resultado: flujo={resultado.flujo}  {diag}")

    # RAG: genera recomendación si hay diagnóstico, estática si no
    if resultado.llm_needed:
        log.info(f"🤖 Consultando RAG para '{resultado.diagnostico}'...")
        try:
            recomendacion = generar_recomendacion(resultado)
            log.info("💬 RAG respondió correctamente")
        except Exception as e:
            log.warning(f"⚠️  RAG falló, usando respuesta estática: {e}")
            recomendacion = obtener_respuesta_estatica(resultado.flujo)
    else:
        recomendacion = obtener_respuesta_estatica(resultado.flujo)

    return PredictResponse(
        flujo          = resultado.flujo,
        llm_needed     = resultado.llm_needed,
        diagnostico    = resultado.diagnostico,
        confianza      = resultado.confianza,
        confianza_pct  = confianza_pct,
        baja_confianza = resultado.baja_confianza,
        categoria      = resultado.categoria,
        recomendacion  = recomendacion,
    )
