# llm/base_conocimiento.py
"""
Base de conocimiento agrícola para BananaVision.

Carga los documentos .md de la carpeta DOCUMENTOS RAG/ e inyecta
solo el documento relevante al diagnóstico en el prompt de Gemini.
"""

import os

# Ruta a la carpeta de documentos RAG
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DOCUMENTOS RAG")

# Mapeo: nombre de clase del pipeline → archivo .md
NOMBRE_A_ARCHIVO = {
    # Enfermedades (CLASES_M4 de pipeline/config.py)
    "Cordana":       "cordana.md",
    "Fusarium":      "fusarium.md",
    "SigatokaNegra": "black-sigatoka.md",
    # Nutrientes (CLASES_M3 de pipeline/config.py)
    "Boron":      "boron.md",
    "Calcium":    "calcium.md",
    "Iron":       "iron.md",
    "Magnesium":  "magnesium.md",
    "Manganese":  "manganese.md",
    "Potassium":  "potassium.md",
    "Sulphur":    "sulphur.md",
    "Zinc":       "zinc.md",
}


def cargar_documento_rag(diagnostico: str) -> str:
    """
    Devuelve el contenido del documento RAG correspondiente al diagnóstico.
    Retorna string vacío si no hay documento para ese diagnóstico.
    """
    if not diagnostico:
        return ""
    nombre_archivo = NOMBRE_A_ARCHIVO.get(diagnostico)
    if not nombre_archivo:
        return ""
    ruta = os.path.join(DOCS_DIR, nombre_archivo)
    if not os.path.exists(ruta):
        return ""
    with open(ruta, "r", encoding="utf-8") as f:
        return f.read()


def _cargar_todos() -> str:
    """Concatena todos los documentos RAG. Usado como fallback."""
    partes = []
    for nombre_archivo in NOMBRE_A_ARCHIVO.values():
        ruta = os.path.join(DOCS_DIR, nombre_archivo)
        if os.path.exists(ruta):
            with open(ruta, "r", encoding="utf-8") as f:
                partes.append(f.read())
    return "\n\n---\n\n".join(partes)


# Fallback: todos los documentos concatenados (se carga una vez al importar)
BASE_COMPLETA = _cargar_todos()
