# llm/respuestas_estaticas.py
"""
Mensajes estáticos para flujos que no necesitan Gemini.
Flujos SANO y NO_BANANA no requieren LLM — son respuestas fijas.

También ofrece respuestas fijas basadas en los documentos de la carpeta
"DOCUMENTOS RAG" (sin llamar a Gemini / OpenAI).
"""

import re
from typing import Dict, Optional

from llm.base_conocimiento import cargar_documento_rag


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


def _parse_frontmatter(md: str) -> (Dict[str, object], str):
    """Extrae el frontmatter YAML simple y el cuerpo restante."""
    if not md.strip().startswith("---"):
        return {}, md

    parts = md.split("---", 2)
    if len(parts) < 3:
        return {}, md

    meta_text = parts[1].strip()
    body = parts[2].strip()

    meta: Dict[str, object] = {}
    current_key: Optional[str] = None

    for raw_line in meta_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line and not line.startswith("-"):
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            if val.startswith("[") and val.endswith("]"):
                items = [item.strip() for item in val[1:-1].split(",") if item.strip()]
                meta[key] = items
                current_key = None
            elif val == "":
                meta[key] = []
                current_key = key
            else:
                if val.lower() in ("true", "false"):
                    meta[key] = val.lower() == "true"
                else:
                    meta[key] = val
                current_key = None
        elif line.startswith("-") and current_key:
            item = line.lstrip("-").strip()
            if isinstance(meta.get(current_key), list):
                meta[current_key].append(item)

    return meta, body


def _extract_section(md: str, heading: str) -> str:
    """Extrae el texto debajo de un título (## heading) hasta el siguiente título de nivel >= 2."""
    lines = md.splitlines()
    # Buscar encabezado que contenga la frase (p.ej. "Acción inmediata" o "Acción inmediata frente a ...")
    pattern = re.compile(rf"^##+\s*{re.escape(heading)}.*$", flags=re.IGNORECASE)

    start = None
    level = None
    for idx, line in enumerate(lines):
        m = re.match(r"^(#+)\s*(.*)$", line)
        if m and pattern.match(line):
            start = idx + 1
            level = len(m.group(1))
            break

    if start is None:
        return ""

    content_lines = []
    for line in lines[start:]:
        m = re.match(r"^(#+)\s+", line)
        if m and len(m.group(1)) <= (level or 2):
            break
        content_lines.append(line)

    return "\n".join(content_lines).strip()


def obtener_recomendacion_fija(diagnostico: str) -> str:
    """Devuelve una recomendación fija basada en el documento RAG correspondiente."""
    if not diagnostico:
        return "No se encontró información específica del diagnóstico. Intenta con otra imagen."

    doc = cargar_documento_rag(diagnostico)
    if not doc:
        return (
            "No hay información de referencia para este diagnóstico. "
            "Consulta a un técnico local o revisa el documento de soporte."
        )

    meta, body = _parse_frontmatter(doc)

    nombre = meta.get("nombre") or diagnostico
    nombre_cientifico = meta.get("nombre_cientifico")
    severidad = meta.get("severidad")
    curable = meta.get("curable")
    categoria = meta.get("categoria")

    accion_inmediata = _extract_section(body, "Acción inmediata")
    recomendaciones = (
        _extract_section(body, "Corrección recomendada")
        or _extract_section(body, "Opciones de manejo")
    )
    productos = (
        _extract_section(body, "Productos recomendados")
        or _extract_section(body, "Productos fertilizantes")
    )

    partes = []

    encabezado = f"{nombre}"
    if nombre_cientifico:
        encabezado += f" ({nombre_cientifico})"
    partes.append(encabezado)

    if categoria:
        partes.append(f"Categoría: {categoria.capitalize()}")

    if severidad:
        partes.append(f"Severidad: {severidad}")

    if curable is not None:
        partes.append(f"Curable: {'Sí' if curable else 'No'}")

    if accion_inmediata:
        partes.append("\nAcción inmediata:\n" + accion_inmediata)

    if recomendaciones:
        partes.append("\nRecomendaciones generales:\n" + recomendaciones)

    if productos:
        partes.append("\nProductos / tratamiento sugerido:\n" + productos)

    return "\n\n".join(partes)
