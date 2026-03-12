# pipeline/schemas.py
"""
Estructuras de datos del resultado del pipeline.

El LLM siempre recibe un PipelineResult sin importar
qué flujo tomó la imagen. Nunca recibe outputs crudos
de los modelos.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class ModelPrediction:
    """
    Resultado de un modelo individual.
    Incluye la clase ganadora, su confianza,
    y las probabilidades de todas las clases.
    """
    clase:        str
    confianza:    float                  # 0.0 – 1.0
    baja_confianza: bool                 # True si confianza < umbral
    probabilidades: Dict[str, float]     # todas las clases con su prob


@dataclass
class PipelineResult:
    """
    Resultado final del pipeline completo.

    flujo: uno de NO_BANANA | ENFERMO | DEFICIT | SANO
    llm_needed: True solo cuando flujo es ENFERMO o DEFICIT
    diagnostico: la clase final (enfermedad o nutriente)
    confianza: confianza del modelo que hizo el diagnóstico final
    baja_confianza: si el modelo no está seguro
    categoria: "enfermedad" | "deficiencia" | None
               usado por el RAG para filtrar el vector store
    predicciones: dict con los outputs crudos de cada modelo
                  que corrió, para trazabilidad y debug
    """
    flujo:          str
    llm_needed:     bool

    diagnostico:    Optional[str]   = None
    confianza:      Optional[float] = None
    baja_confianza: bool            = False
    categoria:      Optional[str]   = None

    # Trazabilidad: qué corrió y qué predijo cada modelo
    predicciones: Dict[str, ModelPrediction] = field(default_factory=dict)

    def to_llm_context(self) -> dict:
        """
        Serializa el resultado para pasarlo al LLM.
        Solo incluye lo que el LLM necesita saber —
        sin objetos internos ni detalles de implementación.
        """
        return {
            "flujo":          self.flujo,
            "diagnostico":    self.diagnostico,
            "confianza_pct":  round(self.confianza * 100, 1) if self.confianza else None,
            "baja_confianza": self.baja_confianza,
            "categoria":      self.categoria,
        }

    def __str__(self) -> str:
        lines = [
            "-" * 50,
            f"  RESULTADO DEL PIPELINE",
            "-" * 50,
            f"  Flujo         : {self.flujo}",
            f"  LLM needed    : {self.llm_needed}",
        ]
        if self.diagnostico:
            lines += [
                f"  Diagnostico   : {self.diagnostico}",
                f"  Confianza     : {self.confianza * 100:.1f}%",
                f"  Baja confianza: {self.baja_confianza}",
                f"  Categoria     : {self.categoria}",
            ]
        lines.append("-" * 50)

        if self.predicciones:
            lines.append("  Modelos corridos:")
            for modelo_id, pred in self.predicciones.items():
                lines.append(
                    f"    {modelo_id.upper()}: {pred.clase} "
                    f"({pred.confianza * 100:.1f}%)"
                )
            lines.append("-" * 50)

        return "\n".join(lines)
