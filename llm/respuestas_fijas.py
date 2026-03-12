# llm/respuestas_fijas.py
"""
Respuestas fijas por diagnóstico, generadas a partir de los documentos
de DOCUMENTOS RAG/. No requieren conexión a ningún LLM.

Formato de cada entrada:
    {
        "nombre":       nombre visible en español,
        "tipo":         "enfermedad" | "deficiencia",
        "curable":      True | False | None,
        "severidad":    "Alta" | "Media" | "Baja",
        "descripcion":  texto breve para el agricultor,
        "acciones":     lista de pasos ordenados por urgencia,
        "productos":    lista de { "nombre", "uso", "donde_comprar" }
    }
"""

RESPUESTAS = {

    # ─────────────────────────────────────────────────────
    # ENFERMEDADES
    # ─────────────────────────────────────────────────────

    "Cordana": {
        "nombre":      "Enfermedad de Cordana",
        "tipo":        "enfermedad",
        "curable":     True,
        "severidad":   "Baja",
        "descripcion": (
            "Cordana es una enfermedad fúngica de baja importancia económica "
            "causada por el hongo Cordana musae. Afecta principalmente las hojas más "
            "viejas y no impacta directamente la producción ni la calidad del racimo. "
            "No requiere fungicidas bajo manejo agronómico normal."
        ),
        "acciones": [
            "Confirma que las manchas están solo en hojas bajas (hoja 8 o más). "
            "Si aparecen en hojas 1-4, consulta con un técnico — podría ser Sigatoka Negra.",
            "Realiza defoliación sanitaria: retira las hojas afectadas con machete desinfectado "
            "y saca el material del campo.",
            "Mejora la ventilación del lote revisando la densidad de siembra y eliminando malezas excesivas.",
            "Corrige problemas de drenaje si el suelo retiene agua — la humedad excesiva favorece el hongo.",
            "NO apliques fungicidas en casos normales de Cordana en hojas bajas: no está justificado económicamente.",
        ],
        "productos": [
            {
                "nombre": "Cobre (hidróxido de cobre u oxicloruro)",
                "uso": "Solo si hay brote severo en hojas productivas (hojas 1-5). Fungicida de contacto multisitio.",
                "donde_comprar": "Agripac, Ecuaquímica, distribuidores locales de insumos agrícolas",
            },
            {
                "nombre": "Mancozeb",
                "uso": "Solo en casos excepcionales. Verifica LMR vigente antes de aplicar en lotes de exportación.",
                "donde_comprar": "Agripac, Ecuaquímica",
            },
        ],
    },

    "Fusarium": {
        "nombre":      "Fusarium oxysporum (Mal de Panamá)",
        "tipo":        "enfermedad",
        "curable":     False,
        "severidad":   "Alta",
        "descripcion": (
            "El Mal de Panamá es una marchitez vascular causada por el hongo de suelo "
            "Fusarium oxysporum f. sp. cubense Raza 1 (Foc R1). Invade el sistema vascular "
            "de la planta bloqueando el transporte de agua y nutrientes hasta causar su muerte. "
            "NO existe control químico efectivo. Las esporas del hongo pueden sobrevivir en el "
            "suelo hasta 30 años. Es obligatorio notificar a AGROCALIDAD."
        ),
        "acciones": [
            "URGENTE: Notifica inmediatamente a AGROCALIDAD, especialmente si tienes variedad "
            "Cavendish — cualquier sospecha de marchitez vascular debe descartar la Raza 4 Tropical.",
            "Confirma el diagnóstico: realiza un corte transversal del pseudotallo a 50 cm del suelo. "
            "Si observas decoloración vascular rojiza-café, envía muestra al laboratorio de "
            "INIAP Pichilingue (Los Ríos) o AGROCALIDAD para diagnóstico definitivo.",
            "Aisla el área afectada de inmediato. Coloca señalización alrededor de las plantas "
            "sintomáticas y NO muevas suelo, herramientas ni material vegetal fuera del área infectada.",
            "Elimina las plantas infectadas: extrae pseudotallo completo con raíces, traslada "
            "el material en bolsas cerradas y aplica cal agrícola al hoyo de extracción.",
            "Desinfecta herramientas con formol al 2% o hipoclorito de sodio al 5% antes de "
            "usarlas en otras áreas del campo.",
        ],
        "productos": [
            {
                "nombre": "Trichoderma spp. (cepas nativas)",
                "uso": "Control biológico: reduce severidad y retrasa síntomas. Única opción de manejo activo disponible.",
                "donde_comprar": "Biohealth, Dinamics — distribuidores de bioinsumos agrícolas en Ecuador",
            },
            {
                "nombre": "Cal agrícola",
                "uso": "Aplicar en hoyos de extracción de plantas infectadas para reducir inóculo en suelo.",
                "donde_comprar": "Distribuidores locales de materiales de construcción e insumos agrícolas",
            },
        ],
    },

    "SigatokaNegra": {
        "nombre":      "Sigatoka Negra",
        "tipo":        "enfermedad",
        "curable":     True,
        "severidad":   "Alta",
        "descripcion": (
            "La Sigatoka Negra es la enfermedad foliar más destructiva del banano a nivel mundial, "
            "causada por el hongo Pseudocercospora fijiensis. Destruye el tejido foliar reduciendo "
            "la fotosíntesis y provoca la maduración prematura del fruto. Sin control adecuado puede "
            "causar pérdidas del 100% de la producción comercializable. Requiere ciclos de fungicidas "
            "de 8 a 30 días según la presión de la enfermedad."
        ),
        "acciones": [
            "Evalúa el estadio de la enfermedad en hojas 1 y 2 usando la Prueba de Hoja Individual (PHI/SLT). "
            "Si el estadio es 4 o mayor, la presión es alta y debes aplicar fungicida curativo de inmediato.",
            "Realiza defoliación sanitaria: retira todas las hojas con más del 50% de área necrosada.",
            "Desinfecta las herramientas con hipoclorito al 5% entre planta y planta.",
            "Aplica fungicida curativo según el estadio detectado y sigue la rotación de grupos FRAC/COMTEC "
            "para evitar resistencia.",
            "En invierno (alta humedad): reduce el intervalo de aplicación a 8-10 días. En verano, "
            "puedes extenderlo hasta 30-90 días.",
        ],
        "productos": [
            {
                "nombre": "Timorex Gold (extracto de árbol de té)",
                "uso": "Preventivo, curativo y antiesporulante. Grupo FRAC BM01.",
                "donde_comprar": "ADAMA Ecuador",
            },
            {
                "nombre": "Regev (Folpet + Tebuconazole)",
                "uso": "Preventivo y curativo. Grupos FRAC M04 + G1.",
                "donde_comprar": "ADAMA Ecuador",
            },
            {
                "nombre": "Difenoconazol (genérico)",
                "uso": "Curativo. Grupo FRAC G1.",
                "donde_comprar": "Agripac, Ecuaquímica, Quifatex",
            },
            {
                "nombre": "Azoxistrobina (genérico)",
                "uso": "Preventivo sistémico. Grupo FRAC C3.",
                "donde_comprar": "Agripac, Ecuaquímica",
            },
            {
                "nombre": "Mancozeb (genérico)",
                "uso": "Protectante de contacto multisitio. Grupo FRAC M03.",
                "donde_comprar": "Amplia disponibilidad nacional",
            },
        ],
    },

    # ─────────────────────────────────────────────────────
    # DEFICIENCIAS NUTRICIONALES
    # ─────────────────────────────────────────────────────

    "Boron": {
        "nombre":      "Deficiencia de Boro",
        "tipo":        "deficiencia",
        "curable":     True,
        "severidad":   "Media",
        "descripcion": (
            "El Boro es esencial para la división celular y la formación de la pared celular. "
            "La deficiencia se manifiesta en los tejidos más jóvenes, mostrando rayas blancas "
            "perpendiculares a las nervaduras en hojas jóvenes y deformaciones en el fruto. "
            "Común en suelos arenosos con pH alto o en condiciones de sequía. "
            "⚠️ Atención: el Boro tiene un margen estrecho entre deficiencia y toxicidad — "
            "no excedas las dosis recomendadas."
        ),
        "acciones": [
            "Realiza análisis foliar (hoja 3) para confirmar nivel por debajo de 10 ppm.",
            "Corrección foliar: aplica ácido bórico al 0.1-0.3% (100-300 g/100 L agua) "
            "en 2-3 aplicaciones cada 15 días.",
            "Corrección de suelo: aplica 0.5-1 kg de B elemental/ha/año (o 3-5 kg de bórax al 15-17% B).",
            "Revisa hojas emergentes a partir de las 2-4 semanas: las hojas nuevas ya no deberían "
            "presentar rayas blancas ni deformaciones.",
            "Las hojas ya afectadas NO se recuperan; la mejora se observa en tejido nuevo.",
        ],
        "productos": [
            {
                "nombre": "Ácido bórico (H₃BO₃) — 17% B",
                "uso": "Aplicación foliar. Soluble en agua caliente.",
                "donde_comprar": "Agripac, Ecuaquímica, farmacias industriales",
            },
            {
                "nombre": "Bórax (Na₂B₄O₇·10H₂O) — 11% B",
                "uso": "Aplicación al suelo. Bajo costo.",
                "donde_comprar": "Distribuidores locales",
            },
            {
                "nombre": "Solubor (octaborato de sodio) — 20% B",
                "uso": "Alta solubilidad. Aplicación foliar o fertirrigación.",
                "donde_comprar": "Distribuidores especializados",
            },
        ],
    },

    "Calcium": {
        "nombre":      "Deficiencia de Calcio",
        "tipo":        "deficiencia",
        "curable":     True,
        "severidad":   "Media",
        "descripcion": (
            "El Calcio es esencial para la estructura de la pared celular y el desarrollo "
            "radicular. La deficiencia aparece primero en hojas jóvenes: la hoja emergente "
            "sale deformada con el ápice doblado o en punta ('cigarro mal abierto'). "
            "Puede ser bloqueada por exceso de amonio, potasio, magnesio o sodio en el suelo. "
            "Raíces débiles y cortas son el primer síntoma invisible de esta deficiencia."
        ),
        "acciones": [
            "Realiza análisis foliar (hoja 3) para confirmar nivel por debajo de 0.4%. "
            "Complementa con análisis de suelo y evaluación del pH.",
            "Corrección estándar: aplica 100-200 kg CaO/ha/año con nitrato de calcio "
            "(disponibilidad rápida) o 500-1000 kg CaO/ha/año con enmiendas de efecto prolongado.",
            "Si el problema es antagonismo con exceso de K o Mg, ajusta primero el programa "
            "de fertilización antes de corregir el Ca.",
            "Yeso agrícola (CaSO₄) puede aplicarse en cualquier condición de pH.",
            "Espera hojas nuevas con forma normal en 3-4 semanas tras iniciar la corrección con nitrato de calcio.",
        ],
        "productos": [
            {
                "nombre": "Nitrato de calcio Ca(NO₃)₂ — ~19% Ca + 15% N",
                "uso": "Corrección rápida. Ideal para deficiencias urgentes.",
                "donde_comprar": "Agripac, Ecuaquímica",
            },
            {
                "nombre": "Yeso agrícola (CaSO₄·2H₂O) — ~23% Ca + 18% S",
                "uso": "No altera el pH. Aplicar al voleo.",
                "donde_comprar": "Distribuidores locales, canteras",
            },
            {
                "nombre": "Calcio foliar quelado",
                "uso": "Para corrección foliar rápida en deficiencias agudas.",
                "donde_comprar": "Agripac, Ecuaquímica",
            },
        ],
    },

    "Iron": {
        "nombre":      "Deficiencia de Hierro",
        "tipo":        "deficiencia",
        "curable":     True,
        "severidad":   "Media",
        "descripcion": (
            "El Hierro es cofactor en la síntesis de clorofila. La deficiencia produce clorosis "
            "total en hojas jóvenes: toda la hoja se vuelve amarilla pálida o blanquecina, "
            "incluidas las nervaduras. La causa más frecuente NO es la falta del elemento en "
            "el suelo, sino su inmovilización por pH alcalino (suelos con pH > 7). "
            "Corregir el pH del suelo es más eficiente que solo aplicar más hierro."
        ),
        "acciones": [
            "Verifica el pH del suelo antes de aplicar hierro: si el pH es mayor a 7, "
            "la acidificación del suelo es la solución estructural.",
            "Para corrección rápida: aplica hierro quelado (Fe-EDTA o Fe-EDDHA) foliar "
            "a 1-2 kg/ha en 200-400 L de agua. Respuesta visible en 2-3 semanas.",
            "En suelos con pH > 7 usa Fe-EDDHA (quelato más estable en condiciones alcalinas).",
            "Para corrección de suelo: aplica Fe-EDTA o sulfato ferroso via fertirrigación "
            "en suelos con pH entre 5 y 6.5.",
            "Si la deficiencia reaparece tras la corrección, prioriza el análisis y ajuste del pH.",
        ],
        "productos": [
            {
                "nombre": "Hierro quelado Fe-EDTA — 6-13% Fe",
                "uso": "Para suelos con pH 4-7. Aplicación foliar o riego.",
                "donde_comprar": "Agripac, Ecuaquímica",
            },
            {
                "nombre": "Hierro quelado Fe-EDDHA — 6% Fe",
                "uso": "Para suelos alcalinos (pH 7-9). Mayor estabilidad.",
                "donde_comprar": "Agripac, distribuidores especializados",
            },
            {
                "nombre": "Sulfato ferroso (FeSO₄) — ~20% Fe",
                "uso": "Bajo costo. Solo efectivo en suelos ácidos (pH < 6.5).",
                "donde_comprar": "Distribuidores locales de insumos",
            },
        ],
    },

    "Magnesium": {
        "nombre":      "Deficiencia de Magnesio",
        "tipo":        "deficiencia",
        "curable":     True,
        "severidad":   "Media",
        "descripcion": (
            "El Magnesio es el componente central de la clorofila y esencial para la fotosíntesis. "
            "Es móvil en la planta, por lo que ante escasez los síntomas aparecen primero en las "
            "hojas más viejas (basales): clorosis intervenal con venas que conservan el color verde. "
            "Los pecíolos presentan manchas morado-azuladas, señal diagnóstica específica en banano. "
            "Frecuente cuando hay exceso de potasio en el suelo (antagonismo K-Mg)."
        ),
        "acciones": [
            "Realiza análisis foliar (hoja 3) para confirmar nivel por debajo de 0.2%. "
            "Evalúa también el nivel de K para identificar antagonismo.",
            "Si hay exceso de K, ajusta el programa de fertilización potásica antes de aplicar Mg.",
            "Aplicación foliar de emergencia: sulfato de magnesio al 1-2% para corrección visible en pocas semanas.",
            "Corrección de suelo: sulfato de magnesio (MgSO₄) o dolomita (solo si pH < 5). "
            "Dosis de referencia: 40-100 kg MgO/ha/año, fraccionado.",
            "Revisa el color de los pecíolos en hojas nuevas: la ausencia de manchas moradas indica corrección exitosa.",
        ],
        "productos": [
            {
                "nombre": "Sulfato de magnesio (MgSO₄·7H₂O) — ~10% Mg + 13% S",
                "uso": "Soluble. Aplicación foliar o al suelo. Acción dual Mg y S.",
                "donde_comprar": "Agripac, Ecuaquímica",
            },
            {
                "nombre": "Sulpomag (doble sulfato K-Mg-S)",
                "uso": "Ideal cuando hay deficiencia simultánea de K y Mg.",
                "donde_comprar": "Distribuidores especializados",
            },
            {
                "nombre": "Dolomita CaMg(CO₃)₂ — ~13% Mg + ~22% Ca",
                "uso": "Para suelos con pH < 5. Efecto de mediano plazo.",
                "donde_comprar": "Distribuidores locales",
            },
        ],
    },

    "Manganese": {
        "nombre":      "Deficiencia de Manganeso",
        "tipo":        "deficiencia",
        "curable":     True,
        "severidad":   "Baja-Media",
        "descripcion": (
            "El Manganeso activa enzimas clave en el fotosistema II. La deficiencia produce "
            "una clorosis en forma de 'peine' en los bordes de las hojas: las nervaduras "
            "secundarias conservan el color verde mientras el tejido entre ellas se vuelve "
            "amarillo-verde pálido, creando un patrón estriado. Frecuente en suelos alcalinos "
            "(pH > 7) o con exceso de hierro y calcio."
        ),
        "acciones": [
            "Realiza análisis foliar (hoja 3) para confirmar nivel por debajo de 25 ppm. "
            "Complementa con análisis de pH del suelo.",
            "Si el pH es mayor a 7, la acidificación del suelo es más efectiva a largo plazo "
            "que solo aplicar Manganeso.",
            "Corrección foliar rápida: sulfato de manganeso (MnSO₄) al 0.2-0.5% en 2-3 "
            "aplicaciones cada 15 días.",
            "Corrección de suelo: 2-5 kg Mn elemental/ha/año como sulfato de manganeso "
            "en suelos con pH < 6.5.",
            "Si la deficiencia persiste o reaparece, revisa el pH del suelo y ajústalo si supera 6.5.",
        ],
        "productos": [
            {
                "nombre": "Sulfato de manganeso (MnSO₄) — 28-32% Mn",
                "uso": "Para aplicación foliar o al suelo. Bajo costo.",
                "donde_comprar": "Agripac, Ecuaquímica",
            },
            {
                "nombre": "Manganeso quelado (Mn-EDTA) — 5-7% Mn",
                "uso": "Mayor eficiencia foliar. Útil en pH moderadamente alto.",
                "donde_comprar": "Distribuidores especializados",
            },
            {
                "nombre": "Fertilizante foliar multinutriente (Mn + Fe + Zn)",
                "uso": "Para corrección simultánea de varios micronutrientes.",
                "donde_comprar": "Agripac, Ecuaquímica",
            },
        ],
    },

    "Potassium": {
        "nombre":      "Deficiencia de Potasio",
        "tipo":        "deficiencia",
        "curable":     True,
        "severidad":   "Media-Alta",
        "descripcion": (
            "El Potasio es el nutriente más importante para la producción de banano. "
            "Participa en el llenado del fruto, transporte de azúcares y síntesis de proteínas. "
            "Un lote de 70 t/ha/año extrae hasta 400 kg de K elemental por año — extracción "
            "extraordinariamente alta que debe reponerse continuamente. La deficiencia produce "
            "clorosis amarillo-naranja en los bordes de las hojas más viejas y dedos cortos "
            "que no alcanzan el calibre mínimo de exportación."
        ),
        "acciones": [
            "Realiza análisis foliar (hoja 3) para confirmar nivel por debajo de 3.0%. "
            "Repite el análisis cada 6 meses.",
            "Análisis de suelo para determinar el nivel de K disponible e identificar "
            "posibles antagonismos con Ca y Mg.",
            "Dosis de corrección de referencia: 400-480 kg K₂O/ha/año, fraccionado en "
            "4-6 aplicaciones para mejorar absorción y reducir pérdidas por lixiviación.",
            "Incorpora K permanentemente en el programa de fertilización — es el nutriente "
            "de mayor extracción y no puede omitirse en ningún ciclo.",
            "Evalúa el calibre de los dedos en racimos en desarrollo para confirmar mejora productiva "
            "a mediano plazo.",
        ],
        "productos": [
            {
                "nombre": "Muriato de potasio (KCl) — 60% K₂O",
                "uso": "Fuente más económica y más usada en banano ecuatoriano.",
                "donde_comprar": "Agripac, Ecuaquímica, Fertisa",
            },
            {
                "nombre": "Sulfato de potasio (K₂SO₄) — 50% K₂O",
                "uso": "Aporta también S. Preferible en suelos de alta salinidad.",
                "donde_comprar": "Agripac, distribuidores locales",
            },
            {
                "nombre": "Nitrato de potasio (KNO₃) — 44% K₂O + 13% N",
                "uso": "Para fertirrigación o aplicación foliar complementaria.",
                "donde_comprar": "Agripac, Ecuaquímica",
            },
        ],
    },

    "Sulphur": {
        "nombre":      "Deficiencia de Azufre",
        "tipo":        "deficiencia",
        "curable":     True,
        "severidad":   "Media",
        "descripcion": (
            "El Azufre es componente esencial en la síntesis de proteínas y aminoácidos. "
            "La deficiencia produce hojas jóvenes de color blanco-amarillento o amarillo pálido "
            "generalizado. A diferencia de la deficiencia de Hierro, las nervaduras pueden "
            "aparecer más claras o engrosadas. Frecuente en suelos arenosos o con alta lixiviación "
            "y en programas de fertilización que no incluyen fuentes azufradas."
        ),
        "acciones": [
            "Realiza análisis foliar (hoja 3) para confirmar nivel por debajo de 0.23%.",
            "Dosis de referencia: 100-200 kg SO₄/ha/año, fraccionado en 4-6 aplicaciones.",
            "Usa fuentes de fertilizante que contengan S: sulfato de potasio, sulfato de magnesio, "
            "sulpomag o sulfato de amonio.",
            "Para corrección de emergencia: sulfato de magnesio foliar al 1-2%.",
            "Incorpora S permanentemente en el programa de fertilización para prevenir recaídas, "
            "especialmente en suelos arenosos.",
        ],
        "productos": [
            {
                "nombre": "Sulfato de magnesio (MgSO₄) — 13% S + 10% Mg",
                "uso": "Acción dual S y Mg. Bajo costo. Foliar o suelo.",
                "donde_comprar": "Agripac, Ecuaquímica",
            },
            {
                "nombre": "Sulfato de potasio (K₂SO₄) — 18% S + 50% K₂O",
                "uso": "Repone S y K simultáneamente.",
                "donde_comprar": "Agripac, Ecuaquímica",
            },
            {
                "nombre": "Sulfato de amonio (NH₄)₂SO₄ — 24% S + 21% N",
                "uso": "Fuente de N y S. Puede acidificar el suelo con uso prolongado.",
                "donde_comprar": "Agripac, Fertisa, Ecuaquímica",
            },
        ],
    },

}


def obtener_recomendacion_fija(diagnostico: str) -> str:
    """Devuelve la recomendación como texto plano (fallback)."""
    entrada = RESPUESTAS.get(diagnostico)
    if not entrada:
        return f"Se detectó: {diagnostico}. Consulta con un técnico agrícola."
    return entrada["descripcion"]


def obtener_detalle_fijo(diagnostico: str) -> dict | None:
    """
    Devuelve el dict estructurado completo para el diagnóstico dado.
    Usado por el frontend para renderizar tabs con formato.
    Retorna None si el diagnóstico no está en la base de conocimiento.
    """
    return RESPUESTAS.get(diagnostico)
