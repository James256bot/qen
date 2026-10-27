TRANSLATIONS = {
    "en": {
        "run": "run simulation",
        "download_csv": "download CSV",
        "copy_permalink": "copy permalink",
        "live_update": "live update (drag a slider to re-run instantly)",
        "recent_runs": "recent runs",
        "raw_response": "raw response",
        "lessons": "lessons",
        "system": "system",
        "error_prefix": "error",
    },
    "sw": {
        "run": "endesha simulation",
        "download_csv": "pakua CSV",
        "copy_permalink": "nakili kiungo",
        "live_update": "sasisho la moja kwa moja (vuta kitelezi)",
        "recent_runs": "matukio ya hivi karibuni",
        "raw_response": "jibu ghafi",
        "lessons": "masomo",
        "system": "mfumo",
        "error_prefix": "kosa",
    },
    "fr": {
        "run": "lancer la simulation",
        "download_csv": "telecharger CSV",
        "copy_permalink": "copier le permalien",
        "live_update": "mise a jour en direct (glissez un curseur)",
        "recent_runs": "executions recentes",
        "raw_response": "reponse brute",
        "lessons": "lecons",
        "system": "systeme",
        "error_prefix": "erreur",
    },
    "es": {
        "run": "ejecutar simulacion",
        "download_csv": "descargar CSV",
        "copy_permalink": "copiar enlace permanente",
        "live_update": "actualizacion en vivo (arrastra un control)",
        "recent_runs": "ejecuciones recientes",
        "raw_response": "respuesta sin procesar",
        "lessons": "lecciones",
        "system": "sistema",
        "error_prefix": "error",
    },
}

DEFAULT_LANG = "en"


def get_strings(lang: str):
    if lang not in TRANSLATIONS:
        lang = DEFAULT_LANG
    return {"lang": lang, "strings": TRANSLATIONS[lang]}
