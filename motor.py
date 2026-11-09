"""Lógica independiente de la interfaz del proyecto Grupo 10.

Ninguna regla de palabras clave decide el sentimiento ni la toxicidad:
los puntajes provienen exclusivamente de los modelos Transformers.
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

MODELO_SENTIMIENTO = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MODELO_TOXICIDAD = "unitary/multilingual-toxic-xlm-roberta"

TRADUCIR_SENTIMIENTO = {
    "negative": "Negativo",
    "neutral": "Neutro",
    "positive": "Positivo",
}


def preparar_texto(texto: str) -> str:
    """Anonimiza referencias técnicas frecuentes, no los datos del autor."""
    texto = re.sub(r"https?://\S+|www\.\S+", "http", texto)
    texto = re.sub(r"(?<!\w)@[\w.]+", "@user", texto)
    return " ".join(texto.split())


def analizar_comentarios(
    comentarios: Sequence[str],
    clasificador_sentimiento: Callable,
    clasificador_toxicidad: Callable,
    umbral: float = 0.50,
) -> list[dict]:
    """Clasifica textos con pipelines reales de Hugging Face inyectados.

    El modelo de toxicidad tiene **una sola salida** etiquetada 'toxic'.
    La puntuación (sigmoide) representa una señal de toxicidad y no
    una probabilidad científicamente calibrada de daño ni una sentencia.
    """
    if not 0.0 < umbral < 1.0:
        raise ValueError("El umbral debe estar estrictamente entre 0 y 1.")
    if not comentarios:
        return []

    originales = [str(t).strip() for t in comentarios]
    if any(not t for t in originales):
        raise ValueError("Los comentarios no pueden estar vacíos.")
    depurados = [preparar_texto(t) for t in originales]

    # Uso de lotes evita recargar los modelos para cada comentario.
    opciones = {"truncation": True, "max_length": 256, "batch_size": 8}
    pred_sent = clasificador_sentimiento(depurados, **opciones)
    pred_tox = clasificador_toxicidad(depurados, **opciones)

    if len(pred_sent) != len(originales) or len(pred_tox) != len(originales):
        raise RuntimeError("Los modelos devolvieron un número de resultados inesperado.")

    filas = []
    for original, sent, tox in zip(originales, pred_sent, pred_tox):
        etiqueta = str(sent["label"]).lower().strip()
        if etiqueta not in TRADUCIR_SENTIMIENTO:
            raise ValueError(f"Etiqueta de sentimiento no reconocida: {etiqueta}")
        valor = float(tox["score"])
        if not 0 <= valor <= 1:
            raise ValueError("Puntuación de toxicidad fuera del intervalo [0, 1].")
        if str(tox["label"]).lower() != "toxic":
            raise ValueError("El modelo de toxicidad no retornó la etiqueta 'toxic'.")
        filas.append({
            "Comentario": original,
            "Sentimiento": TRADUCIR_SENTIMIENTO[etiqueta],
            "Confianza sentimiento": round(float(sent["score"]), 4),
            "Puntuación toxicidad": round(valor, 4),
            "Alerta toxicidad": "Revisar" if valor >= umbral else "Sin alerta",
        })
    return filas
