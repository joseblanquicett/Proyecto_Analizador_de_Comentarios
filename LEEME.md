# Grupo 10 | Análisis de sentimiento y toxicidad en redes sociales

Proyecto académico de **Análisis de Inteligencia Artificial** (Python + Transformers + Hugging Face + Streamlit).

## ¿Qué hace?

Clasifica cada comentario según su **sentimiento** (positivo, neutro o negativo) y obtiene una **puntuación de toxicidad** entre 0 y 1. Si supera un umbral configurable, el texto queda **marcado para revisión humana**, sin bloqueo automático. Procesa un comentario, lista de prueba o CSV (hasta 100 filas). Permite descargar los resultados.

**Importante:** sentimiento negativo y toxicidad no son equivalentes. El número de toxicidad es una salida del modelo, no una prueba legal ni una probabilidad calibrada de que la persona sea peligrosa.

## Requisitos del computador

- Python **3.13** (64 bits) para este paquete actualizado; funciona con la versión 3.13.15 del usuario. Si ya tienes otro entorno `.venv` creado con otra versión, renómbralo antes de iniciar.
- Acceso a internet la **primera vez** para instalar paquetes y descargar los modelos de Hugging Face.
- Se recomiendan **8 GB de RAM o más**, idealmente **16 GB**, y varios GB libres en disco. Los modelos son grandes y la primera inferencia con CPU puede tardar.
- En Windows, PowerShell o Símbolo del sistema. No necesitas tarjeta gráfica ni API de pago.

## Instalación rápida en Windows

1. Descomprime el archivo ZIP, abre la carpeta `Grupo10_IA`.
2. Abre un terminal dentro de ella: en el Explorador de Windows escribe `cmd` en la barra de dirección y pulsa Enter.
3. Escribe:

```bat
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

O ejecuta con doble clic `INICIAR_WINDOWS.bat` (adaptado para Python 3.13).

4. Se abrirá el navegador con la aplicación, normalmente en `http://localhost:8501`.
5. Pulsa **Analizar comentario** o **Analizar lista**: en ese momento se descargan los modelos si no están en caché. Una vez descargados, se usan localmente. **Revisa y prueba esto antes de grabar el video.**

## macOS / Linux

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Guion de demostración (1-2 minutos)

1. Ejecuta `streamlit run app.py` y muestra la interfaz.
2. Ingresa **"El servicio fue lento y no me gustó"**; describe el sentimiento y la toxicidad **que realmente arroje** el modelo.
3. Ingresa **"Eres un inútil, no sirves para nada"**; contrasta las dos clasificaciones reales. Los resultados pueden variar.
4. En la pestaña *Archivo CSV*, usa los 12 ejemplos ficticios y pulsa **Analizar lista**.
5. En *Resultados*, muestra la tabla, las gráficas y exporta el CSV.
6. Cambia el umbral en la barra lateral, vuelve a ejecutar el análisis y explica por qué el umbral **cambia las alertas, no el juicio humano**.

## Estructura del proyecto

| Archivo | Función |
|---|---|
| `app.py` | Interfaz Streamlit, carga de modelos reales, métricas, gráficas y CSV. |
| `motor.py` | Preparación de textos y lógica de clasificación. |
| `requirements.txt` | Dependencias Python. |
| `datos/comentarios_ejemplo.csv` | 12 comentarios ficticios para pruebas y video. |
| `pruebas/test_motor.py` | Pruebas unitarias del código con simuladores explícitos. |
| `documentos/Resumen_Ejecutivo_Grupo10.docx` | Documento de exactamente una página. |
| `documentos/Guion_Video_Grupo10.docx` | Texto para los cuatro integrantes y tiempos del video. |

## Modelos usados y limitaciones

- **Sentimiento:** https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment . Modelo XLM-RoBERTa multilingüe para sentimiento, entrenado previamente por sus autores.
- **Toxicidad:** https://huggingface.co/unitary/multilingual-toxic-xlm-roberta . Clasificador multilingüe de toxicidad, salida `toxic` con valor de 0 a 1. **No tiene una etiqueta separada `no tóxico`:** el umbral de la aplicación genera la alerta. Su ficha advierte que sus resultados pueden diferir de la biblioteca Detoxify.

El proyecto realiza **inferencia**, no entrenamiento. No promete exactitud para dialectos, humor, ironía, ambigüedad o escenarios reales. Los ejemplos son **ficticios**. No hay conexión a redes sociales ni captura de datos personales. El CSV se mantiene en la sesión de Streamlit; la aplicación no lo envía a un servicio externo de inferencia. Los pesos se descargan desde Hugging Face.

## Comprobación técnica sin descargar modelos

Se puede verificar la lógica de manera separada ejecutando:

```bash
python -m pip install pytest
python -m pytest -q pruebas
```

**Estas pruebas usan clasificadores simulados solo para probar el manejo de datos y los umbrales. No sustituyen la demostración con modelos reales.**

## Ética y normas colombianas

- **Constitución Política:** artículos 15 (intimidad y hábeas data) y 20 (libertad de expresión).
- **Ley 1581 de 2012:** marco de tratamiento de datos personales cuando aplique.
- **UNESCO (2021):** recomendación ética sobre justicia, transparencia y supervisión humana; es orientación internacional y no se presenta como ley colombiana.
- Mantener una persona responsable de moderación, posibilidad de revisión y reclamación de errores; evitar censura automática y procesamiento de datos sensibles.

Referencias:
- https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=49981
- https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=4125
- https://www.unesco.org/en/legal-affairs/recommendation-ethics-artificial-intelligence

## Pregunta para debate

**¿Quién debe decidir si un comentario se elimina de una red social: un algoritmo, una persona o una combinación de ambos?**

## Nota de verificación

Se incluye código de ejecución real con modelos públicos verificados en sus fichas. La ejecución de inferencia end-to-end requiere descargar los pesos (aproximadamente varios GB entre modelos y entorno) y **no se certifica aquí**; la prueba final debe hacerse en el equipo que grabará el video. No se incluyen capturas ni porcentajes inventados.
