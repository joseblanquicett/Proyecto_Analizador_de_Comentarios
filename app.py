from __future__ import annotations

import io
from pathlib import Path
import time

import pandas as pd
import streamlit as st
from transformers import pipeline

from motor import MODELO_SENTIMIENTO, MODELO_TOXICIDAD, analizar_comentarios

st.set_page_config(
    page_title="Grupo 10 | Analizador de comentarios",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
<style>
:root{
  --bg-card:#131c2a;
  --bg-soft:#172538;
  --border:#2a3b52;
  --text:#f2f6fb;
  --muted:#b7c6d8;
  --blue:#2f7fd1;
  --blue-soft:#1d4d7c;
  --green:#29c46a;
  --green-soft:#143624;
  --yellow:#f4c15d;
  --yellow-soft:#3b3118;
  --red:#ff5b5b;
  --red-soft:#411d22;
}
.block-container{padding-top:1.25rem;max-width:1240px}
[data-testid="stMetric"]{background:var(--bg-card);border:1px solid var(--border);border-radius:18px;padding:16px}
.hero{
  background:linear-gradient(135deg,#163252 0%,#225b8f 100%);
  border-radius:24px;
  color:#fff;
  padding:30px 34px;
  margin:0 0 18px;
  box-shadow:0 12px 28px rgba(0,0,0,.22)
}
.hero h1{color:white;font-size:2.1rem;margin:0 0 10px;font-weight:800}
.hero p{color:#edf5fd;margin:0;font-size:1.03rem;line-height:1.6}
.card{
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:18px;
  padding:18px 18px;
  height:100%;
  box-shadow:0 6px 16px rgba(0,0,0,.12)
}
.card h4{margin:0 0 10px;color:var(--text);font-size:1.28rem}
.card p{margin:0;color:var(--muted);line-height:1.62;font-size:.98rem}
.callout{
  background:var(--bg-soft);
  border:1px solid var(--border);
  border-left:6px solid #63a7e8;
  border-radius:14px;
  padding:14px 16px;
  margin:10px 0 18px;
  color:var(--text)
}
.callout p{margin:0;color:var(--muted)}
.result-card{
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:18px;
  padding:20px;
  min-height:205px;
}
.result-card h3{margin:0 0 10px;color:var(--text);font-size:1.25rem}
.result-card .main{font-size:1.7rem;font-weight:800;color:#fff;margin:2px 0 10px}
.result-card .desc{color:var(--muted);line-height:1.6;margin:0}
.result-card .foot{color:#96abc1;margin-top:12px;font-size:.94rem}
.badge{display:inline-block;padding:6px 10px;border-radius:999px;font-size:.84rem;font-weight:700;margin-bottom:10px}
.badge.good{background:rgba(41,196,106,.16);color:#86f2ae;border:1px solid rgba(41,196,106,.28)}
.badge.mid{background:rgba(244,193,93,.14);color:#ffd889;border:1px solid rgba(244,193,93,.26)}
.badge.warn{background:rgba(255,91,91,.15);color:#ff9696;border:1px solid rgba(255,91,91,.28)}
.summary-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:8px 0 16px}
.summary-box{background:var(--bg-card);border:1px solid var(--border);border-radius:18px;padding:18px 16px}
.summary-box .label{color:#9db2c8;font-size:.92rem;margin-bottom:6px}
.summary-box .value{color:#fff;font-size:1.8rem;font-weight:800;line-height:1.2}
.summary-box .sub{color:#b7c6d8;font-size:.9rem;margin-top:6px}
.notice{
  background:#1d2631;
  border:1px solid #544320;
  border-left:6px solid var(--yellow);
  border-radius:14px;
  padding:14px 16px;
  color:#f7ebc6;
  margin:12px 0 8px
}
.notice p{margin:0;color:#f2df9b;line-height:1.55}
.section-title{color:#fff;margin-top:.2rem}
.small-muted{color:#9eb1c6;font-size:.93rem}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{
  background:#111826;
  border:1px solid var(--border);
  border-radius:12px 12px 0 0;
  padding:.55rem .95rem;
}
.stTabs [aria-selected="true"]{background:#16273d !important}
@media (max-width:900px){.summary-grid{grid-template-columns:repeat(2,1fr)}}
@media (max-width:560px){.summary-grid{grid-template-columns:1fr}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando la inteligencia artificial por primera vez...")
def cargar_modelos():
    sentimientos = pipeline(
        task="text-classification",
        model=MODELO_SENTIMIENTO,
        tokenizer=MODELO_SENTIMIENTO,
        framework="pt",
        device=-1,
    )
    toxicidad = pipeline(
        task="text-classification",
        model=MODELO_TOXICIDAD,
        tokenizer=MODELO_TOXICIDAD,
        framework="pt",
        device=-1,
    )
    return sentimientos, toxicidad


def _interpretacion_toxicidad(valor: float, umbral: float) -> tuple[str, str, str]:
    if valor < umbral * 0.65:
        return "Sin señales fuertes", "good", "El comentario no muestra una señal fuerte de agresión o lenguaje dañino."
    if valor < umbral:
        return "Zona de atención", "mid", "Hay algunos elementos que podrían generar duda, pero todavía no alcanza el nivel de alerta configurado."
    return "Revisión recomendada", "warn", "La herramienta detecta una señal alta de posible toxicidad. Lo correcto es revisarlo con criterio humano."


def _mensaje_sentimiento(sentimiento: str) -> str:
    mensajes = {
        "Positivo": "El comentario expresa una valoración favorable, agradecimiento o satisfacción.",
        "Neutro": "El comentario tiene un tono más informativo o equilibrado, sin una emoción muy marcada.",
        "Negativo": "El comentario refleja inconformidad, molestia o crítica frente a una situación.",
    }
    return mensajes.get(sentimiento, "Resultado generado por el modelo de lenguaje.")


def _emoji_sentimiento(sentimiento: str) -> str:
    return {"Positivo": "😊", "Neutro": "😐", "Negativo": "😕"}.get(sentimiento, "💬")


def ejecutar(textos: list[str], umbral: float):
    if not textos:
        st.warning("Escribe al menos un comentario para poder analizarlo.")
        return
    try:
        inicio = time.perf_counter()
        clas_sent, clas_tox = cargar_modelos()
        datos = analizar_comentarios(textos, clas_sent, clas_tox, umbral)
        st.session_state["resultado"] = pd.DataFrame(datos)
        st.session_state["duracion"] = time.perf_counter() - inicio
        st.session_state["umbral_usado"] = umbral
        st.success(f"Análisis completado: {len(datos)} comentario(s) procesado(s).")
    except Exception as exc:
        st.error("No se pudo completar el análisis. Revisa la conexión, la instalación o la memoria disponible del equipo.")
        st.exception(exc)


if "texto_principal" not in st.session_state:
    st.session_state["texto_principal"] = "No me gustó el servicio, pero agradezco que me hayan atendido."

st.markdown(
    """
<div class="hero">
  <h1>💬 Analizador de comentarios en redes sociales</h1>
  <p>
    Esta aplicación ayuda a identificar el <strong>sentimiento</strong> de un comentario
    y a detectar si conviene <strong>revisarlo por posible toxicidad</strong>.
    Su función es apoyar el análisis, no reemplazar la decisión humana.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
<div class="card">
  <h4>¿Qué analiza?</h4>
  <p>Lee comentarios escritos en español y detecta si el tono general es positivo, neutro o negativo.</p>
</div>
""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
<div class="card">
  <h4>¿Qué alerta genera?</h4>
  <p>Indica si un comentario parece ofensivo, agresivo o dañino y si conviene revisarlo antes de tomar decisiones.</p>
</div>
""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
<div class="card">
  <h4>¿Para qué sirve?</h4>
  <p>Apoya la moderación responsable, el análisis de opiniones y el estudio de interacciones en redes sociales.</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="callout">
  <p><strong>Idea central del proyecto:</strong> un comentario puede ser negativo sin ser tóxico. Criticar un servicio no es lo mismo que insultar a una persona.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Ajustes")
    sensibilidad = st.slider(
        "Nivel de sensibilidad de la alerta",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Entre más bajo sea el valor, más fácil será que la app marque un comentario para revisión.",
    )
    st.caption("Valor recomendado: 0.50")
    st.divider()
    st.info("Usa comentarios ficticios o anonimizados. Evita ingresar nombres, teléfonos, direcciones u otra información personal.")
    with st.expander("Ver explicación del ajuste"):
        st.write(
            "- **Más sensible**: marca más comentarios para revisión.\n"
            "- **Menos sensible**: solo marca los casos más fuertes.\n"
            "- La alerta no sanciona automáticamente; solo orienta una revisión humana."
        )
    with st.expander("Información de los modelos"):
        st.write("**Modelo de sentimiento**")
        st.code(MODELO_SENTIMIENTO, language=None)
        st.write("**Modelo de toxicidad**")
        st.code(MODELO_TOXICIDAD, language=None)
        st.caption("Los modelos ya vienen entrenados. La aplicación los utiliza para analizar comentarios de prueba.")

p1, p2, p3 = st.tabs(["📝 Probar un comentario", "📂 Analizar una lista", "📊 Resultados y explicación"])

with p1:
    st.subheader("Prueba rápida con un comentario")
    st.write("Escribe un comentario y la aplicación te mostrará de forma clara cómo lo interpreta.")

    ejemplos = {
        "Crítica respetuosa": "No me gustó el servicio, pero agradezco que me hayan atendido.",
        "Comentario positivo": "Me encantó la publicación, está muy bien explicada.",
        "Posible comentario ofensivo": "Eres un inútil, no sabes hacer nada bien.",
        "Comentario neutro": "La reunión quedó programada para las 3 de la tarde.",
    }
    ejemplo_elegido = st.selectbox("Si quieres, puedes cargar un ejemplo:", list(ejemplos.keys()))
    if st.button("Usar ejemplo seleccionado"):
        st.session_state["texto_principal"] = ejemplos[ejemplo_elegido]

    st.text_area(
        "Comentario en español",
        key="texto_principal",
        height=130,
        help="Puedes escribir un comentario como si fuera una publicación o respuesta en redes sociales.",
    )

    if st.button("Analizar comentario", type="primary", use_container_width=True):
        texto = st.session_state.get("texto_principal", "").strip()
        ejecutar([texto] if texto else [], sensibilidad)

    resultado = st.session_state.get("resultado")
    if isinstance(resultado, pd.DataFrame) and len(resultado) == 1:
        fila = resultado.iloc[0]
        nivel_txt, badge_class, explicacion_tox = _interpretacion_toxicidad(float(fila["Puntuación toxicidad"]), sensibilidad)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"""
<div class="result-card">
  <h3>{_emoji_sentimiento(str(fila['Sentimiento']))} Sentimiento detectado</h3>
  <div class="main">{fila['Sentimiento']}</div>
  <p class="desc">{_mensaje_sentimiento(str(fila['Sentimiento']))}</p>
  <div class="foot">Confianza del modelo: {float(fila['Confianza sentimiento']):.2f}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f"""
<div class="result-card">
  <h3>🛡️ Revisión de toxicidad</h3>
  <span class="badge {badge_class}">{nivel_txt}</span>
  <div class="main">{fila['Alerta toxicidad']}</div>
  <p class="desc">{explicacion_tox}</p>
  <div class="foot">Puntuación detectada: {float(fila['Puntuación toxicidad']):.2f}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        st.markdown(
            """
<div class="notice">
  <p><strong>Interpretación recomendada:</strong> este resultado debe usarse como apoyo. La decisión final siempre debe tomarla una persona, especialmente cuando hay ironía, contexto o discusiones complejas.</p>
</div>
""",
            unsafe_allow_html=True,
        )

with p2:
    st.subheader("Analizar varios comentarios a la vez")
    st.write("Puedes cargar un archivo CSV con una columna llamada **comentario** o usar el archivo de ejemplo del proyecto.")
    ejemplo = Path(__file__).resolve().parent / "datos" / "comentarios_ejemplo.csv"
    archivo = st.file_uploader("Selecciona un archivo CSV", type=["csv"])
    st.download_button(
        "Descargar archivo de ejemplo",
        data=ejemplo.read_bytes(),
        file_name="comentarios_ejemplo.csv",
        mime="text/csv",
    )

    if archivo is None:
        df_entrada = pd.read_csv(ejemplo, encoding="utf-8-sig")
        st.caption("Vista previa de los comentarios ficticios del proyecto.")
    else:
        try:
            df_entrada = pd.read_csv(io.BytesIO(archivo.getvalue()), encoding="utf-8-sig")
        except Exception as exc:
            st.error(f"No se pudo leer el archivo: {exc}")
            df_entrada = pd.DataFrame()

    if not df_entrada.empty:
        st.dataframe(df_entrada.head(20), use_container_width=True, hide_index=True)
        st.markdown(
            """
**Antes de analizar la lista, recuerda:**
- La columna debe llamarse `comentario`.
- Cada fila debe tener texto.
- Para esta demostración, se permiten máximo 100 comentarios por archivo.
"""
        )

    if st.button("Analizar lista de comentarios", type="primary", use_container_width=True):
        if "comentario" not in df_entrada.columns:
            st.error("El archivo debe tener una columna llamada 'comentario'.")
        elif len(df_entrada) > 100:
            st.error("Para esta demostración se permiten como máximo 100 filas por archivo.")
        elif df_entrada["comentario"].isna().any():
            st.error("No se permiten comentarios vacíos en la columna 'comentario'.")
        else:
            comentarios = df_entrada["comentario"].astype(str).str.strip().tolist()
            if any(not x for x in comentarios):
                st.error("Todos los comentarios deben contener texto.")
            else:
                ejecutar(comentarios, sensibilidad)

with p3:
    st.subheader("Resultados del análisis")
    resultado = st.session_state.get("resultado")
    if resultado is None:
        st.info("Aún no hay resultados. Primero realiza un análisis en una de las pestañas anteriores.")
    else:
        total = len(resultado)
        alertas = int((resultado["Alerta toxicidad"] == "Revisar").sum())
        positivos = int((resultado["Sentimiento"] == "Positivo").sum())
        negativos = int((resultado["Sentimiento"] == "Negativo").sum())
        neutros = int((resultado["Sentimiento"] == "Neutro").sum())

        st.markdown(
            f"""
<div class="summary-grid">
  <div class="summary-box">
    <div class="label">Comentarios analizados</div>
    <div class="value">{total}</div>
    <div class="sub">Total procesado por la herramienta</div>
  </div>
  <div class="summary-box">
    <div class="label">Con alerta de revisión</div>
    <div class="value">{alertas}</div>
    <div class="sub">Casos que conviene revisar</div>
  </div>
  <div class="summary-box">
    <div class="label">Comentarios positivos</div>
    <div class="value">{positivos}</div>
    <div class="sub">Opiniones favorables detectadas</div>
  </div>
  <div class="summary-box">
    <div class="label">Comentarios neutros / negativos</div>
    <div class="value">{neutros} / {negativos}</div>
    <div class="sub">Tonos informativos o críticos</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.caption(
            f"Tiempo total aproximado: {st.session_state['duracion']:.1f} segundos. "
            "La primera vez puede tardar más porque se cargan los modelos en memoria."
        )

        st.markdown("### Tabla de resultados")
        st.dataframe(
            resultado,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Confianza sentimiento": st.column_config.ProgressColumn(
                    "Confianza sentimiento", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "Puntuación toxicidad": st.column_config.ProgressColumn(
                    "Puntuación toxicidad", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("### ¿Cómo estuvo el tono general?")
            frecuencia = resultado["Sentimiento"].value_counts().reindex(["Positivo", "Neutro", "Negativo"], fill_value=0)
            st.bar_chart(frecuencia)
            st.caption("Esta gráfica muestra si predominan opiniones positivas, neutras o negativas.")
        with g2:
            st.markdown("### ¿Cuántos comentarios conviene revisar?")
            st.bar_chart(resultado["Alerta toxicidad"].value_counts())
            st.caption("Aquí se observa cuántos comentarios generan alerta de revisión humana.")

        st.download_button(
            "Descargar resultados en CSV",
            data=resultado.to_csv(index=False).encode("utf-8-sig"),
            file_name="resultados_grupo10.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption(f"Estos resultados fueron generados con sensibilidad de alerta = {st.session_state['umbral_usado']:.2f}.")

    st.divider()
    st.subheader("Explicación sencilla")
    st.markdown(
        """
- **Sentimiento**: indica si el comentario transmite una opinión positiva, neutra o negativa.
- **Toxicidad**: indica si el comentario parece ofensivo, agresivo o dañino.
- **Alerta de revisión**: significa que vale la pena que una persona lo revise con más cuidado.
- **Punto clave**: una crítica puede ser negativa sin ser tóxica.
"""
    )

    st.subheader("Reflexión ética y social")
    st.markdown(
        """
- **Libertad de expresión**: no todo comentario fuerte o crítico debe censurarse.
- **Censura automatizada**: usar una IA sin revisión humana puede generar decisiones injustas.
- **Sesgos**: el modelo puede equivocarse con sarcasmo, humor, regionalismos o contextos culturales.
- **Privacidad**: no se deben usar datos personales sin autorización.
- **Responsabilidad**: la decisión final no debe quedar solo en la máquina.
"""
    )

    st.markdown("**Pregunta para el debate:** ¿Hasta qué punto una plataforma debería usar inteligencia artificial para moderar comentarios sin afectar la libertad de expresión de los usuarios?")
    st.caption("Normas que puedes mencionar: Constitución Política de Colombia (arts. 15 y 20), Ley 1581 de 2012 y Recomendación de la UNESCO sobre la Ética de la IA (2021).")
