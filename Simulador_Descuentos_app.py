import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# =============================
# Page config
# =============================
st.set_page_config(
    page_title="Simulador Bolsa de Descuentos",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# =============================
# Acceso por token en URL (opción 2)
# =============================
try:
    APP_TOKENS = st.secrets["APP_TOKENS"]  # puede ser string o lista en secrets
except Exception:
    APP_TOKENS = os.getenv("APP_TOKENS", "")

def _normalize_tokens(x):
    # Acepta: "abc,def" o ["abc","def"]
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(t).strip() for t in x if str(t).strip()]
    s = str(x).strip()
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]

VALID_TOKENS = set(_normalize_tokens(APP_TOKENS))

def require_token_access():
    # Si no configuraste tokens, no bloquea (si quieres bloquear siempre, cambia a st.stop())
    if not VALID_TOKENS:
        return True

    q = st.query_params  # Streamlit >= 1.30
    token = (q.get("token", "") or "").strip()

    if token not in VALID_TOKENS:
        st.title("Acceso restringido")
        st.write("Este simulador requiere un enlace autorizado.")
        st.caption("Si no tienes el enlace, solicita acceso al administrador.")
        st.stop()

require_token_access()

# =============================
# Config / Persistencia
# =============================
DATA_DIR = Path("data_store")
DATA_DIR.mkdir(exist_ok=True)

try:
    ADMIN_PASSWORD = st.secrets["APP_ADMIN_PASSWORD"]
except Exception:
    ADMIN_PASSWORD = os.getenv("APP_ADMIN_PASSWORD", "")

# =============================
# Helpers
# =============================
def normalize_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()

def strip_accents(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    rep = (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), ("ñ", "n"))
    for a, b in rep:
        text = text.replace(a, b).replace(a.upper(), b.upper())
    return text

def display_tipo(tipo: str) -> str:
    """Solo visual: Regular / Retención Inmediata / Regreso"""
    if not tipo:
        return ""
    return str(tipo).strip().title()

def riesgo_icono_color(pct: float):
    """
    Devuelve (icono, descripcion) para usar en los KPI.
    Ajusta umbrales si lo requieres.
    """
    try:
        pct = float(pct)
    except Exception:
        pct = 0.0

    if pct <= 8:
        return "🟢", "Conservador / bajo riesgo"
    elif pct <= 14:
        return "🟡", "Equilibrado / riesgo medio"
    else:
        return "🔴", "Agresivo / alto riesgo"

def fmt_crc(x, decimals=0):
    """59.291.800 (miles con punto)"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        x = float(x)
    except Exception:
        return "—"

    if decimals == 0:
        s = f"{int(round(x)):,.0f}"
    else:
        s = f"{x:,.{decimals}f}"

    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_crc_with_symbol(x: float) -> str:
    return f"₡ {fmt_crc(x, 0)}"

def sort_key_period(ap: str):
    try:
        y, p = str(ap).split("-", 1)
        return (int(y), int(p))
    except Exception:
        return (9999, 9999)

TIPO_ORDER = {"regular": 1, "retencion inmediata": 2, "retención inmediata": 2, "regreso": 3}
def tipo_sort_key(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip().str.lower()
    return x.map(lambda v: TIPO_ORDER.get(v, 99))

# ---- Parse de dinero robusto (₡, puntos, comas, etc.) ----
def parse_money(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.str.replace(r"[^0-9,\.\-]", "", regex=True)

    def parse_one(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if x == "" or x.upper() in ("NA", "N/A", "NONE", "<NA>"):
            return np.nan

        # si tiene ambos, decide por el último separador como decimal
        if "," in x and "." in x:
            if x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")
            else:
                x = x.replace(",", "")
        elif "," in x and "." not in x:
            parts = x.split(",")
            if len(parts) == 2 and len(parts[1]) in (1, 2):
                x = x.replace(",", ".")
            else:
                x = x.replace(",", "")
        elif "." in x and "," not in x:
            parts = x.split(".")
            if len(parts) == 2 and len(parts[1]) == 3:
                x = x.replace(".", "")

        try:
            return float(x)
        except Exception:
            return np.nan

    return s.apply(parse_one)

# ---- parse inputs con máscara ----
def parse_crc_text(x: str) -> float:
    """Convierte '₡ 15.000' o '15000' a 15000.0. Vacío => 0"""
    if x is None:
        return 0.0
    s = str(x).strip()
    if s == "":
        return 0.0
    s = s.replace("₡", "").strip()
    s = s.replace(".", "").replace(",", ".")
    s = "".join(ch for ch in s if (ch.isdigit() or ch in ".-"))
    try:
        return float(s) if s else 0.0
    except Exception:
        return 0.0

def parse_pct_text(x: str) -> float:
    """Convierte '15%' o '15.0' o '' a 15.0. Clamp 0-100."""
    if x is None:
        return 0.0
    s = str(x).strip()
    if s == "":
        return 0.0
    s = s.replace("%", "").strip()
    s = s.replace(",", ".")
    s = "".join(ch for ch in s if (ch.isdigit() or ch in ".-"))
    try:
        v = float(s) if s else 0.0
    except Exception:
        v = 0.0
    return max(0.0, min(100.0, v))

def parse_int_text(x: str) -> int:
    """Convierte '' o '10' a int"""
    if x is None:
        return 0
    s = str(x).strip()
    if s == "":
        return 0
    s = "".join(ch for ch in s if ch.isdigit() or ch == "-")
    try:
        return int(s) if s else 0
    except Exception:
        return 0

def display_crc_input(v: float) -> str:
    """Si es 0 => vacío, si no => ₡ 12.345"""
    try:
        v = float(v)
    except Exception:
        v = 0.0
    return "" if v == 0 else fmt_crc_with_symbol(v)

def display_pct_input(v: float) -> str:
    """Si es 0 => vacío, si no => 15.0%"""
    try:
        v = float(v)
    except Exception:
        v = 0.0
    return "" if v == 0 else f"{v:.1f}%"

def display_int_input(v: int) -> str:
    """Si es 0 => vacío, si no => '10'"""
    try:
        v = int(v)
    except Exception:
        v = 0
    return "" if v == 0 else str(v)

# =============================
# File helpers
# =============================
def find_latest(prefix: str):
    candidates = sorted(
        DATA_DIR.glob(f"{prefix}.*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return candidates[0] if candidates else None

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [strip_accents(str(c)).strip().lower().replace("\ufeff", "") for c in df.columns]
    return df

def pick_col(cols, candidates):
    cols_set = set(cols)
    for c in candidates:
        if c in cols_set:
            return c
    return None

def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        encodings = ["utf-8", "utf-8-sig", "latin1", "iso-8859-1"]
        seps = [",", ";", "\t", "|"]
        last_err = None
        for enc in encodings:
            for sep in seps:
                try:
                    df_try = pd.read_csv(path, encoding=enc, sep=sep)
                    if df_try.shape[1] == 1:
                        continue
                    return df_try
                except Exception as e:
                    last_err = e
        raise last_err
    else:
        return pd.read_excel(path)

# =============================
# UI / CSS
# =============================
st.title("Simulador Bolsa de Descuentos")

st.markdown(
    """
    <style>
      div[data-testid="stTextInput"] label { display:none !important; }
      div[data-testid="stTextInput"] input { height: 2.5rem; }
      .tbl-header {
        font-weight: 600;
        font-size: 0.95rem;
        white-space: nowrap;
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# Sidebar admin
# =============================
with st.sidebar:
    st.header("Acceso Administrador")

    admin_ok = False
    admin_mode = st.toggle("Modo Administrador", value=False)
    if admin_mode:
        if ADMIN_PASSWORD:
            pwd = st.text_input("Password admin", type="password")
            if pwd and pwd.strip() == str(ADMIN_PASSWORD).strip():
                admin_ok = True
                st.success("Acceso concedido")
            elif pwd:
                st.error("Password incorrecta")
        else:
            st.warning("⚠️ No hay password configurada. Acceso libre.")
            admin_ok = True

    st.divider()

DEFAULT_TICKET = 133240.0

if admin_ok:
    TICKET_ADMIN = st.number_input(
        "Monto Matrícula",
        min_value=0.0,
        value=DEFAULT_TICKET,
        step=10.0,
        help="🔒 Solo el administrador puede modificar este valor."
    )
else:
    TICKET_ADMIN = DEFAULT_TICKET

st.markdown(
    f"""
    <div style="font-size: 1.05rem; font-weight: 500;">
        💰 <strong>Monto Matrícula referencia:</strong> ₡ {fmt_crc(TICKET_ADMIN, 0)}
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("ℹ️ Se utiliza el costo de matrícula en donde hay la mayor cantidad de estudiantes.")
ticket = float(TICKET_ADMIN)

# =============================
# Cargar Meta_Bolsa (robusto)
# =============================
meta_path = find_latest("Meta_Bolsa")
if meta_path is None:
    st.warning("No hay archivo Meta_Bolsa en data_store/.")
    st.stop()

try:
    bolsas_raw = read_any(meta_path)
except Exception as e:
    st.error(f"Error leyendo {meta_path.name}: {e}")
    st.stop()

bolsas_raw = normalize_cols(bolsas_raw)

# ✅ Diagnóstico NO visible a usuario final (solo admin)
if admin_ok:
    with st.expander("🔎 Diagnóstico Meta_Bolsa (solo admin)"):
        st.write("Archivo:", meta_path.name)
        st.write(bolsas_raw.columns.tolist())
        st.dataframe(bolsas_raw.head(10))

cols_mb = bolsas_raw.columns.tolist()

col_modalidad = pick_col(cols_mb, ["modalidad", "modalidad_matricula", "modalidadmatricula", "modalidad_matrícula", "modalidad matrícula"])
col_periodo   = pick_col(cols_mb, ["periodo", "anio_periodo", "año_periodo", "ano_periodo", "anio_y_periodo", "ano_y_periodo", "año&periodo", "ano&periodo"])
col_tipo      = pick_col(cols_mb, ["tipo", "tipo_estudiante", "tipos", "tipo_matricula", "tipo matrícula"])
col_monto     = pick_col(cols_mb, ["monto", "bolsa", "bolsa_total", "monto_bolsa", "meta_bolsa", "meta", "presupuesto", "monto_total"])

faltantes = [name for name, col in [
    ("modalidad", col_modalidad),
    ("periodo", col_periodo),
    ("tipo", col_tipo),
    ("monto", col_monto),
] if col is None]

if faltantes:
    st.error("No se pudo cargar la configuración de Meta_Bolsa. Por favor contacte al administrador.")
    if admin_ok:
        st.info(f"Faltan columnas (mapeo automático): {faltantes}")
    st.stop()

bolsas = bolsas_raw.rename(columns={
    col_modalidad: "modalidad",
    col_periodo: "periodo",
    col_tipo: "tipo",
    col_monto: "monto",
}).copy()

bolsas = bolsas.rename(columns={"periodo": "anio_periodo", "tipo": "tipo_estudiante", "monto": "bolsa_total"})
bolsas["bolsa_total"] = parse_money(bolsas["bolsa_total"])

bolsas["modalidad"] = normalize_str(bolsas["modalidad"]).str.lower()
bolsas["tipo_estudiante"] = normalize_str(bolsas["tipo_estudiante"]).str.lower()
bolsas["anio_periodo"] = normalize_str(bolsas["anio_periodo"]).astype(str).str.strip()

bolsas = bolsas.dropna(subset=["anio_periodo", "modalidad", "tipo_estudiante", "bolsa_total"]).copy()
if bolsas.empty:
    st.error("Meta_Bolsa no contiene filas válidas luego de limpiar datos.")
    st.stop()

periodo_actual = sorted(bolsas["anio_periodo"].unique().tolist(), key=sort_key_period)[-1]

# =============================
# Cargar Historico_Mat (para pestaña Recomendaciones)
# =============================
historico_path = find_latest("Historico_Mat")
historico = None

if historico_path is not None:
    try:
        historico = read_any(historico_path)
        historico = normalize_cols(historico)
    except Exception as e:
        if admin_ok:
            st.error(f"Error leyendo Historico_Mat ({historico_path.name}): {e}")
        historico = None

# =============================
# Base y cálculos (simulador)
# =============================
def base_table(mod: str) -> pd.DataFrame:
    df = bolsas[(bolsas["anio_periodo"] == periodo_actual) & (bolsas["modalidad"] == mod)].copy()
    df = df.groupby("tipo_estudiante", as_index=False)["bolsa_total"].sum()
    df["tipo_rank"] = tipo_sort_key(df["tipo_estudiante"])
    df = df.sort_values(["tipo_rank", "tipo_estudiante"]).drop(columns=["tipo_rank"]).reset_index(drop=True)

    return pd.DataFrame({
        "Tipo": df["tipo_estudiante"],
        "Meta Bolsa": df["bolsa_total"].astype(float),
        "Bolsa Disponible": 0.0,
        "Cantidad de personas": 0,
        "% Descuento": 0.0,
    }).reset_index(drop=True)

def compute_results(df_inputs: pd.DataFrame) -> pd.DataFrame:
    t = df_inputs.copy()
    t["Meta Bolsa"] = pd.to_numeric(t["Meta Bolsa"], errors="coerce").fillna(0.0)
    t["Bolsa Disponible"] = pd.to_numeric(t["Bolsa Disponible"], errors="coerce").fillna(0.0)
    t["Cantidad de personas"] = pd.to_numeric(t["Cantidad de personas"], errors="coerce").fillna(0).astype(int)
    t["% Descuento"] = pd.to_numeric(t["% Descuento"], errors="coerce").fillna(0.0)

    t["Monto desc. estimado"] = t["Cantidad de personas"] * ticket * (t["% Descuento"] / 100.0)
    t["Disponible final"] = t["Bolsa Disponible"] - t["Monto desc. estimado"]

    def status(x):
        if pd.isna(x):
            return "—"
        if x < 0:
            return f"🚨 Se pasa por ₡ {fmt_crc(abs(x), 0)}"
        return "✅ Dentro de la bolsa"

    t["Resultado"] = t["Disponible final"].apply(status)
    return t

# =============================
# Normalización automática por fila (sin borrado)
# =============================
def normalize_and_save_tipo(mod_key: str, tipo_id: str):
    inputs_key = f"inputs_{mod_key}"
    if inputs_key not in st.session_state:
        return

    bolsa_key = f"{mod_key}_bolsa_{tipo_id}"
    cant_key  = f"{mod_key}_cant_{tipo_id}"
    pct_key   = f"{mod_key}_pct_{tipo_id}"

    bolsa_txt = st.session_state.get(bolsa_key, "")
    cant_txt  = st.session_state.get(cant_key, "")
    pct_txt   = st.session_state.get(pct_key, "")

    bolsa_num = parse_crc_text(bolsa_txt)
    cant_num  = parse_int_text(cant_txt)
    pct_num   = parse_pct_text(pct_txt)

    st.session_state[bolsa_key] = display_crc_input(bolsa_num)
    st.session_state[cant_key]  = display_int_input(cant_num)
    st.session_state[pct_key]   = display_pct_input(pct_num)

    df = st.session_state[inputs_key].copy()
    mask = df["Tipo"].astype(str).str.strip().str.lower().str.replace(" ", "_") == tipo_id
    if mask.any():
        idx = df.index[mask][0]
        df.loc[idx, "Bolsa Disponible"] = float(bolsa_num)
        df.loc[idx, "Cantidad de personas"] = int(cant_num)
        df.loc[idx, "% Descuento"] = float(pct_num)

    st.session_state[inputs_key] = df

# =============================
# Render modalidad (simulador)
# =============================
def render_modalidad(mod_label: str, mod_key: str):
    st.markdown(f"## {mod_label}")
    st.caption("✍️ Se modifica: Bolsa Disp., Cant., % Desc.")

    inputs_key = f"inputs_{mod_key}"
    if inputs_key not in st.session_state:
        st.session_state[inputs_key] = base_table(mod_key)

    df = st.session_state[inputs_key].copy()

    # ordenar siempre de forma consistente
    df["tipo_rank"] = tipo_sort_key(df["Tipo"])
    df = df.sort_values(["tipo_rank", "Tipo"]).drop(columns=["tipo_rank"]).reset_index(drop=True)
    st.session_state[inputs_key] = df

    res = compute_results(df)

    # Encabezados
    h = st.columns([1.4, 1.6, 2.0, 1.2, 1.3, 1.6, 1.6, 1.6])
    headers = [
        ("Tipo", "Tipo de estudiante: Regular, Retención Inmediata o Regreso"),
        ("Meta Bolsa", "Monto total asignado como meta para este tipo de estudiante"),
        ("✍️ Bolsa Disp.", "Monto de bolsa que deseas usar para otorgar descuentos"),
        ("✍️ Cant.", "Cantidad estimada de estudiantes a los que se les aplicará el descuento"),
        ("✍️ % Desc.", "Porcentaje de descuento que se aplicará por estudiante"),
        ("Monto desc. estimado", "Monto total estimado a descontar según cantidad y porcentaje"),
        ("Disponible final", "Bolsa disponible menos el monto estimado de descuentos"),
        ("Resultado", "Indica si el monto se mantiene dentro de la bolsa asignada"),
    ]
    for col, (label, tooltip) in zip(h, headers):
        col.markdown(
            f"<div class='tbl-header' title='{tooltip}'>{label}</div>",
            unsafe_allow_html=True
        )

    # Filas
    for i in range(len(res)):
        tipo = str(res.loc[i, "Tipo"])
        tipo_id = tipo.strip().lower().replace(" ", "_")

        bolsa_key = f"{mod_key}_bolsa_{tipo_id}"
        cant_key  = f"{mod_key}_cant_{tipo_id}"
        pct_key   = f"{mod_key}_pct_{tipo_id}"

        if bolsa_key not in st.session_state:
            st.session_state[bolsa_key] = display_crc_input(res.loc[i, "Bolsa Disponible"])
        if cant_key not in st.session_state:
            st.session_state[cant_key] = display_int_input(res.loc[i, "Cantidad de personas"])
        if pct_key not in st.session_state:
            st.session_state[pct_key] = display_pct_input(res.loc[i, "% Descuento"])

        c = st.columns([1.4, 1.6, 2.0, 1.2, 1.3, 1.6, 1.6, 1.6])

        c[0].write(display_tipo(tipo))
        c[1].write(fmt_crc_with_symbol(res.loc[i, "Meta Bolsa"]))

        c[2].text_input(
            "",
            key=bolsa_key,
            placeholder="₡ 0",
            on_change=normalize_and_save_tipo,
            args=(mod_key, tipo_id),
        )
        c[3].text_input(
            "",
            key=cant_key,
            placeholder="0",
            on_change=normalize_and_save_tipo,
            args=(mod_key, tipo_id),
        )
        c[4].text_input(
            "",
            key=pct_key,
            placeholder="0.0%",
            on_change=normalize_and_save_tipo,
            args=(mod_key, tipo_id),
        )

        df_now = st.session_state[inputs_key]
        mask = df_now["Tipo"].astype(str).str.strip().str.lower().str.replace(" ", "_") == tipo_id
        if mask.any():
            idx = df_now.index[mask][0]
            bolsa_now = float(df_now.loc[idx, "Bolsa Disponible"])
            cant_now  = int(df_now.loc[idx, "Cantidad de personas"])
            pct_now   = float(df_now.loc[idx, "% Descuento"])
        else:
            bolsa_now, cant_now, pct_now = 0.0, 0, 0.0

        gasto = cant_now * ticket * (pct_now / 100.0)
        disp_final = bolsa_now - gasto

        c[5].write(fmt_crc_with_symbol(gasto))
        c[6].write(fmt_crc_with_symbol(disp_final))
        c[7].write("✅ Dentro de la bolsa" if disp_final >= 0 else f"🚨 Se pasa por ₡ {fmt_crc(abs(disp_final), 0)}")

    # KPIs resumen
    st.write("")
    res2 = compute_results(st.session_state[inputs_key])
    total_personas = int(res2["Cantidad de personas"].sum())
    gasto_total = float(res2["Monto desc. estimado"].sum())
    bolsa_total = float(res2["Bolsa Disponible"].sum())
    gap = bolsa_total - gasto_total

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Personas", f"{total_personas:,}".replace(",", "."))
    k2.metric("Monto desc. estimado", f"₡ {fmt_crc(gasto_total, 0)}")
    k3.metric("Bolsa disponible", f"₡ {fmt_crc(bolsa_total, 0)}")
    k4.metric("Gap (disp - gasto)", f"₡ {fmt_crc(gap, 0)}")

# =============================
# Pestaña Recomendaciones (Historico_Mat)
# =============================
def render_recomendaciones():
    st.markdown("## Recomendaciones basadas en histórico")
    st.caption(
        "Primero elige Año&Periodo (opcional), luego Semana Matrícula. "
        "El rango de fechas se calcula únicamente con registros del periodo seleccionado."
    )

    if historico is None:
        st.info("No hay archivo Historico_Mat en data_store/ o no se pudo leer.")
        return

    cols = historico.columns.tolist()

    def pick(cands):
        return pick_col(cols, cands)

    # Columnas principales
    col_fecha = pick(["fecha", "fecha_matricula", "fechamatricula", "fech_inicio", "fechainicio", "fech"])
    col_modalidad_hist = pick(["modalidad", "modalidad_matricula", "modalidadmatricula"])
    col_tipo_hist = pick(["tipo matricula", "tipo", "tipo_estudiante", "tipos", "tipo_matricula", "canjes.tipo"])

    col_bruto = pick(["monto_bruto_matricula", "monto_bruto", "monto_bruto_total", "monto_bruto_m"])
    col_neto  = pick(["monto_neto_matricula", "monto_neto", "monto_neto_total", "monto_neto_m"])
    col_desc  = pick(["monto_descuento", "monto_desc", "monto_beca", "descuento", "monto_descuento_total"])

    # Filtros extra
    col_semana_mat = pick([
        "semana matricula", "semana_matricula", "semanamatricula",
        "codigo_semana", "codigosemana", "semana_matrícula"
    ])
    col_anio_periodo = pick([
        "año&periodo", "ano&periodo", "anio_periodo", "año_periodo",
        "ano_periodo", "anio&periodo"
    ])

    if col_fecha is None:
        st.warning("El histórico no tiene una columna de fecha reconocible.")
        if admin_ok:
            st.caption("Columnas detectadas (solo admin):")
            st.write(cols)
        return

    if col_semana_mat is None:
        st.warning("El histórico no tiene columna reconocible de Semana Matrícula.")
        if admin_ok:
            st.caption("Columnas detectadas (solo admin):")
            st.write(cols)
        return

    # Preparar histórico
    h = historico.copy()
    h[col_fecha] = pd.to_datetime(h[col_fecha], errors="coerce")
    h = h.dropna(subset=[col_fecha])

    if col_modalidad_hist is not None:
        h[col_modalidad_hist] = normalize_str(h[col_modalidad_hist]).str.lower()

    if col_tipo_hist is not None:
        h[col_tipo_hist] = normalize_str(h[col_tipo_hist]).str.lower()

    # Normalizar semana y año&periodo
    h[col_semana_mat] = normalize_str(h[col_semana_mat]).astype("string").str.strip()
    if col_anio_periodo is not None:
        h[col_anio_periodo] = normalize_str(h[col_anio_periodo]).astype("string").str.strip()

    # Montos
    if col_bruto is not None:
        h[col_bruto] = parse_money(h[col_bruto])
    if col_neto is not None:
        h[col_neto] = parse_money(h[col_neto])
    if col_desc is not None:
        h[col_desc] = parse_money(h[col_desc])

    # Helpers orden semanas: -13 ... 3 (numéricas primero)
    def week_sort_key(x):
        s = str(x).strip()
        if s.lstrip("-").isdigit():
            return (0, int(s))
        return (1, s)

    # =============================
    # UI: filtros SIEMPRE visibles (en el orden que pediste)
    # =============================
    c_ap, c_sem, c_tipo, c_mod = st.columns(4)

    # 1) Año&Periodo
    filtro_ap = "(todos)"
    if col_anio_periodo is not None:
        aps = sorted([a for a in h[col_anio_periodo].dropna().unique().tolist()], key=sort_key_period)
        filtro_ap = c_ap.selectbox("Año&Periodo (opcional)", ["(todos)"] + aps, index=0)
    else:
        c_ap.selectbox("Año&Periodo (opcional)", ["(todos)"], index=0, disabled=True)

    # Data base para semanas y rango: si hay Año&Periodo seleccionado, se limita ahí
    h_base = h.copy()
    if col_anio_periodo is not None and filtro_ap != "(todos)":
        h_base = h_base[h_base[col_anio_periodo] == filtro_ap].copy()

    # 2) Semana Matrícula (ordenada de -13 a 3, etc.)
    semanas_raw = sorted(
        [s for s in h_base[col_semana_mat].dropna().unique().tolist()],
        key=week_sort_key
    )
    semanas = ["(todas)"] + semanas_raw
    semana_sel = c_sem.selectbox("Semana Matrícula", semanas, index=0)

    # 3) Tipo estudiante
    filtro_tipo = "(todos)"
    if col_tipo_hist is not None:
        tipos = sorted([t for t in h[col_tipo_hist].dropna().unique().tolist()])
        opciones = ["(todos)"] + [display_tipo(t) for t in tipos]
        tipo_display = c_tipo.selectbox("Tipo de estudiante (opcional)", opciones, index=0)
        if tipo_display != "(todos)":
            filtro_tipo = tipo_display.strip().lower()
    else:
        c_tipo.selectbox("Tipo de estudiante (opcional)", ["(todos)"], index=0, disabled=True)

    # 4) Modalidad
    filtro_mod = "(todas)"
    if col_modalidad_hist is not None:
        mods = sorted([m for m in h[col_modalidad_hist].dropna().unique().tolist()])
        filtro_mod = c_mod.selectbox("Modalidad (opcional)", ["(todas)"] + mods, index=0)
    else:
        c_mod.selectbox("Modalidad (opcional)", ["(todas)"], index=0, disabled=True)

    # =============================
    # Estado inicial: si semana es (todas), no calculamos (pero sí mostramos UI)
    # =============================
    if semana_sel == "(todas)":
        if filtro_ap != "(todos)":
            st.info(f"ℹ️ Selecciona una semana para calcular recomendaciones dentro de **{filtro_ap}**.")
        else:
            st.info("ℹ️ Selecciona una semana para calcular recomendaciones.")

        k1, k2, k3 = st.columns(3)
        k1.metric("🟢 % Bajo", "—")
        k2.metric("🟡 % Sugerido (promedio)", "—")
        k3.metric("🔴 % Agresivo", "—")
        return

    # =============================
    # Semana -> rango automático, definido por el Año&Periodo seleccionado (h_base)
    # =============================
    tmp_sem = h_base[h_base[col_semana_mat] == semana_sel].copy()
    if tmp_sem.empty:
        st.info("No hay datos para esa semana dentro del periodo seleccionado.")
        k1, k2, k3 = st.columns(3)
        k1.metric("🟢 % Bajo", "—")
        k2.metric("🟡 % Sugerido (promedio)", "—")
        k3.metric("🔴 % Agresivo", "—")
        return

    fecha_ini_hist = tmp_sem[col_fecha].min().normalize()
    fecha_fin_hist = tmp_sem[col_fecha].max().normalize()

    if filtro_ap != "(todos)":
        st.info(f"📅 En **{filtro_ap}**, Semana **{semana_sel}** corresponde a: **{fecha_ini_hist.date()} – {fecha_fin_hist.date()}**")
    else:
        st.info(f"📅 Semana **{semana_sel}** corresponde a: **{fecha_ini_hist.date()} – {fecha_fin_hist.date()}**")

    # =============================
    # Filtrar histórico final: rango de la semana (en ese periodo) + filtros extra
    # =============================
    hist = h_base[(h_base[col_fecha] >= fecha_ini_hist) & (h_base[col_fecha] <= fecha_fin_hist)].copy()

    if col_tipo_hist is not None and filtro_tipo != "(todos)":
        hist = hist[hist[col_tipo_hist] == filtro_tipo]

    if col_modalidad_hist is not None and filtro_mod != "(todas)":
        hist = hist[hist[col_modalidad_hist] == filtro_mod]

    if hist.empty:
        st.info("No hay registros en esa semana con los filtros seleccionados. Prueba otros filtros.")
        k1, k2, k3 = st.columns(3)
        k1.metric("🟢 % Bajo", "—")
        k2.metric("🟡 % Sugerido (promedio)", "—")
        k3.metric("🔴 % Agresivo", "—")
        return

    # =============================
    # Calcular % descuento
    # =============================
    pct = None
    if (
        col_bruto is not None and col_neto is not None
        and hist[col_bruto].notna().any() and hist[col_neto].notna().any()
    ):
        denom = hist[col_bruto].replace(0, np.nan)
        pct = (1 - (hist[col_neto] / denom)) * 100
    elif (
        col_desc is not None and col_bruto is not None
        and hist[col_desc].notna().any() and hist[col_bruto].notna().any()
    ):
        denom = hist[col_bruto].replace(0, np.nan)
        pct = (hist[col_desc] / denom) * 100

    if pct is None:
        st.warning("No fue posible calcular % descuento (faltan columnas bruto/neto o descuento/bruto).")
        k1, k2, k3 = st.columns(3)
        k1.metric("🟢 % Bajo", "—")
        k2.metric("🟡 % Sugerido (promedio)", "—")
        k3.metric("🔴 % Agresivo", "—")
        if admin_ok:
            st.caption("Columnas detectadas (solo admin):")
            st.write(cols)
        return

    pct = pct.replace([np.inf, -np.inf], np.nan).dropna()
    if pct.empty:
        st.warning("No fue posible calcular % descuento (valores nulos o denominadores en 0).")
        k1, k2, k3 = st.columns(3)
        k1.metric("🟢 % Bajo", "—")
        k2.metric("🟡 % Sugerido (promedio)", "—")
        k3.metric("🔴 % Agresivo", "—")
        return

    # =============================
    # Sugerencias
    # =============================
    base = float(pct.mean())
    bajo = float(pct.quantile(0.25))
    agresivo = float(pct.quantile(0.75))

    icon_bajo, desc_bajo = riesgo_icono_color(bajo)
    icon_base, desc_base = riesgo_icono_color(base)
    icon_agresivo, desc_agresivo = riesgo_icono_color(agresivo)

    k1, k2, k3 = st.columns(3)
    k1.metric(f"{icon_bajo} % Bajo", f"{bajo:.1f}%", help=desc_bajo)
    k2.metric(f"{icon_base} % Sugerido (promedio)", f"{base:.1f}%", help=desc_base)
    k3.metric(f"{icon_agresivo} % Agresivo", f"{agresivo:.1f}%", help=desc_agresivo)

    # ✅ Vista técnica SOLO ADMIN
    if admin_ok:
        with st.expander("🔎 Ver muestra del histórico usado (solo admin)"):
            st.dataframe(hist.head(300))


# =============================
# Render (Tabs)
# =============================
tab1, tab2 = st.tabs(["Simulador", "Recomendaciones"])

with tab1:
    st.divider()
    render_modalidad("Presencial", "presencial")
    st.divider()
    render_modalidad("Virtual", "virtual")

with tab2:
    render_recomendaciones()
