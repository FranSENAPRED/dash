# app.py — UAT SENAPRED con campos fijos para ALERTAS
import math
import time
import random
import unicodedata
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

# ========== CAMPOS FIJOS (ajusta si cambian en tus fuentes) ==========
REGION_FIELD_FIRE   = "region"   # campo región en GeoJSON de INCENDIOS (cámbialo si tu feed usa otro nombre)
REGION_FIELD_ALERTS = "reg"      # campo región en GeoJSON de ALERTAS (según tu ejemplo)
ALERT_LEVEL_FIELD   = "tipo"     # 'Aviso' | 'Alerta' | 'Alarma'

# ========== Configuración general ==========
st.set_page_config(page_title="UAT - Incendios + Alertas (Tiempo real)", layout="wide")
st.title("🛰️ UAT — Incendios Forestales + Alertas Meteorológicas.")

# ---------- Sidebar: Incendios ----------
st.sidebar.header("Cobertura de Incendios")
GEOJSON_URL = st.sidebar.text_input(
    "URL GeoJSON (Incendios)",
    value="https://storage.googleapis.com/geodata72-incendios/sidco/incendios_sidco.geojson",
    help="FeatureCollection de incendios (WGS84, lon/lat)."
)
pt_radius = st.sidebar.slider("Tamaño de puntos (px)", 6, 40, 16)
pt_opacity = st.sidebar.slider("Opacidad puntos", 80, 255, 230)
line_width_fire = st.sidebar.slider("Grosor líneas incendios (px)", 1, 8, 2)

st.sidebar.divider()

# ---------- Sidebar: Alertas ----------
st.sidebar.header("Cobertura de Alertas Met.")
ALERTS_URL = st.sidebar.text_input(
    "URL GeoJSON (Alertas)",
    value="https://storage.googleapis.com/geodata-dmc-events-bucket/eventos_AAA_fusionados_ordenados.geojson",
    help="Polígonos de alertas (WGS84, lon/lat)."
)
show_alerts = st.sidebar.toggle("Mostrar alertas", value=True)
fill_opacity_alert = st.sidebar.slider("Opacidad relleno alertas", 30, 220, 100)
line_width_alert = st.sidebar.slider("Grosor borde alertas (px)", 1, 8, 2)

st.sidebar.caption("Escala fija: Aviso = 🟡 Amarillo, Alerta = 🟠 Naranja, Alarma = 🔴 Rojo.")

st.sidebar.divider()
refresh_sec = st.sidebar.slider("Refrescar cada (seg)", 5, 300, 60, step=5)
auto_refresh = st.sidebar.toggle("Auto-refrescar", value=True)
compact_mode = st.sidebar.toggle("Vista ejecutiva (compacta)", value=True)

if st.sidebar.button("🧹 Limpiar caché"):
    st.cache_data.clear()
    st.success("Caché limpiada.")
    st.rerun()

# ========== Utilidades ==========
@st.cache_data(ttl=30, show_spinner=True)
def fetch_geojson(url: str) -> dict:
    headers = {"Accept": "application/geo+json, application/json;q=0.9, */*;q=0.1"}
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    gj = r.json()
    if gj.get("type") != "FeatureCollection":
        raise ValueError("El recurso no es un GeoJSON FeatureCollection.")
    return gj

def features_to_dataframe(gj: dict) -> pd.DataFrame:
    rows = []
    for f in gj.get("features", []):
        props = f.get("properties", {}) or {}
        geom = f.get("geometry", {}) or {}
        gtype = geom.get("type")
        lon = lat = None
        if gtype == "Point":
            try:
                lon, lat = geom["coordinates"][:2]
            except Exception:
                pass
        rows.append({**props, "_geom_type": gtype, "_lon": lon, "_lat": lat})
    return pd.DataFrame(rows)

def detect_datetime_column(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    cands = [c for c in df.columns if any(k in c.lower() for k in
             ["fecha","hora","datetime","timestamp","time","fch","updated","emision","emisión"])]
    for c in cands:
        try:
            pd.to_datetime(df[c], errors="raise")
            return c
        except Exception:
            continue
    return None

def parse_event_time(df: pd.DataFrame, dt_col: str | None):
    if dt_col and dt_col in df.columns:
        s = pd.to_datetime(df[dt_col], errors="coerce", utc=True).dt.tz_convert(None)
        df["_event_time"] = s
    else:
        df["_event_time"] = pd.NaT
    return df

def _walk_coords(coords, acc):
    if not isinstance(coords, (list, tuple)) or len(coords) == 0:
        return
    if isinstance(coords[0], (int, float)) and len(coords) >= 2:
        lon, lat = coords[0], coords[1]
        if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
            acc.append((lon, lat))
        return
    for c in coords:
        _walk_coords(c, acc)

def geojson_bounds(gj: dict):
    pts = []
    for f in gj.get("features", []):
        geom = f.get("geometry") or {}
        _walk_coords(geom.get("coordinates"), pts)
    if not pts:
        return None
    lons, lats = zip(*pts)
    return (min(lons), min(lats), max(lons), max(lats))

def _zoom_from_bounds(min_lon, min_lat, max_lon, max_lat):
    dx = max(1e-6, abs(max_lon - min_lon))
    dy = max(1e-6, abs(max_lat - min_lat))
    extent = max(dx, dy)
    z = 8 - math.log2(extent)
    return max(2, min(12, z))

def _merge_bounds(*bounds_list):
    valid = [b for b in bounds_list if b]
    if not valid:
        return None
    min_lon = min(b[0] for b in valid)
    min_lat = min(b[1] for b in valid)
    max_lon = max(b[2] for b in valid)
    max_lat = max(b[3] for b in valid)
    return (min_lon, min_lat, max_lon, max_lat)

# ===== Colores fijos para alertas (según ALERT_LEVEL_FIELD) =====
FIXED_COLORS = {
    "aviso":  [255, 215,   0],  # amarillo
    "alerta": [255, 165,   0],  # naranja
    "alarma": [255,   0,   0],  # rojo
}
def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
def _norm_label(s: str) -> str:
    s2 = _strip_accents((s or "")).lower()
    for k in FIXED_COLORS:
        if k in s2:
            return k
    return s2
def _pastel(seed: str):
    random.seed(hash(seed) & 0xFFFF)
    return [120+random.randint(0,100), 120+random.randint(0,100), 120+random.randint(0,100)]

def colorize_alerts_fixed_scale(gj_alerts: dict, alpha: int):
    """
    Inserta en properties:
      - _color       -> RGBA por severidad (aviso/alerta/alarma) o pastel si distinto
      - _line_width  -> borde más grueso para 'Alarma'
      - _label       -> texto rico para tooltip
    """
    if not gj_alerts or not gj_alerts.get("features"):
        return gj_alerts, {}

    legends = {}
    for f in gj_alerts["features"]:
        props = f.get("properties") or {}
        nivel = props.get(ALERT_LEVEL_FIELD, "Sin dato")
        norm  = _norm_label(str(nivel))
        rgb   = FIXED_COLORS.get(norm, _pastel(norm or "sin_dato"))
        props["_color"] = [*rgb, alpha]
        props["_line_width"] = 4 if norm == "alarma" else 2

        # Construimos un label corto y útil para el tooltip
        titulo   = props.get("titulo", "")
        fenomeno = props.get("fenomeno", "")
        emision  = props.get("emision", "")
        desde    = props.get("desde", "")
        hasta    = props.get("hasta", "")
        props["_label"] = f"<b>{nivel}</b> — {fenomeno}<br><i>{titulo}</i><br>Emisión: {emision}<br>Vigencia: {desde} → {hasta}"

        f["properties"] = props
        legends[str(nivel)] = rgb
    return gj_alerts, legends

# ========== Mapa ==========
def make_deck_multilayer(gj_fire: dict, df_points: pd.DataFrame,
                         gj_alerts_colored: dict | None,
                         pt_radius_px: int, pt_alpha: int,
                         line_w_fire: int, line_w_alert: int):
    layers = []

    # Incendios: polígonos/líneas
    layer_fire = pdk.Layer(
        "GeoJsonLayer", gj_fire,
        stroked=True, filled=True, extruded=False,
        get_line_color=[255,255,255,230],
        get_fill_color=[255,80,80,90],
        lineWidthMinPixels=line_w_fire,
        pickable=True, auto_highlight=True,
        pointRadiusMinPixels=max(4, pt_radius_px - 2),
        pointRadiusMaxPixels=60, get_point_radius=pt_radius_px,
    )
    layers.append(layer_fire)

    # Incendios: puntos
    if not df_points.empty:
        scatter = pdk.Layer(
            "ScatterplotLayer", data=df_points,
            get_position="[_lon, _lat]", pickable=True,
            stroked=True, filled=True, radius_scale=1,
            radius_min_pixels=max(4, pt_radius_px),
            radius_max_pixels=80, get_radius=pt_radius_px,
            line_width_min_pixels=1.5,
            get_fill_color=[255,80,80,pt_alpha],
            get_line_color=[255,255,255,255]
        )
        layers.append(scatter)

    # Alertas (coloreadas)
    if gj_alerts_colored:
        alerts_layer = pdk.Layer(
            "GeoJsonLayer", gj_alerts_colored,
            stroked=True, filled=True, extruded=False,
            get_line_color=[255,255,255,240],
            get_fill_color="_color",
            get_line_width="_line_width",
            lineWidthMinPixels=line_w_alert,
            pickable=True, auto_highlight=True,
        )
        layers.append(alerts_layer)

    # Autofit con ambas coberturas
    b_fire = geojson_bounds(gj_fire)
    b_alert = geojson_bounds(gj_alerts_colored) if gj_alerts_colored else None
    bounds = _merge_bounds(b_fire, b_alert)

    if bounds:
        min_lon, min_lat, max_lon, max_lat = bounds
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        zoom = _zoom_from_bounds(min_lon, min_lat, max_lon, max_lat)
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom)
    else:
        view_state = pdk.ViewState(latitude=-35.6751, longitude=-71.5430, zoom=4.5)

    tooltip = {
        "html": "<b>{id}</b><br>{region} {comuna} {estado}<br>{_label}",
        "style": {"backgroundColor": "rgba(0,0,0,0.85)", "color": "white"}
    }
    return pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip)

# ========== Carga de datos ==========
status = st.empty()

# Incendios
try:
    gj_fire = fetch_geojson(GEOJSON_URL)
    df_fire = features_to_dataframe(gj_fire)
    dt_fire = detect_datetime_column(df_fire)
    df_fire = parse_event_time(df_fire, dt_fire)
    feat_fire = len(gj_fire.get("features", []))
    status.success(f"✅ Incendios: {feat_fire} features")
except Exception as e:
    status.error(f"❌ Error Incendios: {e}")
    st.stop()

# Alertas
gj_alerts = None
legend_map = {}
df_alerts = pd.DataFrame()

if show_alerts:
    try:
        gj_alerts = fetch_geojson(ALERTS_URL)
        gj_alerts, legend_map = colorize_alerts_fixed_scale(gj_alerts, fill_opacity_alert)
        df_alerts = features_to_dataframe(gj_alerts)
        st.success(f"✅ Alertas: {len(gj_alerts.get('features', []))} features · Color por: {ALERT_LEVEL_FIELD} (escala fija)")
    except Exception as e:
        st.warning(f"⚠️ No se cargaron alertas: {e}")
        gj_alerts = None

# ========== KPIs ejecutivos ==========
k1, k2, k3, k4, k5 = st.columns(5)
now = datetime.utcnow().replace(tzinfo=None)
last24 = df_fire[df_fire["_event_time"] >= (now - timedelta(hours=24))] if "_event_time" in df_fire else pd.DataFrame()

k1.metric("🔥 Incendios (total)", len(df_fire))
k2.metric("🔥 Últimas 24h", len(last24) if not last24.empty else 0)

# Conteo de regiones por severidad (usando REGIÓN_FIELD_ALERTS y ALERT_LEVEL_FIELD)
sev_counts = {"aviso":0,"alerta":0,"alarma":0}
if not df_alerts.empty and REGION_FIELD_ALERTS in df_alerts.columns and ALERT_LEVEL_FIELD in df_alerts.columns:
    grp = df_alerts.groupby(REGION_FIELD_ALERTS)[ALERT_LEVEL_FIELD] \
                   .agg(lambda s: set(_norm_label(str(x)) for x in s if pd.notna(x)))
    for levels in grp:
        for l in ["aviso","alerta","alarma"]:
            if l in levels: sev_counts[l] += 1

k3.metric("🟡 Regiones con Aviso",  sev_counts["aviso"])
k4.metric("🟠 Regiones con Alerta", sev_counts["alerta"])
k5.metric("🔴 Regiones con Alarma", sev_counts["alarma"])

# Banner crítico
if sev_counts["alarma"] > 0:
    st.error(f"🔴 **ALERTA MÁXIMA**: {sev_counts['alarma']} región(es) con **Alarma** activa.", icon="🚨")
elif sev_counts["alerta"] > 0:
    st.warning(f"🟠 {sev_counts['alerta']} región(es) con **Alerta**.", icon="⚠️")
elif sev_counts["aviso"] > 0:
    st.info(f"🟡 {sev_counts['aviso']} región(es) con **Aviso**.", icon="ℹ️")

# ========== Mapa ==========
st.subheader("🗺️ Mapa (autofit)")
df_points = df_fire.dropna(subset=["_lon", "_lat"]) if {"_lon","_lat"}.issubset(df_fire.columns) else pd.DataFrame()
deck = make_deck_multilayer(
    gj_fire=gj_fire, df_points=df_points,
    gj_alerts_colored=gj_alerts if show_alerts else None,
    pt_radius_px=pt_radius, pt_alpha=pt_opacity,
    line_w_fire=line_width_fire, line_w_alert=line_width_alert
)
st.pydeck_chart(deck, use_container_width=True)

# ========== Matriz de decisión por Región ==========
st.subheader("🧭 Matriz de decisión por Región")

# Conjunto de regiones: une lo que aparezca en incendios (REGION_FIELD_FIRE) y alertas (REGION_FIELD_ALERTS)
regions = set()
if REGION_FIELD_FIRE in df_fire.columns:
    regions |= set(df_fire[REGION_FIELD_FIRE].dropna().astype(str).unique())
if not df_alerts.empty and REGION_FIELD_ALERTS in df_alerts.columns:
    regions |= set(df_alerts[REGION_FIELD_ALERTS].dropna().astype(str).unique())
regions = sorted(regions)

# Filtro rápido por región
selected_regs = st.multiselect("Filtrar regiones", regions, placeholder="Escribe para buscar…")
df_fire_f   = df_fire.copy()
df_alerts_f = df_alerts.copy()
if selected_regs:
    if REGION_FIELD_FIRE in df_fire_f.columns:
        df_fire_f = df_fire_f[df_fire_f[REGION_FIELD_FIRE].astype(str).isin(selected_regs)]
    if REGION_FIELD_ALERTS in df_alerts_f.columns:
        df_alerts_f = df_alerts_f[df_alerts_f[REGION_FIELD_ALERTS].astype(str).isin(selected_regs)]
    regions = selected_regs

rows = []
for r in regions:
    fires = len(df_fire_f[df_fire_f.get(REGION_FIELD_FIRE, "") == r]) if REGION_FIELD_FIRE in df_fire_f.columns else 0
    avis = aler = alar = False
    if not df_alerts_f.empty and REGION_FIELD_ALERTS in df_alerts_f.columns and ALERT_LEVEL_FIELD in df_alerts_f.columns:
        levels = set(_norm_label(x) for x in df_alerts_f.loc[df_alerts_f[REGION_FIELD_ALERTS]==r, ALERT_LEVEL_FIELD].astype(str))
        avis = "aviso"  in levels
        aler = "alerta" in levels
        alar = "alarma" in levels
    rows.append({"Región": r, "🟡 Aviso": avis, "🟠 Alerta": aler, "🔴 Alarma": alar, "🔥 Incendios": fires})

matrix_df = pd.DataFrame(rows)

# Filtros de priorización
c1, c2, c3, c4 = st.columns(4)
fltr_alarm  = c1.toggle("Solo 🔴 Alarma", value=False)
fltr_alerta = c2.toggle("Solo 🟠 Alerta", value=False)
fltr_aviso  = c3.toggle("Solo 🟡 Aviso", value=False)
fltr_fires  = c4.toggle("Solo con 🔥 Incendios", value=False)

fdf = matrix_df.copy()
if fltr_alarm:  fdf = fdf[fdf["🔴 Alarma"]]
if fltr_alerta: fdf = fdf[fdf["🟠 Alerta"]]
if fltr_aviso:  fdf = fdf[fdf["🟡 Aviso"]]
if fltr_fires:  fdf = fdf[fdf["🔥 Incendios"] > 0]

# Orden: Alarma > Alerta > Aviso > Incendios
fdf = fdf.sort_values(by=["🔴 Alarma","🟠 Alerta","🟡 Aviso","🔥 Incendios","Región"],
                      ascending=[False, False, False, False, True])

# Estilo semáforo para incendios
def _bg(v): return "background-color: rgba(255,80,80,0.25)" if v>0 else ""
styled = fdf.style.apply(lambda s: [ _bg(v) if s.name=="🔥 Incendios" else "" for v in s ], axis=0)

st.dataframe(
    styled if not compact_mode else fdf,  # en vista ejecutiva no aplicamos .style (más liviano)
    use_container_width=True, height=360,
    column_config={
        "🟡 Aviso":  st.column_config.CheckboxColumn(width="small"),
        "🟠 Alerta": st.column_config.CheckboxColumn(width="small"),
        "🔴 Alarma": st.column_config.CheckboxColumn(width="small"),
        "🔥 Incendios": st.column_config.NumberColumn(format="%d"),
    }
)

st.download_button("⬇️ Descargar matriz por región (CSV)",
                   fdf.to_csv(index=False).encode("utf-8"),
                   file_name="matriz_regiones.csv", mime="text/csv")

# ========== Prioridades operativas ==========
st.subheader("🎯 Prioridades operativas")
colA, colB = st.columns(2)

with colA:
    st.markdown("**Top 10 — Regiones por 🔥 Incendios**")
    top_fires = fdf.sort_values("🔥 Incendios", ascending=False).head(10)
    chart_fires = alt.Chart(top_fires).mark_bar().encode(
        x=alt.X("🔥 Incendios:Q", title="Incendios"),
        y=alt.Y("Región:N", sort="-x"),
        tooltip=["Región","🔥 Incendios"]
    ).properties(height=280)
    st.altair_chart(chart_fires, use_container_width=True)

with colB:
    st.markdown("**Regiones con Aviso/Alerta/Alarma**")
    sev_counts_df = pd.DataFrame({
        "Severidad": ["Aviso","Alerta","Alarma"],
        "Regiones": [fdf["🟡 Aviso"].sum(), fdf["🟠 Alerta"].sum(), fdf["🔴 Alarma"].sum()]
    })
    chart_sev = alt.Chart(sev_counts_df).mark_bar().encode(
        x=alt.X("Regiones:Q"),
        y=alt.Y("Severidad:N", sort=["Alarma","Alerta","Aviso"]),
        tooltip=["Severidad","Regiones"]
    ).properties(height=280)
    st.altair_chart(chart_sev, use_container_width=True)

# Sugerencias simples
st.subheader("📝 Sugerencias de acción (reglas simples)")
acciones = []
for _, row in fdf.iterrows():
    if row["🔴 Alarma"]:
        acciones.append(f"🔴 {row['Región']}: activar protocolo SAE y coordinación inmediata con DMC/CONAF.")
    elif row["🟠 Alerta"] and row["🔥 Incendios"]>0:
        acciones.append(f"🟠 {row['Región']}: reforzar monitoreo y preposicionar recursos (hay {row['🔥 Incendios']} incendios).")
    elif row["🟡 Aviso"] and row["🔥 Incendios"]>0:
        acciones.append(f"🟡 {row['Región']}: vigilancia focalizada; validar condiciones locales.")
if acciones:
    st.markdown("\n".join(f"- {a}" for a in acciones))
else:
    st.markdown("_Sin sugerencias críticas por ahora._")

# ========== Tablas base (ocultas en modo compacto) ==========
if not compact_mode:
    with st.expander("📄 Ver tablas base"):
        st.markdown("**Incendios — atributos**")
        st.dataframe(df_fire, use_container_width=True, height=260)
        if not df_alerts.empty:
            st.markdown("**Alertas — atributos**")
            st.dataframe(df_alerts, use_container_width=True, height=260)

# ========== Auto-refresh ==========
if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
