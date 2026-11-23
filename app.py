from __future__ import annotations

import json
from datetime import date, timedelta
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from satid.config import APP_NAME, APP_VERSION, DEFAULT_THRESHOLDS, FEATURE_COLUMNS, TIMEZONE
from satid.geo import make_map
from satid.providers import (
    ProviderError, geocode_place, get_elevation, get_forecast,
    get_historical, get_osm_hydrology, get_river_discharge,
)
from satid.risk import (
    LEVEL_COLORS, assess_risk, build_features, summarize_forecast, train_random_forest,
)
from satid.storage import (
    add_observation, add_source, create_zone, delete_zone, get_zone,
    import_geojson, init_db, list_observations, list_sources, list_zones,
    update_zone, zones_geojson,
)

st.set_page_config(page_title="SATID | Alerta temprana", page_icon="🌧️", layout="wide")
init_db()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_forecast(latitude: float, longitude: float):
    return get_forecast(latitude, longitude)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_geocode(query: str):
    return geocode_place(query)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_elevation(latitude: float, longitude: float):
    return get_elevation(latitude, longitude)


@st.cache_data(ttl=21600, show_spinner=False)
def cached_historical(latitude: float, longitude: float, start: str, end: str, model: str):
    return get_historical(latitude, longitude, start, end, model)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_hydrology(latitude: float, longitude: float, radius: int):
    return get_osm_hydrology(latitude, longitude, radius)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_discharge(latitude: float, longitude: float):
    return get_river_discharge(latitude, longitude)


def fmt(value, unit="", decimals=1):
    if value is None or pd.isna(value):
        return "Sin dato"
    return f"{float(value):,.{decimals}f}{unit}".replace(",", "X").replace(".", ",").replace("X", ".")


def thresholds_from_ui() -> dict[str, float]:
    with st.expander("⚙️ Umbrales experimentales de precipitación", expanded=False):
        st.caption("Son valores iniciales para probar la interfaz. No son umbrales oficiales ni han sido calibrados con registros del municipio.")
        c1, c2, c3 = st.columns(3)
        with c1:
            y24 = st.number_input("Amarillo · últimas 24 h (mm)", min_value=1.0, value=float(DEFAULT_THRESHOLDS["yellow_24h"]), step=1.0, key="thr_y24")
            r24 = st.number_input("Rojo · últimas 24 h (mm)", min_value=1.0, value=float(DEFAULT_THRESHOLDS["red_24h"]), step=1.0, key="thr_r24")
        with c2:
            y72 = st.number_input("Amarillo · últimas 72 h (mm)", min_value=1.0, value=float(DEFAULT_THRESHOLDS["yellow_72h"]), step=1.0, key="thr_y72")
            r72 = st.number_input("Rojo · últimas 72 h (mm)", min_value=1.0, value=float(DEFAULT_THRESHOLDS["red_72h"]), step=1.0, key="thr_r72")
        with c3:
            yf = st.number_input("Amarillo · pronóstico 24 h (mm)", min_value=1.0, value=float(DEFAULT_THRESHOLDS["yellow_forecast_24h"]), step=1.0, key="thr_yf")
            rf = st.number_input("Rojo · pronóstico 24 h (mm)", min_value=1.0, value=float(DEFAULT_THRESHOLDS["red_forecast_24h"]), step=1.0, key="thr_rf")
        if r24 <= y24 or r72 <= y72 or rf <= yf:
            st.error("Cada umbral Rojo debe ser mayor que su umbral Amarillo.")
        return {
            "yellow_24h": y24, "red_24h": r24,
            "yellow_72h": y72, "red_72h": r72,
            "yellow_forecast_24h": yf, "red_forecast_24h": rf,
        }


def download_button(label: str, data: bytes | str, file_name: str, mime: str):
    st.download_button(label, data=data, file_name=file_name, mime=mime, use_container_width=True)


def source_links(zone: dict):
    query = quote_plus(f"{zone['name']} Boyacá inundación precipitación")
    ideam_query = quote_plus(f"site:ideam.gov.co {zone['name']} precipitación inundación")
    corp_query = quote_plus(f"site:corpoboyaca.gov.co {zone['name']} Lago de Tota riesgo hídrico")
    ungrd_query = quote_plus(f"site:gestiondelriesgo.gov.co {zone['name']} inundación")
    col_query = quote_plus(f"{zone['name']} datos hidrometeorológicos")
    st.markdown("**Búsqueda documental externa**")
    st.caption("Los enlaces abren búsquedas para revisión manual. SATID no copia automáticamente artículos o noticias sin verificar su procedencia.")
    cols = st.columns(3)
    links = [
        ("Buscar información web", f"https://www.google.com/search?q={query}"),
        ("IDEAM", f"https://www.google.com/search?q={ideam_query}"),
        ("Corpoboyacá", f"https://www.google.com/search?q={corp_query}"),
        ("UNGRD", f"https://www.google.com/search?q={ungrd_query}"),
        ("Datos abiertos Colombia", f"https://www.google.com/search?q={col_query}+site%3Adatos.gov.co"),
        ("Copernicus Browser (Sentinel)", "https://dataspace.copernicus.eu/browser/"),
    ]
    for i, (label, url) in enumerate(links):
        with cols[i % 3]:
            st.link_button(label, url, use_container_width=True)


zones = list_zones()
if not zones:
    st.error("No hay zonas almacenadas. Reinicia la aplicación o crea una zona desde el menú lateral.")
    st.stop()

st.sidebar.title("🌧️ SATID")
st.sidebar.caption(f"Prototipo académico · v{APP_VERSION}")
zone_labels = {z["id"]: f"{z['name']} ({z['latitude']:.4f}, {z['longitude']:.4f})" for z in zones}
if "selected_zone_id" not in st.session_state or st.session_state.selected_zone_id not in zone_labels:
    st.session_state.selected_zone_id = zones[0]["id"]
zone_ids = list(zone_labels)
selected_index = zone_ids.index(st.session_state.selected_zone_id) if st.session_state.selected_zone_id in zone_ids else 0
selected_id = st.sidebar.selectbox(
    "Zona activa", options=zone_ids, index=selected_index,
    format_func=lambda k: zone_labels[k],
)
st.session_state.selected_zone_id = selected_id
zone = get_zone(selected_id)

with st.sidebar.expander("＋ Añadir zona", expanded=False):
    method = st.radio("Método", ["Buscar lugar", "Ingresar coordenadas"], key="add_zone_method")
    if method == "Buscar lugar":
        place_query = st.text_input("Lugar o dirección", placeholder="Ej. Playa Blanca, Aquitania, Boyacá", key="place_query")
        if st.button("Buscar ubicación", key="geocode_btn", use_container_width=True):
            try:
                matches = cached_geocode(place_query)
                st.session_state.geocode_matches = matches
                if not matches:
                    st.warning("No se encontraron coincidencias. Intenta con un nombre más específico.")
            except ProviderError as exc:
                st.error(str(exc))
        matches = st.session_state.get("geocode_matches", [])
        if matches:
            choice = st.selectbox("Coincidencias", options=range(len(matches)), format_func=lambda i: matches[i]["name"], key="geocode_choice")
            zone_name = st.text_input("Nombre corto para la zona", value=matches[choice]["name"].split(",")[0], key="geocode_name")
            if st.button("Guardar zona seleccionada", key="save_geocode", use_container_width=True):
                m = matches[choice]
                try:
                    new_id = create_zone(zone_name, m["latitude"], m["longitude"], description=m["name"])
                    st.session_state.selected_zone_id = new_id
                    st.session_state.geocode_matches = []
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    else:
        with st.form("manual_zone_form"):
            name = st.text_input("Nombre de la zona")
            lat = st.number_input("Latitud", min_value=-90.0, max_value=90.0, value=5.5190, format="%.6f")
            lon = st.number_input("Longitud", min_value=-180.0, max_value=180.0, value=-72.8840, format="%.6f")
            elevation = st.number_input("Altitud (m, opcional)", min_value=0.0, max_value=9000.0, value=0.0, step=10.0)
            description = st.text_input("Descripción")
            submitted = st.form_submit_button("Guardar coordenadas", use_container_width=True)
        if submitted:
            try:
                new_id = create_zone(name, lat, lon, elevation if elevation > 0 else None, description)
                st.session_state.selected_zone_id = new_id
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

st.sidebar.divider()
page = st.sidebar.radio(
    "Navegación",
    ["Panel de alerta", "Mapa y zonas", "Historial y registros", "Laboratorio IA", "Fuentes y configuración"],
)
st.sidebar.caption("Datos y alertas experimentales. No reemplaza a IDEAM, UNGRD ni organismos de socorro.")

st.title("Sistema de Alerta Temprana de Inundaciones y Desastres")
st.caption("SATID · Ecosistema de alta montaña de Aquitania y otras zonas configurables")

if page == "Panel de alerta":
    st.subheader(zone["name"])
    st.write(f"Coordenadas: `{zone['latitude']:.5f}, {zone['longitude']:.5f}` · {zone.get('description') or 'Sin descripción'}")
    thresholds = thresholds_from_ui()
    if any(thresholds[k] >= thresholds[r] for k, r in [("yellow_24h", "red_24h"), ("yellow_72h", "red_72h"), ("yellow_forecast_24h", "red_forecast_24h")]):
        st.warning("Corrige los umbrales antes de calcular el nivel de alerta.")
    else:
        with st.spinner("Consultando precipitación, humedad del suelo y pronóstico..."):
            try:
                raw_weather = cached_forecast(zone["latitude"], zone["longitude"])
                metrics = summarize_forecast(raw_weather)
                if zone.get("elevation_m") is None and metrics.get("elevation_m") is None:
                    try:
                        metrics["elevation_m"] = cached_elevation(zone["latitude"], zone["longitude"])
                    except ProviderError:
                        metrics["elevation_m"] = None
                risk = assess_risk(metrics, thresholds)
                st.session_state[f"metrics_{selected_id}"] = metrics
                st.session_state[f"risk_{selected_id}"] = risk
                st.session_state[f"weather_raw_{selected_id}"] = raw_weather
                weather_error = None
            except (ProviderError, ValueError, KeyError) as exc:
                weather_error = str(exc)
                metrics = st.session_state.get(f"metrics_{selected_id}")
                risk = st.session_state.get(f"risk_{selected_id}")

        if weather_error:
            st.error(f"No se pudo actualizar la consulta: {weather_error}")
            st.info("Se muestran datos de esta sesión solo si existía una consulta previa. Intenta nuevamente cuando el servicio esté disponible.")
        if metrics and risk:
            model = st.session_state.get("rf_model")
            rf_used = False
            if model is not None and risk.get("level") != "Sin datos suficientes":
                try:
                    features = build_features(metrics, zone.get("elevation_m"))
                    prediction = str(model.predict(features)[0])
                    probabilities = model.predict_proba(features)[0]
                    probability = float(np.max(probabilities))
                    risk = {**risk, "level": prediction, "color": LEVEL_COLORS.get(prediction, "#2563eb"),
                            "index": None, "method": "Random Forest entrenado por el usuario",
                            "reasons": [f"Clase estimada por Random Forest: {prediction}.",
                                        f"Probabilidad interna de la clase ganadora: {probability:.1%} (no calibrada)."]}
                    rf_used = True
                except Exception as exc:
                    st.warning(f"No se pudo aplicar el modelo RF; se conserva la regla: {exc}")

            col_alert, col_score, col_coverage = st.columns([1.35, 1, 1])
            with col_alert:
                st.markdown(f"<div style='background:{risk['color']};padding:18px;border-radius:12px;color:white'>"
                            f"<div style='font-size:14px;opacity:.9'>NIVEL EXPERIMENTAL</div>"
                            f"<div style='font-size:32px;font-weight:800'>{risk['level'].upper()}</div>"
                            f"<div>{'Predicción de modelo RF' if rf_used else 'Motor basado en reglas'}</div></div>", unsafe_allow_html=True)
            with col_score:
                st.metric("Índice heurístico", "Modelo RF" if rf_used else f"{risk['index']}/100")
            with col_coverage:
                st.metric("Cobertura de variables", f"{risk['coverage']}%")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precipitación 24 h", fmt(metrics["rain_24h_mm"], " mm"))
            c2.metric("Precipitación 72 h", fmt(metrics["rain_72h_mm"], " mm"))
            c3.metric("Pronóstico próximo 24 h", fmt(metrics["forecast_24h_mm"], " mm"))
            c4.metric("Humedad suelo 0–7 cm", fmt(metrics["soil_moisture"], " m³/m³", 3))
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Temperatura", fmt(metrics["temperature_c"], " °C"))
            c6.metric("Humedad relativa", fmt(metrics["humidity_pct"], "%", 0))
            c7.metric("Viento", fmt(metrics["wind_kmh"], " km/h"))
            c8.metric("Altitud", fmt(zone.get("elevation_m") or metrics.get("elevation_m"), " m", 0))

            st.markdown("**Factores que explican el nivel**")
            for reason in risk["reasons"]:
                st.write(f"• {reason}")
            if risk.get("limitations"):
                st.warning("Limitaciones: " + " ".join(risk["limitations"]))
            st.caption(f"Consulta: {metrics['time']} ({TIMEZONE}). Datos meteorológicos: Open-Meteo. Método: {risk['method']}.")

            hourly_df = metrics["hourly_df"].copy()
            if not hourly_df.empty:
                fig = px.bar(hourly_df, x="time", y="precipitation_mm", title="Precipitación horaria observada y pronosticada (mm)",
                             labels={"time": "Fecha y hora", "precipitation_mm": "Precipitación (mm)"})
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Mapa de las zonas registradas")
            fmap = make_map(list_zones(), selected_id, risk["level"])
            st_folium(fmap, height=430, use_container_width=True, returned_objects=[])

            with st.expander("Consultar caudal fluvial modelado (GloFAS)", expanded=False):
                st.caption("Indicador complementario: el servicio selecciona una celda de río de aproximadamente 5 km. Puede no corresponder al cauce de interés; no se incorpora automáticamente al nivel de alerta.")
                if st.button("Consultar GloFAS para esta coordenada", key="get_glofas"):
                    try:
                        discharge = cached_discharge(zone["latitude"], zone["longitude"])
                        daily = discharge.get("daily", {})
                        dfq = pd.DataFrame({"fecha": daily.get("time", []), "caudal_m3s": daily.get("river_discharge", [])})
                        if not dfq.empty:
                            st.dataframe(dfq, use_container_width=True, hide_index=True)
                            figq = px.line(dfq, x="fecha", y="caudal_m3s", title="Caudal diario estimado por GloFAS")
                            st.plotly_chart(figq, use_container_width=True)
                            st.caption(f"Celda devuelta por el modelo: {discharge.get('latitude')}, {discharge.get('longitude')}.")
                        else:
                            st.info("GloFAS no devolvió datos de caudal para esta consulta.")
                    except ProviderError as exc:
                        st.error(str(exc))
        else:
            st.warning("Todavía no hay datos meteorológicos disponibles para esta zona.")

elif page == "Mapa y zonas":
    st.subheader("Mapa interactivo y administración de zonas")
    st.write("Cada marcador representa una zona puntual configurada por el usuario. Puedes buscar lugares, registrar coordenadas o hacer clic sobre el mapa para capturar una ubicación.")
    add_hydro = st.checkbox("Consultar y mostrar cauces/cuerpos de agua de OpenStreetMap cercanos (hasta 5 km)", value=False)
    hydro = None
    if add_hydro:
        try:
            with st.spinner("Consultando elementos cartográficos comunitarios..."):
                hydro = cached_hydrology(zone["latitude"], zone["longitude"], 5000)
            st.caption(f"Elementos hídricos devueltos por OpenStreetMap: {len(hydro.get('features', []))}.")
        except ProviderError as exc:
            st.warning(str(exc))
    saved_risk = st.session_state.get(f"risk_{selected_id}", {})
    fmap = make_map(list_zones(), selected_id, saved_risk.get("level"), hydro)
    click_data = st_folium(fmap, height=570, use_container_width=True, returned_objects=["last_clicked"])
    clicked = (click_data or {}).get("last_clicked")
    if clicked:
        st.success(f"Coordenada seleccionada: {clicked['lat']:.6f}, {clicked['lng']:.6f}")
        with st.form("save_map_clicked_zone"):
            clicked_name = st.text_input("Nombre de la nueva zona", value="Nueva zona")
            clicked_description = st.text_input("Descripción", value="Ubicación seleccionada en el mapa")
            save_clicked = st.form_submit_button("Guardar punto como nueva zona")
        if save_clicked:
            try:
                new_id = create_zone(clicked_name, clicked["lat"], clicked["lng"], description=clicked_description)
                st.session_state.selected_zone_id = new_id
                st.success("Zona guardada.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.markdown("**Zonas guardadas**")
    zones_df = pd.DataFrame(list_zones())
    st.dataframe(zones_df[["id", "name", "latitude", "longitude", "elevation_m", "description"]],
                 use_container_width=True, hide_index=True)
    with st.expander("Editar o eliminar la zona activa"):
        with st.form("edit_zone_form"):
            edit_name = st.text_input("Nombre", value=zone["name"])
            edit_lat = st.number_input("Latitud", value=float(zone["latitude"]), format="%.6f")
            edit_lon = st.number_input("Longitud", value=float(zone["longitude"]), format="%.6f")
            edit_elev = st.number_input("Altitud (m; 0 = sin dato)", min_value=0.0, value=float(zone["elevation_m"] or 0), step=10.0)
            edit_desc = st.text_area("Descripción", value=zone.get("description") or "")
            save_edit = st.form_submit_button("Guardar cambios")
        if save_edit:
            try:
                update_zone(selected_id, edit_name, edit_lat, edit_lon, edit_elev if edit_elev > 0 else None, edit_desc)
                st.success("Zona actualizada.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if len(zones) > 1 and st.button("Eliminar zona activa", type="secondary"):
            delete_zone(selected_id)
            st.session_state.selected_zone_id = list_zones()[0]["id"]
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        csv_zones = zones_df.to_csv(index=False).encode("utf-8-sig")
        download_button("Descargar zonas CSV", csv_zones, "satid_zonas.csv", "text/csv")
    with c2:
        geojson_bytes = json.dumps(zones_geojson(), ensure_ascii=False, indent=2).encode("utf-8")
        download_button("Descargar zonas GeoJSON", geojson_bytes, "satid_zonas.geojson", "application/geo+json")
    upload_geo = st.file_uploader("Importar otras zonas (GeoJSON FeatureCollection de puntos)", type=["geojson", "json"], key="zones_geojson_upload")
    if upload_geo is not None and st.button("Importar GeoJSON", key="import_geojson_btn"):
        try:
            count = import_geojson(json.loads(upload_geo.getvalue().decode("utf-8")))
            st.success(f"Zonas importadas: {count}.")
            st.rerun()
        except Exception as exc:
            st.error(f"No se pudo importar el archivo: {exc}")

elif page == "Historial y registros":
    st.subheader("Series históricas y observaciones de campo")
    st.markdown("#### Consultar precipitación y humedad histórica")
    st.info("La API permite solicitar ERA5-Land desde 1950. La resolución aproximada es 0,1° (alrededor de 9–11 km); los datos son de reanálisis y no sustituyen estaciones pluviométricas ni registros del Cuerpo de Bomberos.")
    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.date_input("Fecha inicial", value=date(2016, 1, 1), min_value=date(1950, 1, 1), max_value=date.today() - timedelta(days=6))
    with c2:
        end = st.date_input("Fecha final", value=date(2024, 12, 31), min_value=date(1950, 1, 1), max_value=date.today() - timedelta(days=6))
    with c3:
        model_name = st.selectbox("Producto de reanálisis", ["ERA5-Land", "ERA5", "Best match"], index=0)
    model_key = {"ERA5-Land": "era5_land", "ERA5": "era5", "Best match": "best_match"}[model_name]
    if start > end:
        st.error("La fecha inicial debe ser anterior o igual a la fecha final.")
    elif st.button("Consultar y compilar historial", type="primary"):
        with st.spinner("Descargando la serie histórica desde Open-Meteo..."):
            try:
                history = cached_historical(zone["latitude"], zone["longitude"], start.isoformat(), end.isoformat(), model_key)
                hourly = history.get("hourly", {})
                if not hourly.get("time"):
                    st.warning("La API no devolvió una serie horaria para el periodo.")
                else:
                    hist_df = pd.DataFrame({"time": pd.to_datetime(hourly["time"]),
                                            "precipitation_mm": hourly.get("precipitation", []),
                                            "soil_moisture_0_7cm": hourly.get("soil_moisture_0_to_7cm", [])})
                    st.session_state[f"history_{selected_id}"] = hist_df
                    st.session_state[f"history_model_{selected_id}"] = model_name
                    st.success(f"Serie obtenida: {len(hist_df):,} registros horarios. Coordenada del punto de grilla: {history.get('latitude')}, {history.get('longitude')}.")
            except ProviderError as exc:
                st.error(str(exc))
    hist_df = st.session_state.get(f"history_{selected_id}")
    if hist_df is not None and not hist_df.empty:
        daily = hist_df.set_index("time").resample("D").agg(
            precipitation_mm=("precipitation_mm", "sum"),
            soil_moisture_0_7cm=("soil_moisture_0_7cm", "mean"),
        ).reset_index()
        fig = px.bar(daily, x="time", y="precipitation_mm", title="Precipitación diaria acumulada (mm)",
                     labels={"time": "Fecha", "precipitation_mm": "Precipitación (mm)"})
        st.plotly_chart(fig, use_container_width=True)
        monthly = daily.assign(month=daily["time"].dt.to_period("M").dt.to_timestamp()).groupby("month", as_index=False)["precipitation_mm"].sum()
        figm = px.line(monthly, x="month", y="precipitation_mm", title="Acumulado mensual de precipitación (mm)")
        st.plotly_chart(figm, use_container_width=True)
        st.dataframe(daily.tail(30), use_container_width=True, hide_index=True)
        hist_csv = hist_df.to_csv(index=False).encode("utf-8-sig")
        daily_csv = daily.to_csv(index=False).encode("utf-8-sig")
        c1, c2 = st.columns(2)
        with c1:
            download_button("Descargar historial horario CSV", hist_csv, f"satid_{selected_id}_historico_horario.csv", "text/csv")
        with c2:
            download_button("Descargar resumen diario CSV", daily_csv, f"satid_{selected_id}_historico_diario.csv", "text/csv")

    st.divider()
    st.markdown("#### Registrar evidencia u observación de campo")
    st.caption("Añade eventos históricos confirmados por Bomberos, comunidad o fuente documental. Indica siempre la fuente para poder auditar el dato.")
    with st.form("observation_form"):
        obs_date = st.date_input("Fecha de la observación o evento", value=date.today())
        c1, c2, c3 = st.columns(3)
        p24_known = c1.checkbox("Tengo dato de lluvia 24 h", value=True)
        p24_input = c1.number_input("Precipitación últimas 24 h (mm)", min_value=0.0, value=0.0, step=1.0)
        p72_known = c2.checkbox("Tengo dato de lluvia 72 h", value=True)
        p72_input = c2.number_input("Precipitación últimas 72 h (mm)", min_value=0.0, value=0.0, step=1.0)
        forecast_known = c3.checkbox("Tengo dato de pronóstico 24 h", value=False)
        forecast_input = c3.number_input("Pronóstico 24 h (mm; si aplica)", min_value=0.0, value=0.0, step=1.0)
        c4, c5 = st.columns(2)
        soil_value = c4.number_input("Humedad suelo 0–7 cm (m³/m³; 0 = sin dato)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.3f")
        observed_level = c5.selectbox("Nivel observado / etiqueta", ["Verde", "Amarillo", "Rojo", "Sin evento", "No determinado"])
        event_type = st.text_input("Tipo de evidencia", placeholder="Ej. desbordamiento, inundación de cultivo, lluvia fuerte, sin afectación")
        source_url = st.text_input("URL o referencia de la fuente", placeholder="https://...")
        notes = st.text_area("Notas (sin datos personales sensibles)")
        submitted = st.form_submit_button("Guardar registro", type="primary")
    if submitted:
        try:
            add_observation(selected_id, obs_date.isoformat(),
                            p24_input if p24_known else None,
                            p72_input if p72_known else None,
                            forecast_input if forecast_known else None,
                            soil_value if soil_value > 0 else None,
                            observed_level, event_type, source_url, notes)
            st.success("Registro guardado en la base local.")
        except Exception as exc:
            st.error(str(exc))

    observations = list_observations(selected_id)
    st.markdown("#### Registros guardados")
    if observations.empty:
        st.info("Aún no hay observaciones registradas para esta zona.")
    else:
        st.dataframe(observations, use_container_width=True, hide_index=True)
        download_button("Descargar registros CSV", observations.to_csv(index=False).encode("utf-8-sig"),
                        f"satid_observaciones_{selected_id}.csv", "text/csv")
        train_rows = observations[observations["observed_level"].isin(["Verde", "Amarillo", "Rojo"])].copy()
        if not train_rows.empty:
            training_csv = train_rows.rename(columns={"observed_level": "label"})[FEATURE_COLUMNS + ["label"]]
            download_button("Descargar registros etiquetados para Random Forest",
                            training_csv.to_csv(index=False).encode("utf-8-sig"),
                            f"satid_entrenamiento_{selected_id}.csv", "text/csv")

elif page == "Laboratorio IA":
    st.subheader("Laboratorio de aprendizaje automático")
    st.write("Permite entrenar un Random Forest con observaciones etiquetadas por el usuario. No se inventan registros históricos ni se entrena con valores de demostración.")
    st.warning("No uses este modelo como alerta operativa. Antes de cualquier uso real, se requiere una base representativa, revisión de etiquetas, validación temporal/espacial y concertación con autoridades locales.")
    st.markdown("#### Formato de datos requerido")
    st.code("precip_24h_mm,precip_72h_mm,forecast_24h_mm,soil_moisture,elevation_m,label\n45,85,20,0.34,3030,Amarillo\n70,130,60,0.40,3030,Rojo", language="csv")
    st.caption("Las filas del ejemplo solo explican el formato; no son mediciones reales y no forman un dataset descargable.")
    try:
        from pathlib import Path
        template_path = Path(__file__).parent / "data" / "plantilla_entrenamiento_rf.csv"
        if template_path.exists():
            download_button("Descargar plantilla CSV vacía", template_path.read_bytes(), "plantilla_entrenamiento_rf.csv", "text/csv")
    except Exception:
        pass
    dataset_file = st.file_uploader("Cargar CSV con datos etiquetados", type=["csv"], key="rf_dataset")
    if dataset_file is not None and st.button("Entrenar y evaluar Random Forest", type="primary"):
        try:
            train_df = pd.read_csv(dataset_file)
            model, report = train_random_forest(train_df)
            st.session_state.rf_model = model
            st.session_state.rf_report = report
            st.success("Modelo entrenado en esta sesión. El dashboard intentará usarlo para estimar la clase.")
        except Exception as exc:
            st.error(str(exc))
    report = st.session_state.get("rf_report")
    if report:
        st.markdown("#### Evaluación en conjunto de prueba")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros válidos", report["n_total"])
        c2.metric("Accuracy", f"{report['accuracy']:.1%}")
        c3.metric("Precisión ponderada", f"{report['precision_weighted']:.1%}")
        c4.metric("Recall ponderado", f"{report['recall_weighted']:.1%}")
        st.metric("Balanced accuracy", f"{report['balanced_accuracy']:.1%}")
        rep = pd.DataFrame(report["classification_report"]).T
        st.dataframe(rep, use_container_width=True)
        st.caption(f"Clases en el dataset: {', '.join(report['classes'])}. Partición: {report['n_train']} entrenamiento / {report['n_test']} prueba. Aleatoriedad fija: 42.")
        st.info("Las métricas dependen de la calidad y representatividad de los datos cargados. Una partición aleatoria no reemplaza una validación hindcast temporal entre 2016 y 2024.")
        if st.button("Desactivar modelo RF y volver a reglas", key="clear_model"):
            st.session_state.pop("rf_model", None)
            st.session_state.pop("rf_report", None)
            st.rerun()

elif page == "Fuentes y configuración":
    st.subheader("Fuentes abiertas, búsqueda web y trazabilidad")
    source_links(zone)
    st.divider()
    st.markdown("#### Añadir una fuente documental revisada")
    with st.form("source_form"):
        title = st.text_input("Título o nombre del documento")
        url = st.text_input("URL")
        source_date = st.text_input("Fecha del documento o consulta", placeholder="2024-03-15 o s. f.")
        summary = st.text_area("Resumen y dato que aporta")
        save_source = st.form_submit_button("Guardar fuente")
    if save_source:
        try:
            add_source(selected_id, title, url, source_date, summary)
            st.success("Fuente guardada.")
        except ValueError as exc:
            st.error(str(exc))
    sources_df = list_sources(selected_id)
    st.markdown("#### Fuentes registradas")
    if sources_df.empty:
        st.info("No se han guardado fuentes documentales para esta zona.")
    else:
        st.dataframe(sources_df, use_container_width=True, hide_index=True)
        download_button("Descargar fuentes CSV", sources_df.to_csv(index=False).encode("utf-8-sig"),
                        f"satid_fuentes_{selected_id}.csv", "text/csv")

    st.divider()
    st.markdown("#### Servicios gratuitos o de acceso abierto integrados")
    sources_table = pd.DataFrame([
        {"Servicio": "Open-Meteo Forecast", "Uso en SATID": "Lluvia reciente/pronosticada, humedad del suelo y variables meteorológicas.", "Acceso": "Sin API key; gratuito para uso no comercial dentro de límites."},
        {"Servicio": "Open-Meteo Historical Weather", "Uso en SATID": "ERA5-Land/ERA5 para series horarias históricas, incluidas 2016–2024.", "Acceso": "Sin API key; datos CC BY 4.0."},
        {"Servicio": "GloFAS vía Open-Meteo", "Uso en SATID": "Caudal fluvial modelado complementario; no determina la alerta.", "Acceso": "Consulta pública; la celda de río puede no coincidir con el cauce local."},
        {"Servicio": "OpenStreetMap / Nominatim / Overpass", "Uso en SATID": "Mapa base, búsqueda de lugares y elementos hídricos cartografiados.", "Acceso": "Abierto con límites de uso y atribución."},
        {"Servicio": "Copernicus Data Space", "Uso en SATID": "Visualizar/buscar Sentinel-1 y Sentinel-2 en el navegador.", "Acceso": "Datos abiertos; el Browser solicita cuenta gratuita para determinadas funciones."},
    ])
    st.dataframe(sources_table, use_container_width=True, hide_index=True)
    st.markdown("**Enlaces técnicos**")
    st.markdown("- [Open-Meteo Forecast API](https://open-meteo.com/en/docs)\n- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)\n- [Open-Meteo Flood API / GloFAS](https://open-meteo.com/en/docs/flood-api)\n- [Política de uso de teselas OSM](https://operations.osmfoundation.org/policies/tiles/)\n- [Política de Nominatim](https://operations.osmfoundation.org/policies/nominatim/)\n- [Copernicus Browser](https://dataspace.copernicus.eu/browser/)\n- [NASA GIBS (catálogo de imágenes globales)](https://nasa-gibs.github.io/gibs-api-docs/)")
    st.divider()
    st.markdown("#### Respaldo local de datos")
    all_obs = list_observations()
    all_sources = list_sources()
    all_zones = pd.DataFrame(list_zones())
    c1, c2, c3 = st.columns(3)
    c1.metric("Zonas", len(all_zones))
    c2.metric("Observaciones", len(all_obs))
    c3.metric("Fuentes", len(all_sources))
    if st.button("Preparar respaldo de registros en CSV", use_container_width=True):
        st.session_state.backup_ready = True
    if st.session_state.get("backup_ready"):
        c1, c2, c3 = st.columns(3)
        with c1:
            download_button("Zonas CSV", all_zones.to_csv(index=False).encode("utf-8-sig"), "satid_zonas_backup.csv", "text/csv")
        with c2:
            download_button("Observaciones CSV", all_obs.to_csv(index=False).encode("utf-8-sig"), "satid_observaciones_backup.csv", "text/csv")
        with c3:
            download_button("Fuentes CSV", all_sources.to_csv(index=False).encode("utf-8-sig"), "satid_fuentes_backup.csv", "text/csv")

st.divider()
st.caption("SATID es un prototipo experimental de apoyo al conocimiento del riesgo. No emite alertas oficiales, no garantiza anticipación de inundaciones y no sustituye los canales del IDEAM, UNGRD, Alcaldía ni organismos de socorro.")
