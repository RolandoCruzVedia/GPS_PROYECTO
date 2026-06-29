"""
webapp/app.py
Flask + Leaflet/OpenStreetMap: Dashboard de predicción de congestión vehicular
para el Jefe de Personal de Tránsito de la Policía Boliviana (Tarija).

Enfoque Metodológico de Maestría: Segmentación geométrica lineal orientada a
8 tramos de control (16 puntos) sobre la Av. Víctor Paz Estenssoro, agrupados
en 2 zonas de análisis (Mercado Campesino / Rotonda San Martín), incorporando
variables climáticas (temperatura y precipitación) de SENAMHI.
"""

import json
import glob
import joblib
from flask import Flask, render_template, jsonify, request
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

# ── Rutas de Archivos y Configuración ─────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
MODELS_DIR       = BASE_DIR.parent / "ml" / "models"
DATASET          = BASE_DIR.parent / "processing" / "dataset.csv"
PRED_FILE        = BASE_DIR.parent / "ml" / "predicciones.json"          # por tramo
PRED_ZONAS_FILE  = BASE_DIR.parent / "ml" / "predicciones_zonas.json"    # agregado por zona
GPS_DIR          = BASE_DIR.parent / "ingestion" / "data" / "gps"

# Geometría de los 8 tramos generada por ingestion/parse_batches.py + OSRM
GEOJSON_FILE = BASE_DIR / "data" / "vias.geojson"

# Metadatos de las 2 zonas de análisis para el panel lateral
ZONAS_ESTADISTICAS = {
    "mercado_campesino": {"nombre": "Mercado Campesino"},
    "rotonda_san_martin": {"nombre": "Rotonda San Martín"},
}

NIVEL_COLOR = {
    "ALTO":  "#EF4444",  # Rojo institucional Tránsito
    "MEDIO": "#F59E0B",  # Naranja de advertencia operacional
    "BAJO":  "#10B981",  # Verde flujo libre
}

NIVEL_PESO = {
    "ALTO":  1.0,
    "MEDIO": 0.55,
    "BAJO":  0.15,
}

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


# ── Helpers de Datos ──────────────────────────────────────────────────────────
def cargar_modelo():
    mp = MODELS_DIR / "congestion_model.pkl"
    lp = MODELS_DIR / "label_encoder.pkl"
    if mp.exists() and lp.exists():
        return joblib.load(mp), joblib.load(lp)
    return None, None


def cargar_predicciones():
    """Predicciones por TRAMO (8 segmentos) generadas por ml/train.py"""
    if PRED_FILE.exists():
        with open(PRED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def cargar_predicciones_zonas():
    """Predicciones agregadas por ZONA (2 zonas), promediando sus tramos"""
    if PRED_ZONAS_FILE.exists():
        with open(PRED_ZONAS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def franja_por_tramo(tramo_id, dia_semana, hora, predicciones):
    """Busca la predicción de un tramo específico para una franja espacio-temporal."""
    for p in predicciones:
        if (p["tramo"] == tramo_id
                and p["dia_semana"] == dia_semana
                and p["hora"] == hora):
            return p
    return None


def franja_por_zona(zona, dia_semana, hora, predicciones_zonas):
    """Busca el resumen agregado de una zona para una franja espacio-temporal."""
    for p in predicciones_zonas:
        if (p["zona"] == zona
                and p["dia_semana"] == dia_semana
                and p["hora"] == hora):
            return p
    return None


# ── Rutas del Servidor Flask ──────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predicciones")
def api_predicciones():
    """Filtra y retorna el universo de predicciones por tramo bajo parámetros dinámicos."""
    predicciones = cargar_predicciones()
    if not predicciones:
        return jsonify({"error": "No hay predicciones disponibles."})

    tramo = request.args.get("tramo")
    zona = request.args.get("zona")
    dia = request.args.get("dia", type=int)
    hora = request.args.get("hora", type=int)

    resultado = predicciones
    if tramo:
        resultado = [p for p in resultado if p["tramo"] == tramo]
    if zona:
        resultado = [p for p in resultado if p["zona"] == zona]
    if dia is not None:
        resultado = [p for p in resultado if p["dia_semana"] == dia]
    if hora is not None:
        resultado = [p for p in resultado if p["hora"] == hora]

    return jsonify(resultado)


@app.route("/api/vias_trafico")
def api_vias_trafico():
    """
    Extrae la geometría real de los 8 tramos desde vias.geojson e inyecta
    el nivel de congestión predicho (por tramo, no por zona) correspondiente
    a la franja día/hora solicitada — cada línea se colorea de forma
    independiente según su propio tramo.
    """
    predicciones = cargar_predicciones()
    if not predicciones:
        return jsonify([])

    if not GEOJSON_FILE.exists():
        return jsonify({"error": f"Archivo vias.geojson no encontrado en {GEOJSON_FILE}"}), 404

    dia = request.args.get("dia", type=int)
    hora = request.args.get("hora", type=int)

    if dia is None:
        dia = datetime.now().weekday()
    if hora is None:
        hora = datetime.now().hour

    with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    lineas_vias = []

    for feature in geojson_data.get("features", []):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})

        via_id = props.get("id_tramo")
        zona_tramo = props.get("zona")
        p_inicio = props.get("p_inicio")
        p_fin = props.get("p_fin")
        nombre_tramo = f"Av. Víctor Paz — {zona_tramo} ({p_inicio} → {p_fin})"
        coordenadas = geom.get("coordinates", [])

        # Match directo por tramo (ya no se necesita mapear vía -> zona)
        franja = franja_por_tramo(via_id, dia, hora, predicciones)

        if franja:
            nivel = franja["nivel"]
            peso = NIVEL_PESO.get(nivel, 0.5)

            lineas_vias.append({
                "id_via": via_id,
                "zona": zona_tramo,
                "nombre_tramo": nombre_tramo,
                "coordenadas": coordenadas,
                "peso": peso,
                "nivel": nivel,
                "pct": franja["pct_congestion"],
                "color": NIVEL_COLOR.get(nivel, "#6B7280"),
                "temperatura_max": franja.get("temperatura_max"),
                "temperatura_min": franja.get("temperatura_min"),
                "precipitacion": franja.get("precipitacion"),
            })
        else:
            lineas_vias.append({
                "id_via": via_id,
                "zona": zona_tramo,
                "nombre_tramo": nombre_tramo,
                "coordenadas": coordenadas,
                "peso": 0.0,
                "nivel": "SIN DATOS",
                "pct": 0,
                "color": "#6B7280",
                "temperatura_max": None,
                "temperatura_min": None,
                "precipitacion": None,
            })

    return jsonify(lineas_vias)


@app.route("/api/resumen")
def resumen():
    """
    Compila los indicadores consolidados por ZONA (2 zonas) para el panel
    lateral, leyendo el agregado precalculado en predicciones_zonas.json
    (promedio de los 4 tramos de cada zona), incluyendo clima.
    """
    predicciones_zonas = cargar_predicciones_zonas()
    if not predicciones_zonas:
        return jsonify({"error": "No hay predicciones agregadas por zona."})

    dia = request.args.get("dia", type=int, default=datetime.now().weekday())
    hora = request.args.get("hora", type=int, default=datetime.now().hour)

    resultado = {}
    nivel_general_pesos = []

    for zona_id, z in ZONAS_ESTADISTICAS.items():
        franja = franja_por_zona(zona_id, dia, hora, predicciones_zonas)
        if franja:
            nivel = franja["nivel"]
            resultado[zona_id] = {
                "nombre": z["nombre"],
                "nivel": nivel,
                "pct_congestion": franja["pct_congestion"],
                "tramos_considerados": franja.get("tramos_considerados", 0),
                "registros": franja.get("registros", 0),
                "temperatura_max": franja.get("temperatura_max"),
                "temperatura_min": franja.get("temperatura_min"),
                "precipitacion": franja.get("precipitacion"),
                "color": NIVEL_COLOR.get(nivel, "#6B7280"),
            }
            nivel_general_pesos.append(NIVEL_PESO.get(nivel, 0.5))
        else:
            resultado[zona_id] = {
                "nombre": z["nombre"],
                "nivel": "SIN DATOS",
                "pct_congestion": 0,
                "color": "#6B7280",
            }

    if nivel_general_pesos:
        prom = sum(nivel_general_pesos) / len(nivel_general_pesos)
        nivel_general = "ALTO" if prom >= 0.7 else ("MEDIO" if prom >= 0.4 else "BAJO")
    else:
        nivel_general = "SIN DATOS"

    resultado["nivel_general"] = nivel_general
    resultado["color_general"] = NIVEL_COLOR.get(nivel_general, "#6B7280")
    resultado["dia_nombre"] = DIAS[dia]
    resultado["hora_label"] = f"{hora:02d}:00 – {(hora+1)%24:02d}:00"
    resultado["generado"] = datetime.now().isoformat()

    return jsonify(resultado)


@app.route("/api/estado")
def estado():
    modelo, _ = cargar_modelo()
    # Corregido: la estructura real es gps/month=*/day=*/hour=*/batch_*.json
    archivos_batch = len(list(GPS_DIR.rglob("batch_*.json"))) if GPS_DIR.exists() else 0

    return jsonify({
        "modelo_activo": "Random Forest" if modelo else "ninguno",
        "predicciones_listas": PRED_FILE.exists(),
        "predicciones_zonas_listas": PRED_ZONAS_FILE.exists(),
        "dataset_disponible": DATASET.exists(),
        "archivos_batch": archivos_batch,
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/zonas")
def zonas():
    """Mantiene compatibilidad estructural de inicialización para delimitaciones generales."""
    return jsonify({
        "mercado_campesino": {
            "nombre": "Zona de Control A - Mercado Campesino",
            "lat": -21.520221, "lng": -64.743114,
            "radio_m": 150, "color": "#EF4444",
        },
        "rotonda_san_martin": {
            "nombre": "Zona de Control B - Rotonda San Martín",
            "lat": -21.532387, "lng": -64.740059,
            "radio_m": 150, "color": "#F59E0B",
        },
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
