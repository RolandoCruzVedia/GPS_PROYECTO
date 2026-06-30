"""
webapp/app.py
Flask + Leaflet/OpenStreetMap: Dashboard de predicción de congestión vehicular
para el Jefe de Personal de Tránsito de la Policía Boliviana (Tarija).

Modelo de regresion de velocidad (Etapa 6) con umbrales por percentiles
empiricos (Etapa 7). Muestra 8 tramos coloreados + barra lateral Zona A / Zona B.
"""

import json
import joblib
from flask import Flask, render_template, jsonify, request
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

BASE_DIR        = Path(__file__).resolve().parent
MODELS_DIR      = BASE_DIR.parent / "ml" / "models"
DATASET         = BASE_DIR.parent / "processing" / "dataset.csv"
PRED_FILE       = BASE_DIR.parent / "ml" / "predicciones.json"
PRED_ZONAS_FILE = BASE_DIR.parent / "ml" / "predicciones_zonas.json"
GPS_DIR         = BASE_DIR.parent / "ingestion" / "data" / "gps"
GEOJSON_FILE    = BASE_DIR / "data" / "vias.geojson"

NIVEL_COLOR = {
    "ALTO":  "#EF4444",
    "MEDIO": "#F59E0B",
    "BAJO":  "#10B981",
}

NIVEL_PESO = {
    "ALTO":  1.0,
    "MEDIO": 0.55,
    "BAJO":  0.15,
}

DIAS = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]

ZONAS_META = {
    "mercado_campesino": {"nombre": "Zona A — Mercado Campesino"},
    "rotonda_san_martin": {"nombre": "Zona B — Rotonda San Martín"},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def cargar_modelo():
    mp = MODELS_DIR / "velocidad_model.pkl"
    if mp.exists():
        return joblib.load(mp)
    return None


def cargar_predicciones():
    if PRED_FILE.exists():
        with open(PRED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def cargar_predicciones_zonas():
    if PRED_ZONAS_FILE.exists():
        with open(PRED_ZONAS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def buscar_franja_tramo(tramo_id, dia_semana, hora, predicciones):
    for p in predicciones:
        if (p.get("tramo") == tramo_id
                and p.get("dia_semana") == dia_semana
                and p.get("hora") == hora):
            return p
    return None


def buscar_franja_zona(zona, dia_semana, hora, pred_zonas):
    for p in pred_zonas:
        if (p.get("zona") == zona
                and p.get("dia_semana") == dia_semana
                and p.get("hora") == hora):
            return p
    return None


# ── Rutas ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/vias_trafico")
def api_vias_trafico():
    """
    Retorna la geometria de los 8 tramos (de vias.geojson) con el nivel de
    congestion predicho para la franja dia/hora solicitada.
    Cada tramo se colorea de forma independiente: ALTO=rojo, MEDIO=naranja, BAJO=verde.
    """
    predicciones = cargar_predicciones()

    if not GEOJSON_FILE.exists():
        return jsonify({"error": f"vias.geojson no encontrado"}), 404

    dia  = request.args.get("dia",  type=int, default=datetime.now().weekday())
    hora = request.args.get("hora", type=int, default=datetime.now().hour)

    with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    resultado = []

    for feature in geojson_data.get("features", []):
        props = feature.get("properties", {})
        geom  = feature.get("geometry", {})

        tramo_id   = props.get("id_tramo")
        zona_str   = props.get("zona", "").lower().replace(" ", "_")
        p_inicio   = props.get("p_inicio", "")
        p_fin      = props.get("p_fin", "")
        nombre     = f"{zona_str.replace('_', ' ').title()} — {p_inicio} → {p_fin}"
        coordenadas = geom.get("coordinates", [])

        franja = buscar_franja_tramo(tramo_id, dia, hora, predicciones)

        if franja:
            nivel   = franja.get("nivel", "SIN DATOS")
            pct     = franja.get("pct_congestion", 0)
            v_pred  = franja.get("velocidad_predicha_siguiente", 0)
            v_libre = franja.get("velocidad_libre_tramo", 0)
            color   = NIVEL_COLOR.get(nivel, "#6B7280")

            resultado.append({
                "id_via":           tramo_id,
                "zona":             zona_str,
                "nombre_tramo":     nombre,
                "coordenadas":      coordenadas,
                "nivel":            nivel,
                "pct":              pct,
                "color":            color,
                "velocidad_actual":          franja.get("velocidad_actual", 0),
                "velocidad_predicha":        round(v_pred, 1),
                "velocidad_libre_tramo":     round(v_libre, 1),
                "temperatura_max":           franja.get("temperatura_max"),
                "temperatura_min":           franja.get("temperatura_min"),
                "precipitacion":             franja.get("precipitacion"),
            })
        else:
            resultado.append({
                "id_via":       tramo_id,
                "zona":         zona_str,
                "nombre_tramo": nombre,
                "coordenadas":  coordenadas,
                "nivel":        "SIN DATOS",
                "pct":          0,
                "color":        "#6B7280",
                "velocidad_actual":      None,
                "velocidad_predicha":    None,
                "velocidad_libre_tramo": None,
                "temperatura_max": None,
                "temperatura_min": None,
                "precipitacion":   None,
            })

    return jsonify(resultado)


@app.route("/api/resumen")
def resumen():
    """
    Panel lateral: indicadores consolidados por zona (Zona A y Zona B),
    leyendo predicciones_zonas.json generado por train.py.
    """
    pred_zonas = cargar_predicciones_zonas()

    dia  = request.args.get("dia",  type=int, default=datetime.now().weekday())
    hora = request.args.get("hora", type=int, default=datetime.now().hour)

    resultado = {}
    pesos_general = []

    for zona_id, meta in ZONAS_META.items():
        franja = buscar_franja_zona(zona_id, dia, hora, pred_zonas)

        if franja:
            nivel = franja.get("nivel", "SIN DATOS")
            pct   = franja.get("pct_congestion", 0)
            color = NIVEL_COLOR.get(nivel, "#6B7280")
            pesos_general.append(NIVEL_PESO.get(nivel, 0.5))

            resultado[zona_id] = {
                "nombre":             meta["nombre"],
                "nivel":              nivel,
                "pct_congestion":     pct,
                "color":              color,
                "tramos_considerados": franja.get("tramos_considerados", 0),
                "registros":          franja.get("registros", 0),
                "temperatura_max":    franja.get("temperatura_max"),
                "temperatura_min":    franja.get("temperatura_min"),
                "precipitacion":      franja.get("precipitacion"),
                "velocidad_predicha": franja.get("velocidad_predicha"),
            }
        else:
            resultado[zona_id] = {
                "nombre":         meta["nombre"],
                "nivel":          "SIN DATOS",
                "pct_congestion": 0,
                "color":          "#6B7280",
            }

    # Nivel general agregado de las dos zonas
    if pesos_general:
        prom = sum(pesos_general) / len(pesos_general)
        nivel_general = "ALTO" if prom >= 0.7 else ("MEDIO" if prom >= 0.4 else "BAJO")
    else:
        nivel_general = "SIN DATOS"

    resultado["nivel_general"]  = nivel_general
    resultado["color_general"]  = NIVEL_COLOR.get(nivel_general, "#6B7280")
    resultado["dia_nombre"]     = DIAS[dia]
    resultado["hora_label"]     = f"{hora:02d}:00 – {(hora+1)%24:02d}:00"
    resultado["generado"]       = datetime.now().isoformat()

    return jsonify(resultado)


@app.route("/api/predicciones")
def api_predicciones():
    """Filtra predicciones por tramo/zona/dia/hora (para uso de debug o futuras integraciones)."""
    predicciones = cargar_predicciones()
    if not predicciones:
        return jsonify({"error": "No hay predicciones disponibles."})

    tramo = request.args.get("tramo")
    zona  = request.args.get("zona")
    dia   = request.args.get("dia",  type=int)
    hora  = request.args.get("hora", type=int)

    res = predicciones
    if tramo:        res = [p for p in res if p.get("tramo") == tramo]
    if zona:         res = [p for p in res if p.get("zona")  == zona]
    if dia  is not None: res = [p for p in res if p.get("dia_semana") == dia]
    if hora is not None: res = [p for p in res if p.get("hora") == hora]

    return jsonify(res)


@app.route("/api/estado")
def estado():
    modelo = cargar_modelo()
    archivos_gps = len(list(GPS_DIR.rglob("*.json"))) if GPS_DIR.exists() else 0

    return jsonify({
        "modelo_activo":          "Random Forest (regresion velocidad)" if modelo else "ninguno",
        "predicciones_listas":    PRED_FILE.exists(),
        "predicciones_zonas_listas": PRED_ZONAS_FILE.exists(),
        "dataset_disponible":     DATASET.exists(),
        "archivos_gps":           archivos_gps,
        "timestamp":              datetime.now().isoformat(),
    })


@app.route("/api/zonas")
def zonas():
    """Delimitadores geograficos de las 2 zonas de control para el mapa."""
    return jsonify({
        "mercado_campesino": {
            "nombre":  "Zona A — Mercado Campesino",
            "lat": -21.520221, "lng": -64.743114,
            "radio_m": 150, "color": "#EF4444",
        },
        "rotonda_san_martin": {
            "nombre":  "Zona B — Rotonda San Martín",
            "lat": -21.532387, "lng": -64.740059,
            "radio_m": 150, "color": "#F59E0B",
        },
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
