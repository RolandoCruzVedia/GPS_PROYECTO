import os
import json
import requests
import math
from datetime import datetime, timedelta, timezone

try:
    import parse_gps
except ImportError:
    parse_gps = None

# --- CONFIGURACIÓN DE RUTAS ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CORREGIDO: los datos crudos están dentro de data/gps, no en toda la carpeta data
# (data/ también contiene senamhi/, que tiene otro esquema y no debe recorrerse aquí)
DATA_DIR = os.path.join(BASE_DIR, "data", "gps")

OUTPUT_GEOJSON = os.path.abspath(os.path.join(BASE_DIR, "../webapp/data/vias.geojson"))
OUTPUT_RESUMEN_ZONAS = os.path.abspath(os.path.join(BASE_DIR, "../webapp/data/resumen_zonas.json"))

OSRM_URL = "https://router.project-osrm.org/route/v1/driving/"

# --- TIMEZONE ---
# Los timestamps GPS llegan en UTC (+00:00).
# Los datos de SENAMHI están en horario local de Tarija (UTC-4, Bolivia no usa horario de verano).
# Convertimos cada punto GPS a hora local de Tarija para que cualquier cruce futuro
# por fecha/hora con SENAMHI (o con el selector "Día/Hora" del panel) sea consistente.
TARIJA_OFFSET = timedelta(hours=-4)

def utc_a_tarija(timestamp_str):
    """Convierte un timestamp ISO en UTC a datetime local de Tarija (naive, sin tzinfo)."""
    try:
        dt_utc = datetime.fromisoformat(timestamp_str)
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        dt_local = dt_utc.astimezone(timezone.utc) + TARIJA_OFFSET
        return dt_local.replace(tzinfo=None)
    except Exception:
        return None

# --- PARAMETRIZACIÓN DE LOS 8 TRAMOS (16 PUNTOS) ---
TRAMOS = [
    # --- Zona Mercado Campesino ---
    {
        "id_tramo": "MC_A1",
        "zona": "Mercado Campesino",
        "p_inicio": {"lat": -21.517737321664196, "lng": -64.74488003418877, "codigo": "A.1_inicio"},
        "p_fin":    {"lat": -21.51938172028478,  "lng": -64.7438103737863,  "codigo": "A.1_fin"}
    },
    {
        "id_tramo": "MC_A2",
        "zona": "Mercado Campesino",
        "p_inicio": {"lat": -21.52022907940224,  "lng": -64.74326596501395, "codigo": "A.2_inicio"},
        "p_fin":    {"lat": -21.521567411831715, "lng": -64.74234326060315, "codigo": "A.2_fin"}
    },
    {
        "id_tramo": "MC_A3",
        "zona": "Mercado Campesino",
        "p_inicio": {"lat": -21.52258399825951,  "lng": -64.7409424999812,  "codigo": "A.3_inicio"},
        "p_fin":    {"lat": -21.521584003029606, "lng": -64.74222873259598, "codigo": "A.3_fin"}
    },
    {
        "id_tramo": "MC_A4",
        "zona": "Mercado Campesino",
        "p_inicio": {"lat": -21.519595029938777, "lng": -64.74355949561512, "codigo": "A.4_inicio"},
        "p_fin":    {"lat": -21.519092988362985, "lng": -64.74388009006022, "codigo": "A.4_fin"}
    },
    # --- Zona Rotonda San Martín ---
    {
        "id_tramo": "RSM_B1",
        "zona": "Rotonda San Martín",
        "p_inicio": {"lat": -21.53010340073023,  "lng": -64.74031280967205, "codigo": "B.1_inicio"},
        "p_fin":    {"lat": -21.531346022517923, "lng": -64.74056120111543, "codigo": "B.1_fin"}
    },
    {
        "id_tramo": "RSM_B2",
        "zona": "Rotonda San Martín",
        "p_inicio": {"lat": -21.532320457897868, "lng": -64.74077008468787, "codigo": "B.2_inicio"},
        "p_fin":    {"lat": -21.533540370540205, "lng": -64.74001574787113, "codigo": "B.2_fin"}
    },
    {
        "id_tramo": "RSM_B3",
        "zona": "Rotonda San Martín",
        "p_inicio": {"lat": -21.535137796594412, "lng": -64.73806559762815, "codigo": "B.3_inicio"},
        "p_fin":    {"lat": -21.533051003627207, "lng": -64.73992923991153, "codigo": "B.3_fin"}
    },
    {
        "id_tramo": "RSM_B4",
        "zona": "Rotonda San Martín",
        "p_inicio": {"lat": -21.532259297741135, "lng": -64.74035456869238, "codigo": "B.4_inicio"},
        "p_fin":    {"lat": -21.53133667130067,  "lng": -64.74046409738772, "codigo": "B.4_fin"}
    }
]

def obtener_geometria_osrm(p1, p2):
    """Consulta a OSRM la polilínea exacta adaptada a las calles de Tarija"""
    url = f"{OSRM_URL}{p1['lng']},{p1['lat']};{p2['lng']},{p2['lat']}?overview=full&geometries=geojson"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get('code') == 'Ok':
            return res['routes'][0]['geometry']['coordinates']
    except Exception as e:
        print(f"   [!] Error en OSRM para {p1['codigo']}->{p2['codigo']}: {e}")
    return [[p1['lng'], p1['lat']], [p2['lng'], p2['lat']]]

def calcular_distancia_metros(lat1, lng1, lat2, lng2):
    """Fórmula de Haversine para distancia entre dos puntos GPS"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlng/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def clasificar_congestion(velocidades):
    """Determina nivel de congestión a partir de una lista de velocidades (km/h)"""
    if not velocidades:
        return "BAJO", 0.0
    velocidad_promedio = sum(velocidades) / len(velocidades)
    if velocidad_promedio < 10.0:
        return "ALTO", round(velocidad_promedio, 2)
    elif velocidad_promedio < 22.0:
        return "MEDIO", round(velocidad_promedio, 2)
    else:
        return "BAJO", round(velocidad_promedio, 2)

def procesar_lotes_s3():
    if not os.path.exists(DATA_DIR):
        print(f"Error: No se encuentra la carpeta de datos en {DATA_DIR}")
        return

    archivos_json = []
    for raiz, carpetas, archivos in os.walk(DATA_DIR):
        for archivo in archivos:
            if archivo.endswith(".json"):
                archivos_json.append(os.path.join(raiz, archivo))

    archivos_json.sort()
    print(f"\nTotal archivos GPS encontrados: {len(archivos_json)}")

    # Acumulador de velocidades por tramo (8 tramos)
    metricas_tramos = {t["id_tramo"]: [] for t in TRAMOS}
    RADIO_TOLERANCIA_METROS = 45.0

    for ruta_archivo in archivos_json:
        with open(ruta_archivo, 'r') as f:
            try:
                puntos_gps = json.load(f)
                for punto in puntos_gps:
                    lat = punto.get("lat")
                    lng = punto.get("lng")
                    vel = punto.get("speed", 0.0)
                    ts_utc = punto.get("timestamp")

                    if lat is None or lng is None:
                        continue

                    # Conversión a hora local de Tarija (queda disponible para
                    # futuros cruces por fecha/hora con SENAMHI o el panel de predicción)
                    ts_local = utc_a_tarija(ts_utc) if ts_utc else None

                    # Solo se consideran puntos GPS dentro del radio de alguno
                    # de los 16 puntos de control (inicio o fin de cada tramo)
                    for tramo in TRAMOS:
                        d_inicio = calcular_distancia_metros(lat, lng, tramo["p_inicio"]["lat"], tramo["p_inicio"]["lng"])
                        d_fin = calcular_distancia_metros(lat, lng, tramo["p_fin"]["lat"], tramo["p_fin"]["lng"])

                        if d_inicio <= RADIO_TOLERANCIA_METROS or d_fin <= RADIO_TOLERANCIA_METROS:
                            metricas_tramos[tramo["id_tramo"]].append(vel)

            except Exception as e:
                pass

    # --- Generar GeoJSON con los 8 segmentos, cada uno con su propio color/congestión ---
    features = []
    resumen_por_zona = {}  # acumula velocidades combinadas por zona para el panel agregado

    print("\n==========================================")
    print(" GENERANDO CAPA GEOMETRICA (VIAS.GEOJSON) ")
    print("==========================================")

    for tramo in TRAMOS:
        velocidades = metricas_tramos[tramo["id_tramo"]]
        print(f"-> Tramo {tramo['id_tramo']} ({tramo['p_inicio']['codigo']} -> {tramo['p_fin']['codigo']}): {len(velocidades)} posiciones registradas.")

        congestion, velocidad_promedio = clasificar_congestion(velocidades)

        coordenadas_viales = obtener_geometria_osrm(tramo["p_inicio"], tramo["p_fin"])

        feature = {
            "type": "Feature",
            "properties": {
                "id_tramo": tramo["id_tramo"],
                "zona": tramo["zona"],
                "p_inicio": tramo["p_inicio"]["codigo"],
                "p_fin": tramo["p_fin"]["codigo"],
                "congestion": congestion,
                "velocidad_promedio_kmh": velocidad_promedio
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coordenadas_viales
            }
        }
        features.append(feature)

        resumen_por_zona.setdefault(tramo["zona"], []).extend(velocidades)

    geojson_final = {
        "type": "FeatureCollection",
        "features": features
    }

    os.makedirs(os.path.dirname(OUTPUT_GEOJSON), exist_ok=True)
    with open(OUTPUT_GEOJSON, 'w', encoding='utf-8') as f:
        json.dump(geojson_final, f, ensure_ascii=False, indent=4)

    # --- Resumen agregado por zona (para el panel "NIVEL GENERAL" / tarjetas de zona) ---
    resumen_zonas_final = {}
    for zona, velocidades in resumen_por_zona.items():
        congestion, velocidad_promedio = clasificar_congestion(velocidades)
        # % de congestión estimado: invertimos la velocidad relativa a un máximo de referencia (40 km/h)
        VELOCIDAD_REFERENCIA = 40.0
        pct = max(0.0, min(100.0, round((1 - (velocidad_promedio / VELOCIDAD_REFERENCIA)) * 100, 1))) if velocidades else 0.0
        resumen_zonas_final[zona] = {
            "congestion": congestion,
            "velocidad_promedio_kmh": velocidad_promedio,
            "pct_congestion": pct,
            "muestras": len(velocidades)
        }

    os.makedirs(os.path.dirname(OUTPUT_RESUMEN_ZONAS), exist_ok=True)
    with open(OUTPUT_RESUMEN_ZONAS, 'w', encoding='utf-8') as f:
        json.dump(resumen_zonas_final, f, ensure_ascii=False, indent=4)

    print(f"\n[!] PROCESO COMPLETADO EXITOSAMENTE.")
    print(f"Capa de tráfico (8 tramos) exportada en: {OUTPUT_GEOJSON}")
    print(f"Resumen por zona exportado en: {OUTPUT_RESUMEN_ZONAS}\n")

if __name__ == "__main__":
    procesar_lotes_s3()
