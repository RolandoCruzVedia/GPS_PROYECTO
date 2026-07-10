"""
ingestion/parse_batches.py

Genera UNICAMENTE la capa geometrica (vias.geojson) que usa el mapa para
dibujar los 8 tramos de control, consultando a OSRM la polilinea real
adaptada a las calles de Tarija entre cada punto de inicio y fin.

IMPORTANTE - separacion de responsabilidades:
Este script YA NO calcula velocidad promedio ni nivel de congestion por
tramo. Ese calculo pertenece exclusivamente al pipeline de prediccion:

    build_dataset.py -> generar_sinteticos.py -> train.py -> predicciones.json

El color/nivel de cada tramo que ve el usuario en el mapa lo asigna
services/webapp/app.py (endpoint /api/vias_trafico) leyendo predicciones.json
y usando id_tramo como llave — NO se calcula aqui. Este archivo solo aporta
la geometria (la forma de la linea en el mapa).

Se elimino ademas la generacion de resumen_zonas.json: no era consumido
por ningun endpoint de app.py (webapp/data/resumen_zonas.json era un
calculo huerfano, un segundo pipeline de congestion desconectado del
dashboard real). El panel de zonas ("NIVEL GENERAL" / tarjetas Zona A y B)
se alimenta de predicciones_zonas.json, generado por train.py.
"""

import os
import json
import requests

# --- CONFIGURACIÓN DE RUTAS ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_GEOJSON = os.path.abspath(os.path.join(BASE_DIR, "../webapp/data/vias.geojson"))

OSRM_URL = "https://router.project-osrm.org/route/v1/driving/"

# --- PARAMETRIZACIÓN DE LOS 8 TRAMOS (16 PUNTOS) ---
#
# IMPORTANTE: estas coordenadas deben coincidir EXACTAMENTE con las de
# TRAMOS en services/processing/build_dataset.py. Son los mismos 8
# segmentos fisicos; si se editan aqui sin editar tambien alla (o
# viceversa), la geometria dibujada en el mapa (este archivo) dejara de
# corresponder al segmento real sobre el que se calcula la velocidad y el
# nivel de congestion (build_dataset.py / predicciones.json). Se recomienda
# unificar ambas listas en un solo modulo compartido (ej.
# services/common/tramos.py) para eliminar esta duplicacion de raiz.
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
    """Consulta a OSRM la polilínea exacta adaptada a las calles de Tarija."""
    url = f"{OSRM_URL}{p1['lng']},{p1['lat']};{p2['lng']},{p2['lat']}?overview=full&geometries=geojson"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get('code') == 'Ok':
            return res['routes'][0]['geometry']['coordinates']
    except Exception as e:
        print(f"   [!] Error en OSRM para {p1['codigo']}->{p2['codigo']}: {e}")
    # Fallback: linea recta entre los dos puntos si OSRM falla
    return [[p1['lng'], p1['lat']], [p2['lng'], p2['lat']]]


def generar_geojson_vias():
    """
    Genera vias.geojson con la geometria de los 8 tramos de control.
    NO calcula velocidad ni nivel de congestion — solo geometria.
    El nivel/color se asigna en tiempo real en services/webapp/app.py,
    leyendo predicciones.json (generado por train.py) y cruzando por
    id_tramo + dia_semana + hora.
    """
    print("\n==========================================")
    print(" GENERANDO CAPA GEOMETRICA (VIAS.GEOJSON) ")
    print("==========================================")

    features = []
    for tramo in TRAMOS:
        print(f"-> Tramo {tramo['id_tramo']} ({tramo['p_inicio']['codigo']} -> {tramo['p_fin']['codigo']})")

        coordenadas_viales = obtener_geometria_osrm(tramo["p_inicio"], tramo["p_fin"])

        feature = {
            "type": "Feature",
            "properties": {
                "id_tramo": tramo["id_tramo"],
                "zona":     tramo["zona"],
                "p_inicio": tramo["p_inicio"]["codigo"],
                "p_fin":    tramo["p_fin"]["codigo"],
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coordenadas_viales
            }
        }
        features.append(feature)

    geojson_final = {
        "type": "FeatureCollection",
        "features": features
    }

    os.makedirs(os.path.dirname(OUTPUT_GEOJSON), exist_ok=True)
    with open(OUTPUT_GEOJSON, 'w', encoding='utf-8') as f:
        json.dump(geojson_final, f, ensure_ascii=False, indent=4)

    print(f"\n[!] PROCESO COMPLETADO EXITOSAMENTE.")
    print(f"Capa de tráfico (8 tramos, solo geometria) exportada en: {OUTPUT_GEOJSON}\n")


if __name__ == "__main__":
    generar_geojson_vias()
