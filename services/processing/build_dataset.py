"""
processing/build_dataset.py

Lee datos GPS JSON desde ingestion/data/gps/
integra datos climáticos SENAMHI
y genera dataset.csv como SERIE TEMPORAL REAL por tramo (fecha x hora),
con variables derivadas (lags, promedio movil, tendencia) y la variable
objetivo "velocidad_siguiente" (Etapas 3 a 6 de la metodologia).

IMPORTANTE (correccion de consistencia hora-etiqueta):
"velocidad_siguiente" debe representar una observacion razonablemente
cercana a "hora + 1" para que la prediccion sea confiable. El valor
original (GAP_MAXIMO_HORAS = 4) mezclaba en el entrenamiento observaciones
de 1 a 4 horas despues bajo la misma etiqueta "hora", diluyendo los picos
de congestion (una fila "hora=18" podia tener como objetivo la velocidad
de las 21:00, cuando el pico ya bajo).

Con datos reales muy limitados (pocos registros GPS por tramo/hora),
exigir gap == 1 estricto deja franjas dia/hora completas sin ninguna
prediccion (el dashboard muestra "SIN DATOS"). Como compromiso:
- GAP_MAXIMO_HORAS = 3 (en vez de 1 o del original 4): recupera cobertura
  sin llegar a los gaps largos que mas diluian los picos.
- El propio "gap_siguiente_horas" se pasa como FEATURE al modelo (ver
  train.py), para que el modelo aprenda a ponderar la confiabilidad de
  cada ejemplo segun que tan lejos esta la observacion objetivo, en vez
  de tratar gap=1 y gap=3 como si fueran equivalentes.
"""

import json
import math
import pandas as pd
from pathlib import Path


# ── Rutas ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
INGESTION_DATA_DIR = (BASE_DIR / ".." / "ingestion" / "data").resolve()
GPS_DIR = INGESTION_DATA_DIR / "gps"
SENAMHI_FILE = INGESTION_DATA_DIR / "senamhi" / "senamhi_tarija.json"

OUT_FILE = BASE_DIR / "dataset.csv"

RADIO_TOLERANCIA_METROS = 100.0

# Maximo de horas de separacion aceptado entre "hora actual" y "hora
# siguiente observada" para que el par cuente como ejemplo de entrenamiento.
#
# GAP_MAXIMO_HORAS = 3: compromiso entre cobertura y precision. Un gap=1
# estricto (ideal en teoria) deja franjas dia/hora completas sin ningun
# dato real cuando los registros GPS son escasos, lo que se traduce en
# "SIN DATOS" en el dashboard para esas franjas. Con gap<=3 se recupera
# cobertura, y el sesgo de mezclar distintos gaps se compensa incluyendo
# "gap_siguiente_horas" como feature del modelo (ver FEATURES en train.py),
# para que el modelo mismo aprenda a ajustar su confianza segun el gap.
GAP_MAXIMO_HORAS = 3


# ── Los 8 tramos / 16 puntos de control ─────────────────────────────────────

TRAMOS = [
    {"id_tramo": "MC_A1", "zona": "mercado_campesino",
     "p_inicio": {"lat": -21.517737321664196, "lng": -64.74488003418877},
     "p_fin":    {"lat": -21.51938172028478,  "lng": -64.7438103737863}},
    {"id_tramo": "MC_A2", "zona": "mercado_campesino",
     "p_inicio": {"lat": -21.52022907940224,  "lng": -64.74326596501395},
     "p_fin":    {"lat": -21.521567411831715, "lng": -64.74234326060315}},
    {"id_tramo": "MC_A3", "zona": "mercado_campesino",
     "p_inicio": {"lat": -21.52258399825951,  "lng": -64.7409424999812},
     "p_fin":    {"lat": -21.521584003029606, "lng": -64.74222873259598}},
    {"id_tramo": "MC_A4", "zona": "mercado_campesino",
     "p_inicio": {"lat": -21.519595029938777, "lng": -64.74355949561512},
     "p_fin":    {"lat": -21.519092988362985, "lng": -64.74388009006022}},

    {"id_tramo": "RSM_B1", "zona": "rotonda_san_martin",
     "p_inicio": {"lat": -21.53010340073023,  "lng": -64.74031280967205},
     "p_fin":    {"lat": -21.531346022517923, "lng": -64.74056120111543}},
    {"id_tramo": "RSM_B2", "zona": "rotonda_san_martin",
     "p_inicio": {"lat": -21.532320457897868, "lng": -64.74077008468787},
     "p_fin":    {"lat": -21.533540370540205, "lng": -64.74001574787113}},
    {"id_tramo": "RSM_B3", "zona": "rotonda_san_martin",
     "p_inicio": {"lat": -21.535137796594412, "lng": -64.73806559762815},
     "p_fin":    {"lat": -21.533051003627207, "lng": -64.73992923991153}},
    {"id_tramo": "RSM_B4", "zona": "rotonda_san_martin",
     "p_inicio": {"lat": -21.532259297741135, "lng": -64.74035456869238},
     "p_fin":    {"lat": -21.53133667130067,  "lng": -64.74046409738772}},
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def haversine_m(lat1, lng1, lat2, lng2):
    R = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def asignar_tramo(lat, lng):
    mejor_tramo = None
    mejor_distancia = RADIO_TOLERANCIA_METROS

    for tramo in TRAMOS:
        d_inicio = haversine_m(lat, lng, tramo["p_inicio"]["lat"], tramo["p_inicio"]["lng"])
        d_fin = haversine_m(lat, lng, tramo["p_fin"]["lat"], tramo["p_fin"]["lng"])
        d_min = min(d_inicio, d_fin)

        if d_min <= RADIO_TOLERANCIA_METROS and d_min <= mejor_distancia:
            mejor_distancia = d_min
            mejor_tramo = tramo

    if mejor_tramo is None:
        return None, None

    return mejor_tramo["id_tramo"], mejor_tramo["zona"]


def nivel_congestion(pct_detenido, pct_lento):
    """Se conserva solo como variable descriptiva / comparativa con el
    modelo anterior. NO es la variable objetivo del modelo de regresion."""
    congestionado = pct_detenido + pct_lento
    if congestionado >= 0.70:
        return "ALTO"
    elif congestionado >= 0.40:
        return "MEDIO"
    else:
        return "BAJO"


# ── Cargar GPS ───────────────────────────────────────────────────────────────


def cargar_batches():
    archivos = list(GPS_DIR.rglob("*.json"))
    print(f"[INFO] Archivos GPS encontrados: {len(archivos)}")

    registros = []
    for archivo in archivos:
        with open(archivo, "r") as f:
            try:
                datos = json.load(f)
                if isinstance(datos, list):
                    registros.extend(datos)
                else:
                    registros.append(datos)
            except json.JSONDecodeError as e:
                print("[WARN]", archivo, e)

    print(f"[INFO] Registros GPS cargados: {len(registros)}")
    return registros


# ── Cargar SENAMHI ───────────────────────────────────────────────────────────


def cargar_senamhi():
    print(f"[INFO] Cargando SENAMHI: {SENAMHI_FILE}")

    with open(SENAMHI_FILE, "r") as f:
        clima = json.load(f)

    df_clima = pd.DataFrame(clima)
    df_clima["fecha"] = pd.to_datetime(df_clima["fecha"]).dt.date
    return df_clima


# ── Procesamiento GPS + clima ────────────────────────────────────────────────


def procesar_pings(df_raw):
    df = df_raw.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    antes = len(df)
    df = df.drop_duplicates(subset=["device_id", "timestamp"])
    print("[INFO] Duplicados eliminados:", antes - len(df))

    df["hora"] = df["timestamp"].dt.tz_convert("America/La_Paz").dt.hour
    df["dia_semana"] = df["timestamp"].dt.tz_convert("America/La_Paz").dt.dayofweek
    df["fecha"] = df["timestamp"].dt.tz_convert("America/La_Paz").dt.date

    clima = cargar_senamhi()
    anios_clima = pd.to_datetime(clima["fecha"]).apply(lambda d: d.year)
    anio_clima_default = int(anios_clima.mode()[0])
    anios_disponibles = set(anios_clima.unique())

    def mapear_fecha_clima(fecha_gps):
        anio = fecha_gps.year if fecha_gps.year in anios_disponibles else anio_clima_default
        try:
            return fecha_gps.replace(year=anio)
        except ValueError:
            return fecha_gps.replace(year=anio, day=28)

    df["fecha_clima"] = pd.to_datetime(df["fecha"]).apply(mapear_fecha_clima).dt.date

    df[["tramo", "zona"]] = df.apply(
        lambda r: pd.Series(asignar_tramo(r["lat"], r["lng"])), axis=1
    )

    df = df.dropna(subset=["tramo"])
    print("[INFO] GPS dentro de los 16 puntos de control (8 tramos):", len(df))

    df = df.merge(
        clima[["fecha", "temperatura_max", "temperatura_min", "precipitacion"]],
        left_on="fecha_clima",
        right_on="fecha",
        how="left",
        suffixes=("", "_clima"),
    )
    df = df.drop(columns=["fecha_clima_clima"], errors="ignore")

    print("[INFO] Clima agregado:")
    print(df[["temperatura_max", "temperatura_min", "precipitacion"]].notnull().sum())

    def clasificar(speed):
        if speed == 0:
            return "DETENIDO"
        elif speed <= 15:
            return "LENTO"
        elif speed <= 30:
            return "NORMAL"
        else:
            return "FLUIDO"

    df["estado"] = df["speed"].apply(clasificar)

    return df


# ── Etapa 3 + 4: estado del tramo por hora REAL (serie temporal verdadera) ──


def agregar_por_tramo_fecha_hora(df):
    group = df.groupby(["tramo", "zona", "fecha", "dia_semana", "hora"])

    agg = group.agg(
        total_registros=("speed", "count"),
        speed_promedio=("speed", "mean"),
        pct_detenido=("estado", lambda x: (x == "DETENIDO").sum() / len(x)),
        pct_lento=("estado", lambda x: (x == "LENTO").sum() / len(x)),
        pct_normal=("estado", lambda x: (x == "NORMAL").sum() / len(x)),
        pct_fluido=("estado", lambda x: (x == "FLUIDO").sum() / len(x)),
        temperatura_max=("temperatura_max", "mean"),
        temperatura_min=("temperatura_min", "mean"),
        precipitacion=("precipitacion", "mean"),
    ).reset_index()

    agg["pct_congestion"] = (agg["pct_detenido"] + agg["pct_lento"]).round(4)
    agg["nivel_congestion"] = agg.apply(
        lambda r: nivel_congestion(r["pct_detenido"], r["pct_lento"]), axis=1
    )
    agg["es_hora_pico"] = agg["hora"].apply(
        lambda h: 1 if h in range(7, 9) or h in range(17, 20) else 0
    )

    zona_map = {z: i for i, z in enumerate(sorted(agg["zona"].unique()))}
    tramo_map = {t: i for i, t in enumerate(sorted(agg["tramo"].unique()))}
    agg["zona_cod"] = agg["zona"].map(zona_map)
    agg["tramo_cod"] = agg["tramo"].map(tramo_map)

    agg = agg.sort_values(["tramo", "fecha", "hora"]).reset_index(drop=True)
    print(f"\n[INFO] Filas tramo x fecha x hora (serie real): {len(agg)}")
    return agg


# ── Etapa 5 (variables derivadas) + Etapa 6 (variable objetivo) ────────────


def construir_variables_derivadas(agg):
    agg = agg.copy()
    grp = agg.groupby(["tramo", "fecha"])

    # --- Etapa 5: lags / promedio movil / tendencia ---
    agg["velocidad_hora_anterior"] = grp["speed_promedio"].shift(1)
    agg["hora_anterior_obs"] = grp["hora"].shift(1)
    agg["gap_anterior_horas"] = agg["hora"] - agg["hora_anterior_obs"]

    agg["promedio_movil_2"] = (
        grp["speed_promedio"].shift(1).rolling(2, min_periods=1).mean().reset_index(drop=True)
    )
    agg["promedio_movil_3"] = (
        grp["speed_promedio"].shift(1).rolling(3, min_periods=1).mean().reset_index(drop=True)
    )

    agg["tendencia"] = agg["speed_promedio"] - agg["velocidad_hora_anterior"]

    # --- Etapa 6: variable objetivo (velocidad de la SIGUIENTE observacion) ---
    agg["velocidad_siguiente"] = grp["speed_promedio"].shift(-1)
    agg["hora_siguiente_obs"] = grp["hora"].shift(-1)
    agg["gap_siguiente_horas"] = agg["hora_siguiente_obs"] - agg["hora"]

    antes = len(agg)
    agg = agg.dropna(subset=["velocidad_siguiente"])
    agg = agg[agg["gap_siguiente_horas"] <= GAP_MAXIMO_HORAS]
    print(f"[INFO] Filas con objetivo valido (gap <= {GAP_MAXIMO_HORAS}h): {len(agg)} de {antes}")
    print("[INFO] Distribucion de gap_siguiente_horas:")
    print(agg["gap_siguiente_horas"].value_counts().sort_index())

    # ── Verificacion explicita de consistencia hora-etiqueta ──
    # Con GAP_MAXIMO_HORAS = 3, todo gap_siguiente_horas debe estar entre
    # 1 y 3 (nunca mayor, nunca negativo/cero). Si aparece algun valor fuera
    # de ese rango, es senal de que algo en la logica de agrupamiento/orden
    # esta mal. Se deja como asercion dura para que el pipeline falle
    # ruidosamente en vez de generar datos silenciosamente inconsistentes.
    gaps_unicos = sorted(agg["gap_siguiente_horas"].unique())
    if not all(1 <= g <= GAP_MAXIMO_HORAS for g in gaps_unicos):
        raise ValueError(
            f"[ERROR] Se esperaba que todos los gap_siguiente_horas estuvieran "
            f"entre 1 y {GAP_MAXIMO_HORAS}, pero se encontraron: {gaps_unicos}. "
            f"Revisa el agrupamiento en construir_variables_derivadas()."
        )
    print(f"[OK] Verificacion de consistencia: 'velocidad_siguiente' esta siempre "
          f"entre 1 y {GAP_MAXIMO_HORAS} horas despues de su fila correspondiente.")

    return agg.reset_index(drop=True)


# ── MAIN ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 55)
    print("Construccion dataset GPS + SENAMHI (serie temporal por tramo)")
    print("=" * 55)

    registros = cargar_batches()
    df_raw = pd.DataFrame(registros)

    df_pings = procesar_pings(df_raw)
    agg = agregar_por_tramo_fecha_hora(df_pings)
    df_final = construir_variables_derivadas(agg)

    df_final.to_csv(OUT_FILE, index=False)

    print("\n[OK] Dataset generado:", OUT_FILE)
    print(df_final[["tramo", "fecha", "hora", "speed_promedio",
                     "velocidad_hora_anterior", "velocidad_siguiente",
                     "gap_siguiente_horas"]].head(10))


if __name__ == "__main__":
    main()
