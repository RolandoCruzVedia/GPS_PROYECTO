"""
processing/build_dataset.py

Lee datos GPS JSON desde ingestion/data/gps/
integra datos climáticos SENAMHI
y genera dataset.csv para ML de predicción de congestión,
con granularidad por TRAMO (8 segmentos / 16 puntos de control).
"""

import json
import math
import pandas as pd
from pathlib import Path


# ── Rutas ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

# Antes era una ruta absoluta hardcodeada a /home/rolando/...
# Ahora se calcula en relación al script, igual que en ingestion/parse_batches.py
INGESTION_DATA_DIR = (BASE_DIR / ".." / "ingestion" / "data").resolve()
GPS_DIR = INGESTION_DATA_DIR / "gps"
SENAMHI_FILE = INGESTION_DATA_DIR / "senamhi" / "senamhi_tarija.json"

OUT_FILE = BASE_DIR / "dataset.csv"

RADIO_TOLERANCIA_METROS = 45.0


# ── Los 8 tramos / 16 puntos de control (mismos que ingestion/parse_batches.py) ─

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
    """
    Determina a qué tramo (de los 8) pertenece un punto GPS, según su
    cercanía a alguno de los 16 puntos de control (inicio o fin).
    Si el punto está dentro del radio de tolerancia de más de un tramo,
    se asigna al más cercano. Si no está cerca de ninguno, se descarta
    (devuelve None, None) — así el dataset final solo contiene registros
    realmente ubicados en los 16 puntos de control.
    """
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

    # Hora y fecha real Tarija
    df["hora"] = df["timestamp"].dt.tz_convert("America/La_Paz").dt.hour
    df["dia_semana"] = df["timestamp"].dt.tz_convert("America/La_Paz").dt.dayofweek
    df["fecha"] = df["timestamp"].dt.tz_convert("America/La_Paz").dt.date

    # Cargar SENAMHI antes de construir fecha_clima, para saber qué año(s) tiene disponible
    clima = cargar_senamhi()
    anios_clima = pd.to_datetime(clima["fecha"]).apply(lambda d: d.year)
    anio_clima_default = int(anios_clima.mode()[0])  # año más frecuente en SENAMHI

    # Buscar clima del mismo día/mes, usando el año disponible en SENAMHI
    # (proxy estacional: el GPS puede ser de un año sin registro climático propio).
    # NOTA: si el año del GPS coincide con un año presente en SENAMHI, se usa ese mismo año.
    anios_disponibles = set(anios_clima.unique())

    def mapear_fecha_clima(fecha_gps):
        anio = fecha_gps.year if fecha_gps.year in anios_disponibles else anio_clima_default
        try:
            return fecha_gps.replace(year=anio)
        except ValueError:
            # 29 de febrero u otro caso límite
            return fecha_gps.replace(year=anio, day=28)

    df["fecha_clima"] = pd.to_datetime(df["fecha"]).apply(mapear_fecha_clima).dt.date

    # Asignar tramo + zona (filtra automáticamente todo lo que no esté
    # dentro de ~45m de alguno de los 16 puntos de control)
    df[["tramo", "zona"]] = df.apply(
        lambda r: pd.Series(asignar_tramo(r["lat"], r["lng"])), axis=1
    )

    df = df.dropna(subset=["tramo"])
    print("[INFO] GPS dentro de los 16 puntos de control (8 tramos):", len(df))

    # Integrar SENAMHI
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

    # Velocidad → estado
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


# ── Agregación ───────────────────────────────────────────────────────────────


def agregar_franjas(df):
    # Granularidad por TRAMO (no por zona): hasta 8 filas por hora/día
    group = df.groupby(["tramo", "zona", "dia_semana", "hora"])

    agg = group.agg(
        total_registros=("speed", "count"),
        speed_promedio=("speed", "mean"),
        pct_detenido=("estado", lambda x: (x == "DETENIDO").sum() / len(x)),
        pct_lento=("estado", lambda x: (x == "LENTO").sum() / len(x)),
        pct_normal=("estado", lambda x: (x == "NORMAL").sum() / len(x)),
        pct_fluido=("estado", lambda x: (x == "FLUIDO").sum() / len(x)),
        dias_observados=("fecha", "nunique"),
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
    agg["zona_cod"] = agg["zona"].map(zona_map)

    tramo_map = {t: i for i, t in enumerate(sorted(agg["tramo"].unique()))}
    agg["tramo_cod"] = agg["tramo"].map(tramo_map)

    print("\n[INFO] Franjas (tramo x dia_semana x hora):", len(agg))
    return agg


# ── MAIN ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 50)
    print("Construcción dataset GPS + SENAMHI (por tramo)")
    print("=" * 50)

    registros = cargar_batches()
    df_raw = pd.DataFrame(registros)

    df_pings = procesar_pings(df_raw)
    df_final = agregar_franjas(df_pings)

    df_final.to_csv(OUT_FILE, index=False)

    print("[OK] Dataset generado:", OUT_FILE)
    print(df_final.head())


if __name__ == "__main__":
    main()
