import pandas as pd
import numpy as np

archivo = "dato_diarios (83).xlsx"

# 1. Leer el archivo Excel original
df_original = pd.read_excel(archivo)

# =====================================================
# LIMPIEZA DE COLUMNAS
# =====================================================
df_original = df_original.rename(columns={
    "Precipitación": "precipitacion",
    '"Temperatura Máxima"': "temperatura_max",
    '"Temperatura Mínima"': "temperatura_min"
})

# =====================================================
# CREAR FECHA EN EL DATASET ORIGINAL
# =====================================================
df_original["fecha"] = pd.to_datetime(
    df_original["gestion"].astype(str) + "-" +
    df_original["mes"].astype(str) + "-" +
    df_original["dia"].astype(str)
)

# =====================================================
# EXTENDER EL DATASET HASTA HOY (29/06/2026)
# =====================================================
fecha_minima = df_original["fecha"].min()
fecha_actual = pd.to_datetime("2026-06-29")

rango_fechas = pd.date_range(start=fecha_minima, end=fecha_actual, freq='D')
df_completo = pd.DataFrame({"fecha": rango_fechas})

df = pd.merge(df_completo, df_original, on="fecha", how="left")

df["aux_mes"] = df["fecha"].dt.month
df["aux_dia"] = df["fecha"].dt.day

# Rellenar metadatos fijos de la estación
df["estacion"] = df["estacion"].ffill()
df["longitud"] = df["longitud"].ffill()
df["latitud"] = df["latitud"].ffill()
df["altura"] = df["altura"].ffill()

# =====================================================
# CAPA 1: RELLENO ESTADÍSTICO HISTÓRICO (Mismo Día/Mes)
# =====================================================
climatologia = df.groupby(["aux_mes", "aux_dia"])[["temperatura_max", "temperatura_min", "precipitacion"]].transform("mean")

df["temperatura_max"] = df["temperatura_max"].fillna(climatologia["temperatura_max"])
df["temperatura_min"] = df["temperatura_min"].fillna(climatologia["temperatura_min"])
df["precipitacion"] = df["precipitacion"].fillna(climatologia["precipitacion"])

# =====================================================
# CAPA 2: ESCUDO ANTI-NULLS (Interpolación Lineal Fallback)
# =====================================================
# Aseguramos el orden cronológico estricto
df = df.sort_values(by="fecha").reset_index(drop=True)

# CAMBIO AQUÍ: Usamos method="linear" para evitar la restricción del DatetimeIndex
df["temperatura_max"] = df["temperatura_max"].interpolate(method="linear").ffill().bfill().round(1)
df["temperatura_min"] = df["temperatura_min"].interpolate(method="linear").ffill().bfill().round(1)
df["precipitacion"] = df["precipitacion"].interpolate(method="linear").fillna(0.0).round(1)

# =====================================================
# SELECCIONAR Y EXPORTAR A JSON
# =====================================================
df = df[
    [
        "fecha",
        "estacion",
        "longitud",
        "latitud",
        "altura",
        "precipitacion",
        "temperatura_max",
        "temperatura_min"
    ]
]

df["fecha"] = df["fecha"].dt.strftime("%Y-%m-%d")

archivo_salida = "senamhi_tarija.json"
df.to_json(
    archivo_salida,
    orient="records",
    force_ascii=False,
    indent=4
)

print(f"\n✔ ¡Procesado con doble capa de seguridad! JSON generado en: '{archivo_salida}'")
