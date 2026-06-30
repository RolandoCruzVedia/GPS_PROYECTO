"""
ml/train.py
Etapa 6: regresion para predecir velocidad_siguiente (Random Forest vs XGBoost),
comparada contra un baseline naive (persistencia: velocidad_siguiente = speed_promedio).
Etapa 7: umbrales BAJO/MEDIO/ALTO calculados con percentiles de los datos reales.

pct_congestion basado en velocidad libre por tramo (percentil 85 empirico),
estandar en ingenieria de transito.

Genera:
  - predicciones.json     -> por (tramo, dia_semana, hora) para el mapa
  - predicciones_zonas.json -> por (zona, dia_semana, hora) para la barra lateral
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report

try:
    from xgboost import XGBRegressor
    XGBOOST_DISPONIBLE = True
except ImportError:
    XGBOOST_DISPONIBLE = False
    print("[WARN] XGBoost no instalado. Solo se usara Random Forest.")

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR.parent / "processing" / "dataset_train_aumentado.csv"
TEST_FILE  = BASE_DIR.parent / "processing" / "dataset_test_real.csv"
REAL_FILE  = BASE_DIR.parent / "processing" / "dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_FILE      = MODELS_DIR / "velocidad_model.pkl"
THRESHOLDS_FILE = MODELS_DIR / "umbrales_congestion.json"
VLIBRES_FILE    = MODELS_DIR / "velocidades_libres.json"
PRED_FILE       = BASE_DIR  / "predicciones.json"
PRED_ZONAS_FILE = BASE_DIR  / "predicciones_zonas.json"

FEATURES = [
    "tramo_cod", "zona_cod", "dia_semana", "hora", "es_hora_pico",
    "total_registros", "temperatura_max", "temperatura_min", "precipitacion",
    "velocidad_hora_anterior", "promedio_movil_2", "promedio_movil_3", "tendencia",
]
TARGET = "velocidad_siguiente"
DIAS_NOMBRE = ["Lunes","Martes","Miercoles","Jueves","Viernes","Sabado","Domingo"]


# ── Carga y limpieza ──────────────────────────────────────────────────────────

def limpiar(df):
    for col in ["temperatura_max", "temperatura_min", "precipitacion"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    for col in ["velocidad_hora_anterior", "promedio_movil_2", "promedio_movil_3"]:
        df[col] = df[col].fillna(df["speed_promedio"])
    df["tendencia"] = df["tendencia"].fillna(0.0)
    return df


def cargar_datasets():
    df_train = limpiar(pd.read_csv(TRAIN_FILE))
    df_test  = limpiar(pd.read_csv(TEST_FILE))
    df_real  = limpiar(pd.read_csv(REAL_FILE))

    n_r = (df_train["es_sintetico"] == 0).sum()
    n_s = (df_train["es_sintetico"] == 1).sum()
    print(f"[INFO] Train: {len(df_train)} filas ({n_r} reales + {n_s} sinteticas)")
    print(f"[INFO] Test:  {len(df_test)} filas (100% reales)")
    print(f"[INFO] Dataset real completo (para v_libre + predicciones): {len(df_real)} filas")
    return df_train, df_test, df_real


# ── Velocidad libre por tramo (Etapa 7 base) ─────────────────────────────────

def calcular_velocidad_libre(df_real):
    """
    Percentil 85 de velocidades observadas por tramo.
    Estandar en ingenieria de transito para estimar la velocidad de flujo
    libre: la velocidad que el 85% de los vehiculos no supera en condiciones
    sin congestion. Se calcula sobre TODOS los datos reales (no solo train),
    porque es un parametro descriptivo del tramo, no del modelo.
    """
    v_global = float(df_real["speed_promedio"].quantile(0.85))
    v_libres = (
        df_real.groupby("tramo")["speed_promedio"]
        .quantile(0.85)
        .fillna(v_global)
        .to_dict()
    )
    print("\n[INFO] Velocidad libre por tramo (percentil 85 empirico):")
    for tramo, v in v_libres.items():
        print(f"  {tramo:<10} v_libre = {v:.2f} km/h")
    return v_libres


def velocidad_a_pct_congestion(v_pred, v_libre):
    """
    Porcentaje de congestion respecto a la velocidad libre del tramo.
    0%   = flujo libre (v_pred >= v_libre)
    100% = completamente congestionado (v_pred = 0)
    """
    if v_libre <= 0:
        return 0.0
    pct = max(0.0, (1.0 - v_pred / v_libre) * 100.0)
    return round(min(pct, 100.0), 1)


# ── Umbrales de clasificacion ─────────────────────────────────────────────────

def calcular_umbrales(velocidades, terciles=(33, 66)):
    p_bajo = np.percentile(velocidades, terciles[0])
    p_alto = np.percentile(velocidades, terciles[1])
    return {
        "umbral_alto_congestion":  round(float(p_bajo), 2),
        "umbral_medio_congestion": round(float(p_alto), 2),
    }


def velocidad_a_categoria(v, umbrales):
    if v <= umbrales["umbral_alto_congestion"]:
        return "ALTO"
    elif v <= umbrales["umbral_medio_congestion"]:
        return "MEDIO"
    else:
        return "BAJO"


# ── Evaluacion ────────────────────────────────────────────────────────────────

def evaluar_baseline_naive(df_test):
    y_true  = df_test[TARGET].values
    y_naive = df_test["speed_promedio"].values
    mae  = mean_absolute_error(y_true, y_naive)
    rmse = np.sqrt(mean_squared_error(y_true, y_naive))
    r2   = r2_score(y_true, y_naive)
    print("=" * 52)
    print("  BASELINE NAIVE (v_siguiente = v_actual)")
    print("=" * 52)
    print(f"  MAE:  {mae:.3f} km/h  |  RMSE: {rmse:.3f}  |  R2: {r2:.3f}")
    return mae, rmse, r2, y_naive


def evaluar_modelo(nombre, model, X_train, X_test, y_train, y_test, X_cv, y_cv):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("=" * 52)
    print(f"  {nombre}")
    print("=" * 52)
    print(f"  MAE:  {mae:.3f} km/h  |  RMSE: {rmse:.3f}  |  R2: {r2:.3f}")

    if len(X_cv) >= 6:
        n_splits = min(5, len(X_cv) // 2)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_cv, y_cv, cv=cv, scoring="neg_mean_absolute_error")
        print(f"  CV MAE ({n_splits}-fold, reales): {-scores.mean():.3f} +/- {scores.std():.3f}")

    if hasattr(model, "feature_importances_"):
        print("\n  Importancia de features:")
        for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
            print(f"  {feat:<25} {'#' * int(imp * 40)} {imp:.4f}")

    return mae, rmse, r2, y_pred


# ── Generacion de predicciones para el dashboard ──────────────────────────────

def generar_predicciones(df_real, model, umbrales, v_libres):
    """
    Genera predicciones agregadas por (tramo, dia_semana, hora) para que
    el dashboard las encuentre al filtrar por dia/hora — no por fecha exacta.
    Incluye pct_congestion basado en velocidad libre empirica del tramo.
    """
    df = df_real.copy()
    df["velocidad_predicha"] = model.predict(df[FEATURES])

    grp = df.groupby(["tramo", "zona", "dia_semana", "hora"]).agg(
        velocidad_actual          = ("speed_promedio", "mean"),
        velocidad_predicha_sig    = ("velocidad_predicha", "mean"),
        temperatura_max           = ("temperatura_max", "mean"),
        temperatura_min           = ("temperatura_min", "mean"),
        precipitacion             = ("precipitacion", "mean"),
        registros                 = ("total_registros", "sum"),
    ).reset_index()

    salida = []
    for _, row in grp.iterrows():
        v_pred  = float(row["velocidad_predicha_sig"])
        v_libre = v_libres.get(row["tramo"], float(df_real["speed_promedio"].quantile(0.85)))
        pct     = velocidad_a_pct_congestion(v_pred, v_libre)
        nivel   = velocidad_a_categoria(v_pred, umbrales)

        salida.append({
            "tramo":                      row["tramo"],
            "zona":                       row["zona"],
            "dia_semana":                 int(row["dia_semana"]),
            "dia_nombre":                 DIAS_NOMBRE[int(row["dia_semana"])],
            "hora":                       int(row["hora"]),
            "velocidad_actual":           round(float(row["velocidad_actual"]), 2),
            "velocidad_predicha_siguiente": round(v_pred, 2),
            "velocidad_libre_tramo":      round(v_libre, 2),
            "nivel":                      nivel,
            "pct_congestion":             pct,
            "temperatura_max":            round(float(row["temperatura_max"]), 1),
            "temperatura_min":            round(float(row["temperatura_min"]), 1),
            "precipitacion":              round(float(row["precipitacion"]), 1),
            "registros":                  int(row["registros"]),
        })

    print(f"[OK] predicciones.json: {len(salida)} franjas (tramo x dia_semana x hora)")
    return salida


def generar_predicciones_zonas(predicciones):
    """
    Agrega las predicciones de tramo a nivel de zona (promedio de los
    tramos de cada zona), para la barra lateral del dashboard.
    El nivel de zona se re-clasifica con los mismos umbrales de pct_congestion.
    """
    df = pd.DataFrame(predicciones)

    agg = df.groupby(["zona", "dia_semana", "dia_nombre", "hora"]).agg(
        pct_congestion        = ("pct_congestion", "mean"),
        velocidad_predicha    = ("velocidad_predicha_siguiente", "mean"),
        temperatura_max       = ("temperatura_max", "mean"),
        temperatura_min       = ("temperatura_min", "mean"),
        precipitacion         = ("precipitacion", "mean"),
        tramos_considerados   = ("tramo", "nunique"),
        registros             = ("registros", "sum"),
    ).reset_index()

    # Nivel de zona basado en pct_congestion promedio de sus tramos
    def nivel_zona(pct):
        if pct >= 70:
            return "ALTO"
        elif pct >= 40:
            return "MEDIO"
        else:
            return "BAJO"

    agg["nivel"]           = agg["pct_congestion"].apply(nivel_zona)
    agg["pct_congestion"]  = agg["pct_congestion"].round(1)
    agg["temperatura_max"] = agg["temperatura_max"].round(1)
    agg["temperatura_min"] = agg["temperatura_min"].round(1)
    agg["precipitacion"]   = agg["precipitacion"].round(1)

    salida = agg.to_dict(orient="records")
    print(f"[OK] predicciones_zonas.json: {len(salida)} franjas (zona x dia_semana x hora)")
    return salida


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("train.py - Regresion velocidad + pct_congestion por v_libre")
    print("=" * 52)

    df_train, df_test, df_real = cargar_datasets()

    X_train, y_train = df_train[FEATURES], df_train[TARGET]
    X_test,  y_test  = df_test[FEATURES],  df_test[TARGET]

    df_train_real   = df_train[df_train["es_sintetico"] == 0]
    df_real_cv      = pd.concat([df_train_real, df_test], ignore_index=True)
    X_cv, y_cv      = df_real_cv[FEATURES], df_real_cv[TARGET]

    # Velocidad libre por tramo (parametro descriptivo, usa TODOS los reales)
    v_libres = calcular_velocidad_libre(df_real)

    # Umbrales de clasificacion (percentiles 33/66 de velocidad_siguiente real)
    umbrales = calcular_umbrales(df_real_cv[TARGET])
    print(f"\n[INFO] Umbrales (percentiles 33/66 sobre velocidad_siguiente real):")
    print(f"  ALTO   si v <= {umbrales['umbral_alto_congestion']} km/h")
    print(f"  MEDIO  si v <= {umbrales['umbral_medio_congestion']} km/h")
    print(f"  BAJO   si v >  {umbrales['umbral_medio_congestion']} km/h\n")

    # Baseline
    mae_naive, rmse_naive, r2_naive, y_naive = evaluar_baseline_naive(df_test)

    # Modelos ML
    modelos = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=42
        ),
    }
    if XGBOOST_DISPONIBLE:
        modelos["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        )

    resultados = {"Baseline Naive": (mae_naive, rmse_naive, r2_naive)}
    modelos_entrenados = {}
    for nombre, model in modelos.items():
        mae, rmse, r2, _ = evaluar_modelo(
            nombre, model, X_train, X_test, y_train, y_test, X_cv, y_cv
        )
        resultados[nombre] = (mae, rmse, r2)
        modelos_entrenados[nombre] = model

    # Tabla comparativa
    print("\n" + "=" * 52)
    print("  COMPARATIVA FINAL (vs baseline naive)")
    print("=" * 52)
    print(f"  {'Modelo':<18} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print(f"  {'-'*44}")
    for nombre, (mae, rmse, r2) in resultados.items():
        marca = "  <- baseline" if nombre == "Baseline Naive" else ""
        print(f"  {nombre:<18} {mae:>8.3f} {rmse:>8.3f} {r2:>8.3f}{marca}")

    mejores_ml = {k: v for k, v in resultados.items() if k != "Baseline Naive"}
    mejor_nombre = min(mejores_ml, key=lambda k: mejores_ml[k][1])
    mejor_modelo = modelos_entrenados[mejor_nombre]
    mejora_pct = (1 - mejores_ml[mejor_nombre][0] / mae_naive) * 100
    print(f"\n  Mejor modelo ML: {mejor_nombre}")
    if mejora_pct > 0:
        print(f"  -> Mejora un {mejora_pct:.1f}% sobre baseline naive")
    else:
        print(f"  -> NO mejora al baseline ({abs(mejora_pct):.1f}% peor en MAE)")

    # Clasificacion derivada
    y_pred_test  = mejor_modelo.predict(X_test)
    cat_real     = [velocidad_a_categoria(v, umbrales) for v in y_test]
    cat_pred_ml  = [velocidad_a_categoria(v, umbrales) for v in y_pred_test]
    cat_naive    = [velocidad_a_categoria(v, umbrales) for v in y_naive]
    print(f"\n[INFO] Clasificacion derivada - {mejor_nombre}:")
    print(classification_report(cat_real, cat_pred_ml, zero_division=0))
    print("[INFO] Clasificacion derivada - Baseline Naive:")
    print(classification_report(cat_real, cat_naive, zero_division=0))

    # Guardar modelo y parametros
    joblib.dump(mejor_modelo, MODEL_FILE)
    with open(THRESHOLDS_FILE, "w", encoding="utf-8") as f:
        json.dump(umbrales, f, ensure_ascii=False, indent=2)
    with open(VLIBRES_FILE, "w", encoding="utf-8") as f:
        json.dump(v_libres, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Modelo: {MODEL_FILE}")
    print(f"[OK] Umbrales: {THRESHOLDS_FILE}")
    print(f"[OK] Velocidades libres: {VLIBRES_FILE}")

    # Generar JSONs para el dashboard
    predicciones = generar_predicciones(df_real, mejor_modelo, umbrales, v_libres)
    predicciones_zonas = generar_predicciones_zonas(predicciones)

    with open(PRED_FILE, "w", encoding="utf-8") as f:
        json.dump(predicciones, f, ensure_ascii=False, indent=2)
    with open(PRED_ZONAS_FILE, "w", encoding="utf-8") as f:
        json.dump(predicciones_zonas, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] {PRED_FILE}")
    print(f"[OK] {PRED_ZONAS_FILE}")
    print("\n[OK] Entrenamiento completado.")


if __name__ == "__main__":
    main()
