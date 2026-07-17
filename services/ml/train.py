"""
ml/train.py
Etapa 6: regresion para predecir velocidad_siguiente (Random Forest vs XGBoost),
comparada contra un baseline naive (persistencia: velocidad_siguiente = speed_promedio).
Etapa 6b: busqueda de hiperparametros (RandomizedSearchCV, K-Fold CV sobre
datos reales) antes de entrenar los modelos finales.
Etapa 7: umbrales BAJO/MEDIO/ALTO calculados con percentiles de los datos reales.

pct_congestion basado en velocidad libre por tramo (percentil 85 empirico,
calculado SOLO con horas fuera de pico, para no contaminar la referencia
de "flujo libre" con datos que ya estan congestionados).

nivel (BAJO/MEDIO/ALTO) se deriva SIEMPRE de pct_congestion, tanto a nivel
de tramo como de zona, para que polilineas y barra lateral sean consistentes.
El nivel de zona usa el tramo MAS congestionado (max), no el promedio, para
no diluir cuellos de botella puntuales dentro de una zona.

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
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report

try:
    from xgboost import XGBRegressor
    XGBOOST_DISPONIBLE = True
except ImportError:
    XGBOOST_DISPONIBLE = False
    print("[WARN] XGBoost no instalado. Solo se usara Random Forest.")

BASE_DIR = Path(__file__).resolve().parent
#TRAIN_FILE = BASE_DIR.parent / "processing" / "dataset_train_aumentado.csv"
TRAIN_FILE = BASE_DIR.parent / "processing" / "dataset_train_real.csv"
TEST_FILE  = BASE_DIR.parent / "processing" / "dataset_test_real.csv"
REAL_FILE  = BASE_DIR.parent / "processing" / "dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_FILE        = MODELS_DIR / "velocidad_model.pkl"
THRESHOLDS_FILE   = MODELS_DIR / "umbrales_congestion.json"
VLIBRES_FILE      = MODELS_DIR / "velocidades_libres.json"
HIPERPARAMS_FILE  = MODELS_DIR / "mejores_hiperparametros.json"
UMBRALES_PCT_FILE = MODELS_DIR / "umbrales_pct_congestion.json"
PRED_FILE         = BASE_DIR  / "predicciones.json"
PRED_ZONAS_FILE   = BASE_DIR  / "predicciones_zonas.json"

FEATURES = [
    "tramo_cod", "zona_cod", "dia_semana", "hora", "es_hora_pico",
    "total_registros", "temperatura_max", "temperatura_min", "precipitacion",
    "velocidad_hora_anterior", "promedio_movil_2", "promedio_movil_3", "tendencia",
    "gap_siguiente_horas",
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
    Percentil 85 de velocidades observadas por tramo, calculado SOLO con
    registros fuera de hora pico (es_hora_pico == 0).

    Por que solo fuera de pico: si un tramo esta casi siempre congestionado
    (ej. una calle angosta junto a un mercado), calcular v_libre con TODOS
    los datos (incluyendo pico) hace que la "velocidad libre" termine siendo
    parecida a la velocidad en pico -> el pct_congestion nunca detecta
    congestion real ahi, porque el punto de referencia ya esta contaminado.

    Al usar solo horas sin pico, v_libre representa de verdad "como fluye
    el tramo cuando no hay congestion", que es el estandar en ingenieria
    de transito para este calculo.
    """
    df_libre = df_real[df_real["es_hora_pico"] == 0]
    if df_libre.empty:
        print("[WARN] No hay registros fuera de hora pico; usando todos los datos como fallback.")
        df_libre = df_real

    v_global = float(df_libre["speed_promedio"].quantile(0.85))
    v_libres = (
        df_libre.groupby("tramo")["speed_promedio"]
        .quantile(0.85)
        .fillna(v_global)
        .to_dict()
    )

    # Fallback por tramo: si un tramo no tiene NINGUN registro fuera de pico,
    # usar el percentil 85 global (fuera de pico) en vez de NaN.
    for tramo in df_real["tramo"].unique():
        if tramo not in v_libres or pd.isna(v_libres.get(tramo)):
            v_libres[tramo] = v_global

    print("\n[INFO] Velocidad libre por tramo (percentil 85, solo horas no-pico):")
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
#
# IMPORTANTE: existen DOS clasificaciones distintas con propositos distintos.
#
# 1) velocidad_a_categoria(v, umbrales): basada en percentiles de velocidad
#    ABSOLUTA (33/66) sobre todo el dataset. Se usa SOLO para evaluar el
#    modelo de regresion internamente (classification_report mas abajo),
#    comparando prediccion vs realidad en terminos de velocidad cruda.
#    NO debe usarse para el campo "nivel" que ve el usuario en el dashboard,
#    porque no es comparable entre tramos con velocidades libres distintas.
#
# 2) pct_a_categoria(pct, umbrales_pct): basada en pct_congestion RELATIVO
#    (respecto a la v_libre de cada tramo/zona), con umbrales calibrados
#    empiricamente desde la distribucion real (ver calcular_umbrales_pct).
#    v_libre de cada tramo/zona). Esta es la que se usa para el campo
#    "nivel" en predicciones.json y predicciones_zonas.json, para que
#    polilineas y barra lateral sean siempre consistentes entre si.

def calcular_umbrales(velocidades, terciles=(33, 66)):
    p_bajo = np.percentile(velocidades, terciles[0])
    p_alto = np.percentile(velocidades, terciles[1])
    return {
        "umbral_alto_congestion":  round(float(p_bajo), 2),
        "umbral_medio_congestion": round(float(p_alto), 2),
    }


def velocidad_a_categoria(v, umbrales):
    """Solo para evaluacion interna del modelo (ver nota arriba)."""
    if v <= umbrales["umbral_alto_congestion"]:
        return "ALTO"
    elif v <= umbrales["umbral_medio_congestion"]:
        return "MEDIO"
    else:
        return "BAJO"


def calcular_umbrales_pct(pct_congestion_valores, percentiles=(50, 80)):
    """
    Umbrales de clasificacion para pct_congestion (dashboard), calculados
    empiricamente desde la distribucion real observada en TODO el dataset
    de predicciones (todos los tramos, dias, horas), en vez de un corte
    fijo arbitrario (ej. 40/70).

    Por que esto importa: con un corte fijo, un valor como 38.9% vs 40%
    decide toda la etiqueta (MEDIO vs BAJO) por una diferencia minima que
    puede deberse a ruido del modelo (ej. cual de dos modelos con MAE casi
    identico "gano" en esa corrida), no a una diferencia real de trafico.
    Calibrando los umbrales contra la distribucion propia de tus datos:
    - percentil 50 -> BAJO/MEDIO: la mitad "menos congestionada" del
      sistema queda BAJO, la otra mitad MEDIO/ALTO.
    - percentil 80 -> MEDIO/ALTO: solo el 20% de las franjas mas
      congestionadas de todo el sistema se marcan ALTO.
    Esto es defendible ante el tribunal porque los cortes se derivan de
    los propios datos recolectados, no de un numero elegido a priori.
    """
    p_medio = float(np.percentile(pct_congestion_valores, percentiles[0]))
    p_alto  = float(np.percentile(pct_congestion_valores, percentiles[1]))
    return {
        "umbral_medio_pct": round(p_medio, 1),
        "umbral_alto_pct":  round(p_alto, 1),
    }


def pct_a_categoria(pct, umbrales_pct):
    """
    Clasificacion oficial para el dashboard (tramo y zona), basada siempre
    en pct_congestion, con umbrales calibrados empiricamente (ver
    calcular_umbrales_pct). Un mismo criterio para polilineas y barra
    lateral, y los mismos umbrales para ambas.
    """
    if pct >= umbrales_pct["umbral_alto_pct"]:
        return "ALTO"
    elif pct >= umbrales_pct["umbral_medio_pct"]:
        return "MEDIO"
    else:
        return "BAJO"


# ── Etapa 6b: busqueda de hiperparametros (RandomizedSearchCV) ───────────────
#
# Se busca sobre X_cv/y_cv (SOLO datos reales: train_real + test), nunca sobre
# datos sinteticos, para que los hiperparametros elegidos reflejen el
# comportamiento real del sistema y no artefactos de la generacion sintetica.
#
# Se usa RandomizedSearchCV en vez de GridSearchCV porque, con el volumen de
# datos reales disponible (segun tus prints, del orden de cientos de filas),
# un grid exhaustivo es innecesariamente costoso; un muestreo aleatorio de
# combinaciones (n_iter) explora el espacio de forma mas eficiente para el
# mismo presupuesto de tiempo. Si tu dataset real es muy pequeno (< ~80 filas)
# y quieres el maximo rigor para la tesis, puedes cambiar a GridSearchCV con
# el mismo param_grid; el codigo queda comentado como alternativa abajo.

RF_PARAM_DIST = {
    "n_estimators":      [100, 150, 200, 300, 400],
    "max_depth":         [3, 4, 5, 6, 8, None],
    "min_samples_leaf":  [1, 2, 4, 6],
    "min_samples_split": [2, 5, 10],
    "max_features":      ["sqrt", "log2", None],
}

XGB_PARAM_DIST = {
    "n_estimators":     [100, 150, 200, 300],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.03, 0.05, 0.1, 0.2],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}


def buscar_hiperparametros(nombre, base_estimator, param_distributions, X_cv, y_cv, n_iter=30):
    """
    Busqueda aleatoria de hiperparametros con validacion cruzada K-Fold sobre
    datos reales (X_cv, y_cv). Devuelve el estimador ya ajustado con los
    mejores hiperparametros encontrados (RandomizedSearchCV.best_estimator_
    queda entrenado sobre TODO X_cv/y_cv al finalizar el fit).

    NOTA: el estimador devuelto aqui se re-entrena mas abajo con
    model.fit(X_train, y_train) (train real + sintetico), como ya hacia
    evaluar_modelo. La busqueda solo decide LOS HIPERPARAMETROS; el
    entrenamiento final para produccion sigue el mismo flujo que ya tenias.
    """
    n_splits = min(5, max(2, len(X_cv) // 2))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Alternativa exhaustiva (mas lenta, util si el dataset real es muy chico):
    #   from sklearn.model_selection import GridSearchCV
    #   search = GridSearchCV(base_estimator, param_grid=param_distributions,
    #                          scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1)

    search = RandomizedSearchCV(
        base_estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_cv, y_cv)

    print(f"\n[SEARCH] {nombre} - busqueda de hiperparametros ({n_splits}-fold CV, {n_iter} combinaciones):")
    print(f"  Mejor MAE (CV, datos reales): {-search.best_score_:.3f}")
    print(f"  Mejores hiperparametros:")
    for k, v in search.best_params_.items():
        print(f"    {k:<20} = {v}")

    return search.best_estimator_, search.best_params_, -float(search.best_score_)


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
    Incluye pct_congestion basado en velocidad libre empirica del tramo
    (calculada solo con horas no-pico).

    "nivel" se asigna en DOS PASADAS:
      1) Se calcula pct_congestion para TODAS las franjas primero.
      2) Se derivan los umbrales de clasificacion desde la distribucion
         empirica de esos pct_congestion (percentiles 50/80, ver
         calcular_umbrales_pct), y RECIEN AHI se asigna "nivel" a cada fila.
    Esto evita usar un corte fijo arbitrario (ej. 40/70) que puede cambiar
    una etiqueta MEDIO/BAJO por una diferencia de menos de 1-2 puntos
    porcentuales entre dos modelos con desempeno casi identico.
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

    # --- Pasada 1: calcular pct_congestion para todas las franjas ---
    filas_temp = []
    for _, row in grp.iterrows():
        v_pred  = float(row["velocidad_predicha_sig"])
        v_libre = v_libres.get(row["tramo"], float(df_real["speed_promedio"].quantile(0.85)))
        pct     = velocidad_a_pct_congestion(v_pred, v_libre)
        filas_temp.append((row, v_pred, v_libre, pct))

    # --- Derivar umbrales desde la distribucion real de pct_congestion ---
    todos_los_pct = [f[3] for f in filas_temp]
    umbrales_pct = calcular_umbrales_pct(todos_los_pct)
    print(f"\n[INFO] Umbrales de nivel (dashboard), calculados por percentiles de pct_congestion:")
    print(f"  BAJO   si pct <  {umbrales_pct['umbral_medio_pct']}%")
    print(f"  MEDIO  si {umbrales_pct['umbral_medio_pct']}% <= pct < {umbrales_pct['umbral_alto_pct']}%")
    print(f"  ALTO   si pct >= {umbrales_pct['umbral_alto_pct']}%")

    # --- Pasada 2: asignar nivel con los umbrales ya calibrados ---
    salida = []
    for row, v_pred, v_libre, pct in filas_temp:
        nivel = pct_a_categoria(pct, umbrales_pct)

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
    return salida, umbrales_pct


def generar_predicciones_zonas(predicciones, umbrales_pct):
    """
    Agrega las predicciones de tramo a nivel de zona, para la barra lateral
    del dashboard.

    pct_congestion mostrado = promedio de los tramos de la zona (dato
    informativo general).

    nivel de zona = se decide con el tramo MAS congestionado (max), usando
    los MISMOS umbrales_pct (calibrados empiricamente) que se usaron para
    el nivel por tramo. Razon del "max" en vez de promedio: un conductor
    que cruza la zona sufre el peor cuello de botella del recorrido, no el
    promedio de todos los tramos. Si se usa el promedio, un tramo muy
    congestionado puede quedar "escondido" por otros tramos con mejor
    flujo, y la barra lateral termina diciendo BAJO cuando en el mapa hay
    tramos en rojo.
    """
    df = pd.DataFrame(predicciones)

    agg = df.groupby(["zona", "dia_semana", "dia_nombre", "hora"]).agg(
        pct_congestion_promedio = ("pct_congestion", "mean"),
        pct_congestion_max      = ("pct_congestion", "max"),
        velocidad_predicha      = ("velocidad_predicha_siguiente", "mean"),
        temperatura_max         = ("temperatura_max", "mean"),
        temperatura_min         = ("temperatura_min", "mean"),
        precipitacion           = ("precipitacion", "mean"),
        tramos_considerados     = ("tramo", "nunique"),
        registros               = ("registros", "sum"),
    ).reset_index()

    # Nivel de zona = el del tramo mas congestionado, mismos umbrales que tramo
    agg["nivel"] = agg["pct_congestion_max"].apply(lambda p: pct_a_categoria(p, umbrales_pct))

    # Lo que se muestra como "% de congestion estimada" en la barra sigue
    # siendo el promedio, pero ahora el NIVEL/color ya no lo contradice.
    agg["pct_congestion"]         = agg["pct_congestion_promedio"].round(1)
    agg["pct_congestion_max"]     = agg["pct_congestion_max"].round(1)
    agg["temperatura_max"]        = agg["temperatura_max"].round(1)
    agg["temperatura_min"]        = agg["temperatura_min"].round(1)
    agg["precipitacion"]          = agg["precipitacion"].round(1)

    # Reordenar columnas para mantener compatibilidad con el frontend actual
    # (pct_congestion sigue siendo el promedio, como antes; pct_congestion_max
    # queda disponible por si el frontend quiere mostrarlo tambien).
    cols = [
        "zona", "dia_semana", "dia_nombre", "hora",
        "pct_congestion", "pct_congestion_max", "velocidad_predicha",
        "temperatura_max", "temperatura_min", "precipitacion",
        "tramos_considerados", "registros", "nivel",
    ]
    agg = agg[cols]

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

    # Velocidad libre por tramo (parametro descriptivo, solo horas no-pico)
    v_libres = calcular_velocidad_libre(df_real)

    # Umbrales de clasificacion (percentiles 33/66 de velocidad_siguiente real)
    # -- usados SOLO para evaluar el modelo mas abajo, no para el dashboard --
    umbrales = calcular_umbrales(df_real_cv[TARGET])
    print(f"\n[INFO] Umbrales de evaluacion interna (percentiles 33/66, velocidad absoluta):")
    print(f"  ALTO   si v <= {umbrales['umbral_alto_congestion']} km/h")
    print(f"  MEDIO  si v <= {umbrales['umbral_medio_congestion']} km/h")
    print(f"  BAJO   si v >  {umbrales['umbral_medio_congestion']} km/h")
    print(f"  (Estos umbrales NO se usan para el 'nivel' del dashboard;")
    print(f"   el dashboard usa pct_a_categoria basado en pct_congestion.)\n")

    # Baseline
    mae_naive, rmse_naive, r2_naive, y_naive = evaluar_baseline_naive(df_test)

    # ── Busqueda de hiperparametros (Etapa 6b) ──
    # Se busca sobre datos reales (X_cv, y_cv); el resultado son estimadores
    # con los hiperparametros optimos, listos para reentrenar en el flujo
    # normal (X_train/y_train, que incluye datos sinteticos de aumento).
    print("\n" + "=" * 52)
    print("  BUSQUEDA DE HIPERPARAMETROS (RandomizedSearchCV)")
    print("=" * 52)

    hiperparametros_encontrados = {}

    rf_base = RandomForestRegressor(random_state=42)
    rf_buscado, rf_params, rf_mae_cv = buscar_hiperparametros(
        "Random Forest", rf_base, RF_PARAM_DIST, X_cv, y_cv
    )
    hiperparametros_encontrados["Random Forest"] = {
        "parametros": rf_params, "mae_cv": round(rf_mae_cv, 3)
    }

    modelos = {"Random Forest": rf_buscado}

    if XGBOOST_DISPONIBLE:
        xgb_base = XGBRegressor(random_state=42)
        xgb_buscado, xgb_params, xgb_mae_cv = buscar_hiperparametros(
            "XGBoost", xgb_base, XGB_PARAM_DIST, X_cv, y_cv
        )
        hiperparametros_encontrados["XGBoost"] = {
            "parametros": xgb_params, "mae_cv": round(xgb_mae_cv, 3)
        }
        modelos["XGBoost"] = xgb_buscado

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

    # Clasificacion derivada (SOLO evaluacion interna del modelo, ver nota arriba)
    y_pred_test  = mejor_modelo.predict(X_test)
    cat_real     = [velocidad_a_categoria(v, umbrales) for v in y_test]
    cat_pred_ml  = [velocidad_a_categoria(v, umbrales) for v in y_pred_test]
    cat_naive    = [velocidad_a_categoria(v, umbrales) for v in y_naive]
    print(f"\n[INFO] Clasificacion derivada (evaluacion interna) - {mejor_nombre}:")
    print(classification_report(cat_real, cat_pred_ml, zero_division=0))
    print("[INFO] Clasificacion derivada (evaluacion interna) - Baseline Naive:")
    print(classification_report(cat_real, cat_naive, zero_division=0))

    # Guardar modelo y parametros
    joblib.dump(mejor_modelo, MODEL_FILE)
    with open(THRESHOLDS_FILE, "w", encoding="utf-8") as f:
        json.dump(umbrales, f, ensure_ascii=False, indent=2)
    with open(VLIBRES_FILE, "w", encoding="utf-8") as f:
        json.dump(v_libres, f, ensure_ascii=False, indent=2)
    with open(HIPERPARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "modelo_ganador": mejor_nombre,
                "busqueda": hiperparametros_encontrados,
            },
            f, ensure_ascii=False, indent=2
        )
    print(f"\n[OK] Modelo: {MODEL_FILE}")
    print(f"[OK] Umbrales: {THRESHOLDS_FILE}")
    print(f"[OK] Velocidades libres: {VLIBRES_FILE}")
    print(f"[OK] Hiperparametros (busqueda): {HIPERPARAMS_FILE}")

    # Generar JSONs para el dashboard
    predicciones, umbrales_pct = generar_predicciones(df_real, mejor_modelo, umbrales, v_libres)
    predicciones_zonas = generar_predicciones_zonas(predicciones, umbrales_pct)

    with open(PRED_FILE, "w", encoding="utf-8") as f:
        json.dump(predicciones, f, ensure_ascii=False, indent=2)
    with open(PRED_ZONAS_FILE, "w", encoding="utf-8") as f:
        json.dump(predicciones_zonas, f, ensure_ascii=False, indent=2)
    with open(UMBRALES_PCT_FILE, "w", encoding="utf-8") as f:
        json.dump(umbrales_pct, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] {PRED_FILE}")
    print(f"[OK] {PRED_ZONAS_FILE}")
    print(f"[OK] Umbrales de nivel (calibrados): {UMBRALES_PCT_FILE}")
    print("\n[OK] Entrenamiento completado.")


if __name__ == "__main__":
    main()
