"""
ml/train.py
Entrena y compara Random Forest vs XGBoost para predecir
el nivel de congestión (BAJO / MEDIO / ALTO) dado:
tramo + zona + dia_semana + hora + clima (temperatura, precipitación).

Guarda el mejor modelo y dos JSON para el dashboard:
  - predicciones.json         -> una predicción por TRAMO (8 segmentos del mapa)
  - predicciones_zonas.json   -> resumen agregado por ZONA (panel de 2 zonas)
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

try:
    from xgboost import XGBClassifier
    XGBOOST_DISPONIBLE = True
except ImportError:
    XGBOOST_DISPONIBLE = False
    print("[WARN] XGBoost no instalado. Solo se usará Random Forest.")

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATASET      = BASE_DIR.parent / "processing" / "dataset.csv"
MODELS_DIR   = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_FILE        = MODELS_DIR / "congestion_model.pkl"
ENCODER_FILE       = MODELS_DIR / "label_encoder.pkl"
PRED_FILE          = BASE_DIR   / "predicciones.json"
PRED_ZONAS_FILE    = BASE_DIR   / "predicciones_zonas.json"

# Features: se agregan tramo_cod (segmentación 8 puntos) y clima.
# zona_cod se mantiene por si quieres inspeccionar su importancia,
# aunque está fuertemente correlacionada con tramo_cod (cada tramo
# pertenece a una sola zona).
FEATURES = [
    "tramo_cod",
    "zona_cod",
    "dia_semana",
    "hora",
    "es_hora_pico",
    "total_registros",
    "dias_observados",
    "temperatura_max",
    "temperatura_min",
    "precipitacion",
]
TARGET = "nivel_congestion"

# Umbrales para re-bucketizar el % de congestión agregado por zona
# (deben coincidir con nivel_congestion() de processing/build_dataset.py)
UMBRAL_ALTO = 70.0
UMBRAL_MEDIO = 40.0


# ── Carga ─────────────────────────────────────────────────────────────────────
def cargar_dataset():
    df = pd.read_csv(DATASET)
    print(f"[INFO] Dataset cargado: {len(df)} franjas")

    # Clima puede venir con NaN si algún día no tiene registro SENAMHI
    nan_clima = df[["temperatura_max", "temperatura_min", "precipitacion"]].isna().sum()
    if nan_clima.sum() > 0:
        print(f"[WARN] Valores NaN en clima antes de imputar:\n{nan_clima}")
        for col in ["temperatura_max", "temperatura_min", "precipitacion"]:
            df[col] = df[col].fillna(df[col].mean())

    print(f"[INFO] Distribución:\n{df[TARGET].value_counts()}\n")
    if df[TARGET].nunique() < 2:
        raise ValueError("Menos de 2 clases. Necesitas más datos.")
    return df


# ── Evaluación de un modelo ───────────────────────────────────────────────────
def evaluar_modelo(nombre, model, X_train, X_test, y_train, y_test, le, X_full, y_full):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=" * 52)
    print(f"  {nombre}")
    print("=" * 52)

    print(classification_report(y_test, y_pred,
                                 target_names=le.classes_,
                                 zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index   = [f"Real: {c}"  for c in le.classes_],
        columns = [f"Pred: {c}" for c in le.classes_],
    )
    print("  Matriz de confusión:")
    print(cm_df.to_string())

    f1_cv = 0.0
    if len(X_full) >= 6:
        n_splits = min(5, len(X_full) // 2)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_full, y_full,
                                  cv=cv, scoring="f1_weighted")
        f1_cv = scores.mean()
        print(f"\n  CV F1 ({n_splits}-fold): {f1_cv:.4f} ± {scores.std():.4f}")
    else:
        f1_cv = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        print(f"\n  F1 weighted (test): {f1_cv:.4f}  [pocas muestras para CV]")

    if hasattr(model, "feature_importances_"):
        print("\n  Importancia de features:")
        for feat, imp in sorted(zip(FEATURES, model.feature_importances_),
                                 key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"  {feat:<25} {bar} {imp:.4f}")

    return f1_cv


# ── Entrenamiento y comparación ───────────────────────────────────────────────
def entrenar(df):
    X = df[FEATURES]
    y = df[TARGET]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"[INFO] Clases: {list(le.classes_)}")

    if len(df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
    else:
        print("[WARN] Pocas franjas — usando todo para train y test.")
        X_train, X_test, y_train, y_test = X, X, y_enc, y_enc

    modelos = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
    }

    if XGBOOST_DISPONIBLE:
        modelos["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
        )

    resultados = {}
    modelos_entrenados = {}

    for nombre, model in modelos.items():
        f1 = evaluar_modelo(
            nombre, model,
            X_train, X_test, y_train, y_test,
            le, X, y_enc
        )
        resultados[nombre] = f1
        modelos_entrenados[nombre] = model

    print("\n" + "=" * 52)
    print("  COMPARATIVA FINAL")
    print("=" * 52)
    print(f"  {'Modelo':<20} {'F1 (weighted)':>14}")
    print(f"  {'-'*36}")
    for nombre, f1 in resultados.items():
        print(f"  {nombre:<20} {f1:>14.4f}")

    mejor_nombre = max(resultados, key=resultados.get)
    mejor_modelo = modelos_entrenados[mejor_nombre]
    print(f"\n  ✓ Mejor modelo: {mejor_nombre}  (F1={resultados[mejor_nombre]:.4f})")

    return mejor_modelo, mejor_nombre, le


# ── Predicciones por tramo (para el mapa segmentado) ─────────────────────────
def generar_predicciones(df, model, le):
    X       = df[FEATURES]
    probs   = model.predict_proba(X)
    preds   = model.predict(X)
    niveles = le.inverse_transform(preds)

    dias_nombre = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]

    salida = []
    for i, row in df.reset_index(drop=True).iterrows():
        prob_dict = {cls: round(float(p), 3)
                     for cls, p in zip(le.classes_, probs[i])}
        salida.append({
            "tramo":           row["tramo"],
            "zona":            row["zona"],
            "dia_semana":      int(row["dia_semana"]),
            "dia_nombre":      dias_nombre[int(row["dia_semana"])],
            "hora":            int(row["hora"]),
            "nivel":           niveles[i],
            "pct_congestion":  round(float(row["pct_congestion"]) * 100, 1),
            "probabilidades":  prob_dict,
            "registros":       int(row["total_registros"]),
            "dias_observados": int(row["dias_observados"]),
            "temperatura_max": round(float(row["temperatura_max"]), 1),
            "temperatura_min": round(float(row["temperatura_min"]), 1),
            "precipitacion":   round(float(row["precipitacion"]), 1),
        })

    with open(PRED_FILE, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Predicciones por tramo guardadas: {PRED_FILE}  ({len(salida)} franjas)")
    return salida


# ── Resumen agregado por zona (para el panel de las 2 zonas) ─────────────────
def generar_resumen_zonas(salida):
    """
    Agrupa las predicciones por tramo en (zona, dia_semana, hora),
    promediando el % de congestión de los tramos de cada zona, y
    re-bucketizando el nivel con los mismos umbrales del dataset.
    """
    df = pd.DataFrame(salida)

    agg = df.groupby(["zona", "dia_semana", "dia_nombre", "hora"]).agg(
        pct_congestion=("pct_congestion", "mean"),
        temperatura_max=("temperatura_max", "mean"),
        temperatura_min=("temperatura_min", "mean"),
        precipitacion=("precipitacion", "mean"),
        tramos_considerados=("tramo", "nunique"),
        registros=("registros", "sum"),
    ).reset_index()

    def bucketizar(pct):
        if pct >= UMBRAL_ALTO:
            return "ALTO"
        elif pct >= UMBRAL_MEDIO:
            return "MEDIO"
        else:
            return "BAJO"

    agg["nivel"] = agg["pct_congestion"].apply(bucketizar)
    agg["pct_congestion"] = agg["pct_congestion"].round(1)
    agg["temperatura_max"] = agg["temperatura_max"].round(1)
    agg["temperatura_min"] = agg["temperatura_min"].round(1)
    agg["precipitacion"] = agg["precipitacion"].round(1)

    salida_zonas = agg.to_dict(orient="records")

    with open(PRED_ZONAS_FILE, "w", encoding="utf-8") as f:
        json.dump(salida_zonas, f, ensure_ascii=False, indent=2)

    print(f"[OK] Resumen por zona guardado: {PRED_ZONAS_FILE}  ({len(salida_zonas)} franjas)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 52)
    print("train.py — comparación Random Forest vs XGBoost")
    print("=" * 52)

    df = cargar_dataset()
    mejor_modelo, mejor_nombre, le = entrenar(df)

    joblib.dump(mejor_modelo, MODEL_FILE)
    joblib.dump(le,           ENCODER_FILE)
    print(f"\n[OK] Modelo guardado  ({mejor_nombre}): {MODEL_FILE}")
    print(f"[OK] Encoder guardado: {ENCODER_FILE}")

    salida = generar_predicciones(df, mejor_modelo, le)
    generar_resumen_zonas(salida)

    print("\n[OK] Entrenamiento completado.")


if __name__ == "__main__":
    main()
