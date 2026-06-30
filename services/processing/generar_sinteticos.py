"""
processing/generar_sinteticos.py

Bootstrap suavizado, con separacion train/test ANTES de generar
sinteticos, para evitar fuga de datos (data leakage): ninguna fila
sintetica puede derivar de una fila que termino en el test set.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IN_FILE = BASE_DIR / "dataset.csv"
OUT_TRAIN_FILE = BASE_DIR / "dataset_train_aumentado.csv"
OUT_TEST_FILE = BASE_DIR / "dataset_test_real.csv"

N_SINTETICOS_POR_FILA = 5
FRACCION_RUIDO = 0.5
FRAC_TEST = 0.2
SEED = 42


def calcular_sigma_por_tramo(df):
    sigma_global = df["speed_promedio"].std()
    sigmas = df.groupby("tramo")["speed_promedio"].std()
    sigmas = sigmas.fillna(sigma_global)
    sigmas = sigmas.replace(0, sigma_global)
    return sigmas.to_dict()


def generar_variantes(df, sigmas, n_por_fila, fraccion_ruido, seed):
    rng = np.random.default_rng(seed)
    filas_sinteticas = []
    columnas_a_perturbar = ["speed_promedio", "velocidad_hora_anterior", "velocidad_siguiente"]

    for _, fila in df.iterrows():
        sigma = sigmas.get(fila["tramo"], df["speed_promedio"].std()) * fraccion_ruido

        for _ in range(n_por_fila):
            nueva = fila.copy()
            for col in columnas_a_perturbar:
                if pd.notna(nueva[col]):
                    ruido = rng.normal(0, sigma)
                    nueva[col] = max(0.0, nueva[col] + ruido)

            nueva["tendencia"] = nueva["speed_promedio"] - nueva["velocidad_hora_anterior"]
            nueva["promedio_movil_2"] = np.mean([nueva["speed_promedio"], nueva["velocidad_hora_anterior"]])
            if pd.isna(fila.get("promedio_movil_3")):
                nueva["promedio_movil_3"] = nueva["promedio_movil_2"]
            else:
                nueva["promedio_movil_3"] = np.mean([
                    nueva["speed_promedio"], nueva["velocidad_hora_anterior"], fila["promedio_movil_3"]
                ])

            nueva["es_sintetico"] = 1
            filas_sinteticas.append(nueva)

    return pd.DataFrame(filas_sinteticas)


def main():
    print("=" * 55)
    print("Generacion de sinteticos CON separacion train/test previa")
    print("=" * 55)

    df_real = pd.read_csv(IN_FILE)
    df_real["es_sintetico"] = 0
    print(f"[INFO] Filas reales totales: {len(df_real)}")

    # --- PRIMERO el split, usando solo datos reales ---
    df_test = df_real.sample(frac=FRAC_TEST, random_state=SEED)
    df_train_real = df_real.drop(df_test.index)
    print(f"[INFO] Train real: {len(df_train_real)}  |  Test real (apartado, intocable): {len(df_test)}")

    # --- LUEGO los sinteticos, solo a partir del train ---
    sigmas = calcular_sigma_por_tramo(df_train_real)
    print("\n[INFO] Sigma por tramo (calculado solo con TRAIN real):")
    for tramo, sigma in sigmas.items():
        print(f"  {tramo:<10} sigma={sigma:.2f} km/h (ruido = {sigma*FRACCION_RUIDO:.2f})")

    df_sint = generar_variantes(df_train_real, sigmas, N_SINTETICOS_POR_FILA, FRACCION_RUIDO, SEED)
    print(f"\n[INFO] Filas sinteticas generadas (solo de train): {len(df_sint)}")

    df_train_final = pd.concat([df_train_real, df_sint], ignore_index=True)
    df_train_final.to_csv(OUT_TRAIN_FILE, index=False)
    df_test.to_csv(OUT_TEST_FILE, index=False)

    print(f"\n[OK] Train aumentado guardado: {OUT_TRAIN_FILE} ({len(df_train_final)} filas)")
    print(f"[OK] Test real guardado:       {OUT_TEST_FILE} ({len(df_test)} filas)")
    print("\n[INFO] Comparacion de distribuciones (train real vs sintetico):")
    for col in ["speed_promedio", "velocidad_siguiente"]:
        r_mean, r_std = df_train_real[col].mean(), df_train_real[col].std()
        s_mean, s_std = df_sint[col].mean(), df_sint[col].std()
        print(f"  {col}: Real train -> media={r_mean:.2f} std={r_std:.2f} | Sintetico -> media={s_mean:.2f} std={s_std:.2f}")


if __name__ == "__main__":
    main()
