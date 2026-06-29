# ── services/processing/convertir_gps.py ─────────────────────────────────────
import json
from pathlib import Path

def pipeline_geometria_completa(archivo_geojson_salida):
    BASE_DIR = Path(__file__).resolve().parent.parent
    ruta_data_raiz = BASE_DIR / "ingestion" / "data"
    ruta_salida = BASE_DIR / "webapp" / "data" / archivo_geojson_salida

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    
    coordenadas_bajada = []  # Sentido Norte -> Sur (Toda la bajada de la Av. Víctor Paz)
    coordenadas_subida = []  # Sentido Sur -> Norte (Toda la subida de la Av. Víctor Paz)

    print(f"[*] Escaneando todos los lotes en: {ruta_data_raiz}")
    archivos_encontrados = sorted(list(ruta_data_raiz.rglob("*.json")))
    
    if not archivos_encontrados:
        print("[!] No se encontraron archivos JSON de ingestión.")
        return

    print(f"[*] Extrayendo coordenadas absolutas de {len(archivos_encontrados)} lotes...")

    ultimo_lat = None

    for ruta_json in archivos_encontrados:
        try:
            with open(ruta_json, "r", encoding="utf-8") as f:
                lote = json.load(f)
                if not isinstance(lote, list):
                    continue
                
                for reg in lote:
                    try:
                        lat = float(reg["lat"])
                        lng = float(reg["lng"])
                        
                        # Bounding Box de seguridad para enmarcar la Av. Víctor Paz en Tarija
                        if not (-21.535 <= lat <= -21.515) or not (-64.745 <= lng <= -64.735):
                            continue

                        punto_geojson = [lng, lat] # Estándar [Longitud, Latitud]

                        # CLASIFICACIÓN SIN FILTRADO DE RUIDO:
                        # Evaluamos la dirección del vector respecto al punto anterior para saber el sentido
                        if ultimo_lat is not None:
                            delta_lat = lat - ultimo_lat
                            
                            if delta_lat < 0:
                                # El vehículo se mueve hacia el Sur (Carril de Bajada)
                                coordenadas_bajada.append(punto_geojson)
                            else:
                                # El vehículo se mueve hacia el Norte (Carril de Subida)
                                coordenadas_subida.append(punto_geojson)

                        ultimo_lat = lat

                    except (ValueError, KeyError):
                        continue
        except Exception:
            continue

    # Estructura GeoJSON FeatureCollection con las geometrías completas
    geojson_final = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": "victor_paz_bajada_campesino",
                    "nombre": "Av. Víctor Paz (Geometría Completa - Bajada)"
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordenadas_bajada
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "victor_paz_subida_campesino",
                    "nombre": "Av. Víctor Paz (Geometría Completa - Subida)"
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordenadas_subida
                }
            }
        ]
    }

    with open(ruta_salida, "w", encoding="utf-8") as out:
        json.dump(geojson_final, out, indent=4, ensure_ascii=False)
        
    print("\n==========================================================")
    print(f"[+] ¡PROCESAMIENTO DE GEOMETRÍA COMPLETA EXITOSO!")
    print(f"[+] Archivo generado: {ruta_salida}")
    print(f"[+] Coordenadas totales Carril Bajada: {len(coordenadas_bajada)} puntos.")
    print(f"[+] Coordenadas totales Carril Subida: {len(coordenadas_subida)} puntos.")
    print("==========================================================")

if __name__ == "__main__":
    pipeline_geometria_completa("vias.geojson")
