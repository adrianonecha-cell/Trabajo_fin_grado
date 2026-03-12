"""
Caso de prueba solicitado:
- Paciente: 0  (Source Directory == 'Patients\\0' en el Excel)
- Lesión: Marker_MarkText == 3
- Nombre de fichero: YYYY-MM-DD_ImageName (p.ej. 2013-10-22_FotoFinder2007_....jpg)
- Las imágenes están en: <dataset_root>/Patients/0/images/(Folder_3 o donde estén)

Alinea una secuencia temporal de una lesión (misma lesión en distintas fechas)
usando AKAZE + matching Hamming + RANSAC + transformación rígida (Euclídea: rot+tras, sin escala).
"""

from __future__ import annotations #perimte usar anotaciones de tipos más modernas (Python 3.10+)

import argparse 
from pathlib import Path #para manejo de rutas de archivos

import cv2 #leeer y procesar imágenes
import numpy as np
import pandas as pd


# -----------------------
# Utilidades
# -----------------------
def build_filename(row: pd.Series) -> str: #entrada: fila de DataFrame, salida: nombre de archivo formateado
    y = int(row["ShootingDate_Year"])
    m = int(row["ShootingDate_Month"])
    d = int(row["ShootingDate_Day"])
    return f"{y:04d}-{m:02d}-{d:02d}_{row['ImageName']}"


def sort_cols() -> list[str]:
    return [
        "ShootingDate_Year", "ShootingDate_Month", "ShootingDate_Day",
        "ShootingDate_Hour", "ShootingDate_Minute", "ShootingDate_Second",
    ]


def index_all_images(patient_images_dir: Path) -> dict[str, Path]: 
    """
    Indexa todas las imágenes debajo de patient_images_dir (recursivo) y crea
    un mapping {nombre_de_archivo: ruta_completa}. 
    A través de una "llave" (nombre de archivo) se puede acceder a la ruta completa.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    mapping: dict[str, Path] = {}
    for p in patient_images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            mapping[p.name] = p
    return mapping


# -----------------------
# Transformación rígida (rot + tras, sin escala)
# -----------------------
def rigid_from_points(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray: #src_xy son los puntos que quiero mover, dst_xy son los puntos a los que quiero llegar. Devuelve la matriz de transformación rígida (2x3) para cv2.warpAffine.
    """
    Estima transformación rígida 2D (R,t) tal que: dst ≈ R*src + t
    src_xy, dst_xy: arrays (N,2)
    R es una matriz de rotación 2x2, t es un vector de traslación 2x1.
    Se basa en el método de Procrustes (SVD) para encontrar la mejor rotación, y luego calcula la traslación a partir de las medias.
    Devuelve M (2x3) para cv2.warpAffine.

    """
    src_mean = src_xy.mean(axis=0) #hacer la media por columa, sale una (1,2)
    dst_mean = dst_xy.mean(axis=0)

    src_c = src_xy - src_mean #se centran los puntos (al rededor del 0,0). 
    #Sirve para separar la parte de traslación (t) de la rotación (R). La traslación se calcula a partir de las medias, y la rotación se calcula a partir de los puntos centrados.
    dst_c = dst_xy - dst_mean

    H = src_c.T @ dst_c #matriz de covarianza cruzada entre src_c y dst_c. Es una matriz 2x2 que captura cómo se relacionan los puntos centrados de src y dst. Se usa para calcular la rotación. Método de Procrustes
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T #construir la matriz de rotación a partir de la SVD. R es una matriz de rotación 2x2 que alinea src_c con dst_c. Método de Procrustes
    #como de torcida está una imagen respecto a la otra. Si R es la identidad, no hay rotación. Si R es una rotación de 90 grados, entonces las imágenes están perpendiculares, etc.

    # Evitar reflexión (det(R) debe ser +1). Significa que no queremos que sea un espejo, forzamos a que sea una rotación pura.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean - (R @ src_mean) #una vez sabemos lo que hay que rotar, calculamos la traslación necesaria para alinear las medias.
    M = np.hstack([R, t.reshape(2, 1)]).astype(np.float32) #une la rotación y la traslación en una sola matriz 2x3 para cv2.warpAffine.
    return M


# -----------------------
# Alineamiento (par)
# -----------------------
def align_pair_rigid(
    ref_bgr: np.ndarray, #imagen referencia
    dst_bgr: np.ndarray, #imagen que queremos alinear a la referencia
    *,
    out_size: tuple[int, int] = (400, 320),  # (W,H)
    akaze_threshold: float = 1e-3, #keypoints para AKAZE, cuanto más bajo más keypoints
    ratio_test: float = 0.75, #para filtrar matches con el test de Lowe (distancia del mejor match vs segundo mejor match)
    ransac_reproj_threshold: float = 3.0, #tolerancia de reporyección
    min_good_matches: int = 12, #número mínimo de mataches buenos 
) -> tuple[np.ndarray, dict]: #imagen alineada y debug info
    """
    Alinea dst -> ref con una transformación tipo Euclídea (rot+tras; sin escala)
    usando:
      - AKAZE para keypoints
      - Hamming (BFMatcher) para matching
      - RANSAC (estimateAffinePartial2D) para inliers
      - Estimación rígida pura (SVD/Procrustes) sobre inliers

    Devuelve: (dst_alineada, debug_info)
    """
    W, H = out_size
    ref = cv2.resize(ref_bgr, (W, H), interpolation=cv2.INTER_AREA)
    dst = cv2.resize(dst_bgr, (W, H), interpolation=cv2.INTER_AREA) #asegurar que ambas imágenes tengan mismo tamaño. Inter_area sirve para reducir tamaño sin perder mucha calidad.

    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) #AKAZE funciona mejor en imágenes en escala de grises, así que convertimos las imágenes a escala de grises.
    dst_g = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create(threshold=akaze_threshold) #algoritmo de detección de características AKAZE, con un umbral para controlar la cantidad de keypoints detectados. Cuanto más bajo el umbral, más keypoints se detectan (y viceversa).
    kp_ref, des_ref = akaze.detectAndCompute(ref_g, None) #un keypoint es un punto de interés en la imagen (p.ej. una esquina, un borde, etc.) que es estable y distintivo. Un descriptor es un vector que describe el entorno local alrededor de cada keypoint, y se usa para encontrar correspondencias entre las dos imágenes.
    kp_dst, des_dst = akaze.detectAndCompute(dst_g, None)#lista de keypoints y descriptores

    dbg = {
        "kp_ref": 0 if kp_ref is None else len(kp_ref),
        "kp_dst": 0 if kp_dst is None else len(kp_dst),
        "good_matches": 0,
        "inliers": 0, 
        "M": None,
        "status": "ok",
        #se usa para debug y análisis posterior, para saber cuántos keypoints se detectaron en cada imagen, cuántos matches buenos se encontraron, cuántos inliers quedaron después de RANSAC, cuál fue la matriz de transformación estimada, y si hubo algún problema (p.ej. no se detectaron descriptores, no hubo suficientes matches, RANSAC falló, etc.)
    }

    if des_ref is None or des_dst is None or len(des_ref) < 2 or len(des_dst) < 2:
        dbg["status"] = "no_descriptors"
        return dst, dbg #si no hay descriptores no se pueden alinear, return y cambio de status

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)#algoritmo que compara cada descriptor de la referencia con los descriptores del destino. Se queda con los mejores a traves de la distancia Hamming (bits direrentes)
    knn = bf.knnMatch(des_ref, des_dst, k=2) #para cada descriptor de la referencia, encuentra los 2 mejores matches en el destino. Esto se hace para aplicar el ratio test de Lowe, que ayuda a filtrar matches ambiguos.
    #la salida knn es una lista de listas de matches, donde cada sublista contiene los 2 mejores matches para cada descriptor de la referencia.

    good = []
    for m, n in knn:
        if m.distance < ratio_test * n.distance:
            good.append(m) #aplicar el ratio test de Lowe: un match se considera bueno si la distancia del mejor match (m) es significativamente menor que la distancia del segundo mejor match (n). Esto ayuda a eliminar matches ambiguos que podrían no ser fiables para la alineación.

    dbg["good_matches"] = len(good)

    if len(good) < min_good_matches:
        dbg["status"] = "too_few_matches"
        return dst, dbg #si no hay suficientes matches buenos, no se puede alinear de forma fiable, return y cambio el status

    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_dst = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #convierte los indices de la referencia/destino a coordenadas de los keypoints

    # 1) RANSAC para obtener inliers (usamos la función para la máscara robusta)
    M_aff, inliers = cv2.estimateAffinePartial2D(
        pts_dst, pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        maxIters=5000,
        confidence=0.99,
        refineIters=20,
    )#devuelve una matriz de transformación afín (que puede incluir rotación, traslación y algo de escala) y una máscara de inliers que indica qué matches se consideran inliers según RANSAC. Se usa para filtrar los matches buenos y quedarse solo con los que son consistentes con una transformación afín robusta.
    #un inliner es un match que se ajusta bien a la transformación estimada, mientras que un outlier es un match que no se ajusta bien y probablemente sea incorrecto. RANSAC ayuda a identificar y eliminar los outliers para mejorar la calidad de la alineación.
    if M_aff is None:
        dbg["status"] = "ransac_failed"
        return dst, dbg #si rasnac falla, return y cambio el status

    # 2) Estimar transformación rígida pura (sin escala) sobre inliers
    if inliers is not None:
        mask = inliers.ravel().astype(bool) #transforma los inliners a una máscara boolena para quedarnos con los buenos
    else:
        mask = np.ones((len(good),), dtype=bool)

    # Si por algún motivo hay muy pocos inliers, usamos todos los good matches
    if mask.sum() < 3:
        mask = np.ones((len(good),), dtype=bool)

    dbg["inliers"] = int(mask.sum())

    src_in = pts_dst[mask].reshape(-1, 2)#coordenadas de los puntos del destino que son inliers
    dst_in = pts_ref[mask].reshape(-1, 2)

    M = rigid_from_points(src_in, dst_in) #ahora si, obtenemos la transformación rígida pura (rot+tras) a partir de los puntos inliers. Esto nos da una matriz de transformación que alinea el destino con la referencia usando solo rotación y traslación, sin permitir escala. (2x3)

    dbg["M"] = M.tolist() #guardamos la matriz de transformación en el debug info para análisis posterior. La convertimos a lista para que sea serializable (JSON, CSV, etc.)

    aligned = cv2.warpAffine(
        dst, M, (W, H), #destino, matriz de transformación, tamaño de salida
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    ) #crea la imagen alineada aplicando la transformación rígida al destino. Se usa cv2.warpAffine con la matriz M, el tamaño de salida (W,H), interpolación lineal, y un borde
    return aligned, dbg


# -----------------------
# Alineamiento (secuencia)
# -----------------------
def align_sequence(
    image_paths: list[Path], #lista de rutas a las imágenes que queremos alinear, en orden temporal (según el Excel). La primera imagen se toma como referencia, y cada imagen siguiente se alinea secuencialmente a la anterior alineada.
    out_dir: Path, #carpeta donde se guardarán las imágenes alineadas
    *,
    out_size: tuple[int, int] = (400, 320),
    akaze_threshold: float = 1e-3,
) -> list[dict]: #devuelve logs
    """
    Alinea secuencialmente: X2->X1, X3->X2_alineada, ...
    Guarda las imágenes alineadas en out_dir y devuelve logs (debug).
    """
    out_dir.mkdir(parents=True, exist_ok=True) #crear carpetas 
    logs: list[dict] = []

    # 1ª imagen: referencia
    ref_path = image_paths[0]
    ref = cv2.imread(str(ref_path)) #leer la primera imagen, se toma como referencia
    if ref is None:
        raise FileNotFoundError(f"No pude leer: {ref_path}")

    # guardamos la 1ª como base (solo resize)
    W, H = out_size
    ref_resized = cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA)
    out0 = out_dir / ref_path.name #construye la primera imagen redimensionada
    cv2.imwrite(str(out0), ref_resized) #guardamos la primera imagen redimensionada como base para las siguientes alineaciones. No se alinea porque es la referencia, pero se redimensiona para que todas las imágenes tengan el mismo tamaño.
    logs.append({"i": 1, "file": ref_path.name, "status": "reference_saved"})

    ref_current = ref_resized

    # para el resto de imagenes las va añadiendo secuencialmente, alineandolas previamente
    for idx, p in enumerate(image_paths[1:], start=2):
        dst = cv2.imread(str(p))
        if dst is None:
            logs.append({"i": idx, "file": p.name, "status": "read_failed"})
            continue

        aligned, dbg = align_pair_rigid(
            ref_current, dst,
            out_size=out_size,
            akaze_threshold=akaze_threshold,
        )

        cv2.imwrite(str(out_dir / p.name), aligned)

        dbg_row = {"i": idx, "file": p.name}
        dbg_row.update(dbg)
        logs.append(dbg_row)

        # actualización secuencial
        ref_current = aligned

    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_root",
        type=str,
        default="./data",
        help="Ruta al dataset (por defecto ./data)."
    )
    ap.add_argument("--excel", type=str, default="Combined_Database_Unified.xlsx",
                    help="Nombre (o ruta) del Excel maestro.")
    ap.add_argument("--patient_id", type=int, default=0)
    ap.add_argument("--marker_marktext", type=int, default=3)
    ap.add_argument("--out_dir", type=str, default="aligned_out2/patient0_marktext3")
    ap.add_argument("--images_subdir", type=str, default="Patients/0/images",
                    help="Subcarpeta donde están las imágenes del paciente (por defecto Patients/0/images).")
    ap.add_argument("--akaze_threshold", type=float, default=1e-5)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    excel_path = Path(args.excel)
    if not excel_path.is_file():
        excel_path = dataset_root / args.excel
    if not excel_path.is_file():
        raise FileNotFoundError(f"No encuentro el Excel: {excel_path}")

    # 1) leer excel
    df = pd.read_excel(excel_path)

    # 2) filtrar paciente y lesión
    src_dir_key = f"Patients\\{args.patient_id}"  # como viene en el Excel
    df_p = df[df["Source Directory"] == src_dir_key].copy()

    df_l = df_p[df_p["Marker_MarkText"] == float(args.marker_marktext)].copy()
    if df_l.empty:
        raise ValueError(f"No hay filas para Source Directory={src_dir_key} y Marker_MarkText={args.marker_marktext}")

    df_l = df_l.sort_values(sort_cols())
    df_l["BuiltFile"] = df_l.apply(build_filename, axis=1)

    # 3) indexar imágenes del paciente (para resolver rutas)
    patient_images_dir = dataset_root / args.images_subdir
    if not patient_images_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de imágenes: {patient_images_dir}")

    name2path = index_all_images(patient_images_dir)

    # 4) resolver rutas de los ficheros en orden temporal
    wanted = df_l["BuiltFile"].tolist()
    missing = [w for w in wanted if w not in name2path]
    if missing:
        print("⚠️ No encontré estos ficheros dentro de images/ (quizá estén en otra subcarpeta):")
        for m in missing:
            print("  -", m)
        print("\nSigo con los que sí he encontrado.\n")

    image_paths = [name2path[w] for w in wanted if w in name2path]
    if len(image_paths) < 2:
        raise RuntimeError("Necesito al menos 2 imágenes encontradas para alinear.")

    # 5) alinear y guardar
    out_dir = dataset_root / args.out_dir
    logs = align_sequence(
        image_paths,
        out_dir=out_dir,
        akaze_threshold=args.akaze_threshold,
    )

    # 6) guardar logs
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(out_dir / "alignment_log.csv", index=False)

    print(f"✅ Guardadas {len(image_paths)} imágenes (alineadas) en: {out_dir}")
    print(f"📝 Log: {out_dir / 'alignment_log.csv'}")
    print("\nOrden (según Excel):")
    for p in image_paths:
        print(" -", p.name)


if __name__ == "__main__":
    main()
