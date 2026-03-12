
"""
Alinea una secuencia temporal de una lesión (misma lesión en distintas fechas)
usando AKAZE + matching Hamming + RANSAC + transformación rígida (aprox. Euclídea).

Caso de prueba solicitado:
- Paciente: 0  (Source Directory == 'Patients\\0' en el Excel)
- Lesión: Marker_MarkText == 3
- Nombre de fichero: YYYY-MM-DD_ImageName (p.ej. 2013-10-22_FotoFinder2007_....jpg)
- Las imágenes están en: <dataset_root>/Patients/0/images/(Folder_3 o donde estén)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


# -----------------------
# Utilidades
# -----------------------
def build_filename(row: pd.Series) -> str:
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
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    mapping: dict[str, Path] = {}
    for p in patient_images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            mapping[p.name] = p
    return mapping


# -----------------------
# Alineamiento (par)
# -----------------------
def align_pair_rigid(
    ref_bgr: np.ndarray,
    dst_bgr: np.ndarray,
    *,
    out_size: tuple[int, int] = (400, 320),  # (W,H)
    akaze_threshold: float = 1e-3,
    ratio_test: float = 0.75,
    ransac_reproj_threshold: float = 3.0,
    min_good_matches: int = 12,
) -> tuple[np.ndarray, dict]:
    """
    Alinea dst -> ref con una transformación tipo Euclídea (rot+tras; sin escala idealmente)
    usando estimateAffinePartial2D + RANSAC.

    Devuelve: (dst_alineada, debug_info)
    """
    W, H = out_size
    ref = cv2.resize(ref_bgr, (W, H), interpolation=cv2.INTER_AREA)
    dst = cv2.resize(dst_bgr, (W, H), interpolation=cv2.INTER_AREA)

    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    dst_g = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    

    akaze = cv2.AKAZE_create(threshold=akaze_threshold)
    kp_ref, des_ref = akaze.detectAndCompute(ref_g, None)
    kp_dst, des_dst = akaze.detectAndCompute(dst_g, None)

    dbg = {
        "kp_ref": 0 if kp_ref is None else len(kp_ref),
        "kp_dst": 0 if kp_dst is None else len(kp_dst),
        "good_matches": 0,
        "inliers": 0,
        "M": None,
        "status": "ok",
    }

    if des_ref is None or des_dst is None or len(des_ref) < 2 or len(des_dst) < 2:
        dbg["status"] = "no_descriptors"
        return dst, dbg

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des_ref, des_dst, k=2)

    good = []
    for m, n in knn:
        if m.distance < ratio_test * n.distance:
            good.append(m)

    dbg["good_matches"] = len(good)

    if len(good) < min_good_matches:
        dbg["status"] = "too_few_matches"
        return dst, dbg

    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_dst = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #convierte indices a coordenadas

    # estimateAffinePartial2D: rot+tras (+ algo de escala; en la práctica suele salir ~1)
    M, inliers = cv2.estimateAffinePartial2D(
        pts_dst, pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        maxIters=5000,
        confidence=0.99,
        refineIters=20,
    )

    if M is None:
        dbg["status"] = "ransac_failed"
        return dst, dbg

    dbg["M"] = M.tolist()
    dbg["inliers"] = int(inliers.sum()) if inliers is not None else 0

    aligned = cv2.warpAffine(
        dst, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, dbg


# -----------------------
# Alineamiento (secuencia)
# -----------------------
def align_sequence(
    image_paths: list[Path],
    out_dir: Path,
    *,
    out_size: tuple[int, int] = (400, 320),
    akaze_threshold: float = 1e-3,
) -> list[dict]:
    """
    Alinea secuencialmente: X2->X1, X3->X2_alineada, ...
    Guarda las imágenes alineadas en out_dir y devuelve logs (debug).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logs: list[dict] = []

    # 1ª imagen: referencia
    ref_path = image_paths[0]
    ref = cv2.imread(str(ref_path))
    if ref is None:
        raise FileNotFoundError(f"No pude leer: {ref_path}")

    # guardamos la 1ª como base (solo resize)
    W, H = out_size
    ref_resized = cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA)
    out0 = out_dir / ref_path.name
    cv2.imwrite(str(out0), ref_resized)
    logs.append({"i": 1, "file": ref_path.name, "status": "reference_saved"})

    ref_current = ref_resized

    # resto
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
    ap.add_argument("--out_dir", type=str, default="aligned_out1/patient0_marktext3")
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