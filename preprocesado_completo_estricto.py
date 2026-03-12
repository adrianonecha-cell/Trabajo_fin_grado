#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# -----------------------
# Utilidades
# -----------------------
def build_filename(row: pd.Series) -> str:
    """YYYY-MM-DD_ImageName"""
    y = int(row["ShootingDate_Year"])
    m = int(row["ShootingDate_Month"])
    d = int(row["ShootingDate_Day"])
    return f"{y:04d}-{m:02d}-{d:02d}_{row['ImageName']}"


def sort_cols() -> list[str]:
    return [
        "ShootingDate_Year", "ShootingDate_Month", "ShootingDate_Day",
        "ShootingDate_Hour", "ShootingDate_Minute", "ShootingDate_Second",
    ]


def parse_patient_id(source_dir: str) -> str:
    """
    Source Directory en el Excel viene como 'Patients\\0' (o parecido).
    Devuelve '0'.
    """
    s = str(source_dir).replace("\\", "/")
    parts = [p for p in s.split("/") if p]
    return parts[-1] if parts else ""


def to_bool(x) -> bool:
    """Convierte valores tipo 0/1, True/False, 'TRUE'/'FALSE' a bool de forma robusta."""
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer, float, np.floating)):
        return bool(int(x))
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def folder_name(mark_text: int, deactivated: bool) -> str:
    """
    Según tu organización:
    - Marker_Deactivated = FALSE -> Folder_{n}
    - Marker_Deactivated = TRUE  -> Folder_{n}_ext
    """
    base = f"Folder_{int(mark_text)}"
    return f"{base}_ext" if deactivated else base


def index_images_in_folder(folder: Path) -> dict[str, Path]:
    """Mapping {filename: fullpath} dentro de una carpeta (no recursivo)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    mapping: dict[str, Path] = {}
    if not folder.is_dir():
        return mapping
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            mapping[p.name] = p
    return mapping


# -----------------------
# Transformación rígida (rot + tras, sin escala)
# -----------------------
def rigid_from_points(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """
    Estima transformación rígida 2D (R,t): dst ≈ R*src + t
    src_xy, dst_xy: arrays (N,2)
    Devuelve M (2x3) para cv2.warpAffine.
    """
    src_mean = src_xy.mean(axis=0)
    dst_mean = dst_xy.mean(axis=0)

    src_c = src_xy - src_mean
    dst_c = dst_xy - dst_mean

    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Evitar reflexión (det(R) debe ser +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean - (R @ src_mean)
    M = np.hstack([R, t.reshape(2, 1)]).astype(np.float32)
    return M


# -----------------------
# Alineamiento (par) - ESTRICTO
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
    Alinea dst -> ref con transformación rígida pura (rot+tras, sin escala):
    - AKAZE (keypoints)
    - BFMatcher Hamming + ratio test
    - RANSAC para inliers (estimateAffinePartial2D)
    - rigid_from_points sobre inliers
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

    # RANSAC para obtener inliers (máscara robusta)
    M_aff, inliers = cv2.estimateAffinePartial2D(
        pts_dst, pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        maxIters=5000,
        confidence=0.99,
        refineIters=20,
    )

    if M_aff is None:
        dbg["status"] = "ransac_failed"
        return dst, dbg

    if inliers is not None:
        mask = inliers.ravel().astype(bool)
    else:
        mask = np.ones((len(good),), dtype=bool)

    if mask.sum() < 3:
        mask = np.ones((len(good),), dtype=bool)

    dbg["inliers"] = int(mask.sum())

    src_in = pts_dst[mask].reshape(-1, 2)
    dst_in = pts_ref[mask].reshape(-1, 2)

    M = rigid_from_points(src_in, dst_in)
    dbg["M"] = M.tolist()

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
    out_dir.mkdir(parents=True, exist_ok=True)
    logs: list[dict] = []

    # 1ª imagen: referencia
    ref_path = image_paths[0]
    ref = cv2.imread(str(ref_path))
    if ref is None:
        raise FileNotFoundError(f"No pude leer: {ref_path}")

    W, H = out_size
    ref_resized = cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(out_dir / ref_path.name), ref_resized)
    logs.append({"i": 1, "file": ref_path.name, "status": "reference_saved"})

    ref_current = ref_resized

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

        ref_current = aligned

    return logs


# -----------------------
# Procesado global (TODO)
# -----------------------
def process_all(
    in_root: Path,
    out_root: Path,
    excel_name: str,
    *,
    out_size: tuple[int, int],
    akaze_threshold: float,
    save_global_log: bool = True,
) -> None:
    excel_path = Path(excel_name)
    if not excel_path.is_file():
        excel_path = in_root / excel_name
    if not excel_path.is_file():
        raise FileNotFoundError(f"No encuentro el Excel: {excel_path}")

    df = pd.read_excel(excel_path)

    # Asegurar columnas clave
    needed = [
        "Source Directory", "Marker_MarkText", "Marker_Deactivated",
        "ShootingDate_Year", "ShootingDate_Month", "ShootingDate_Day",
        "ShootingDate_Hour", "ShootingDate_Minute", "ShootingDate_Second",
        "ImageName",
    ]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Falta columna en Excel: {c}")

    # Crear patient_id desde Source Directory
    df = df.dropna(subset=["Source Directory"]).copy()
    df["patient_id"] = df["Source Directory"].apply(parse_patient_id)

    # Quitamos filas sin MarkText (no se puede mapear a carpeta Folder_n)
    df = df.dropna(subset=["Marker_MarkText"]).copy()

    # Convertir MarkText a int (muchas veces viene como float)
    df["marktext_int"] = df["Marker_MarkText"].astype(int)

    # Convertir deactivated a bool robusto
    df["deactivated_bool"] = df["Marker_Deactivated"].apply(to_bool)

    # Log global
    global_rows: list[dict] = []

    # Agrupar por paciente + número de lesión
    grouped = df.groupby(["patient_id", "marktext_int"], sort=True)

    total_groups = len(grouped)
    print(f"Grupos a procesar (paciente+lesión): {total_groups}")

    for gi, ((patient_id, marktext), g) in enumerate(grouped, start=1):
        # carpeta de lesión (ext si ANY True)
        deact = bool(g["deactivated_bool"].any())
        fold = folder_name(marktext, deact)

        in_folder = in_root / "Patients" / str(patient_id) / "images" / fold
        out_folder = out_root / "Patients" / str(patient_id) / "images" / fold
        out_folder.mkdir(parents=True, exist_ok=True)

        if not in_folder.is_dir():
            # No existe carpeta: lo dejamos registrado y seguimos
            global_rows.append({
                "patient_id": patient_id,
                "marktext": marktext,
                "folder": fold,
                "status": "missing_input_folder",
                "in_folder": str(in_folder),
            })
            continue

        # Orden temporal dentro del grupo
        g2 = g.sort_values(sort_cols()).copy()
        g2["BuiltFile"] = g2.apply(build_filename, axis=1)

        wanted = g2["BuiltFile"].tolist()

        # Indexar solo esa carpeta (rápido)
        name2path = index_images_in_folder(in_folder)

        # Resolver rutas en el orden del Excel
        image_paths = []
        missing_files = 0
        for w in wanted:
            p = name2path.get(w)
            if p is None:
                missing_files += 1
            else:
                image_paths.append(p)

        if len(image_paths) == 0:
            global_rows.append({
                "patient_id": patient_id,
                "marktext": marktext,
                "folder": fold,
                "status": "no_images_found_in_folder",
                "missing_files": missing_files,
                "in_folder": str(in_folder),
            })
            continue

        # Si solo hay 1 imagen: la “preprocesamos” guardándola redimensionada (sin alineación)
        if len(image_paths) == 1:
            img = cv2.imread(str(image_paths[0]))
            if img is None:
                global_rows.append({
                    "patient_id": patient_id,
                    "marktext": marktext,
                    "folder": fold,
                    "status": "single_image_read_failed",
                    "file": image_paths[0].name,
                })
                continue

            W, H = out_size
            img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_folder / image_paths[0].name), img_resized)

            global_rows.append({
                "patient_id": patient_id,
                "marktext": marktext,
                "folder": fold,
                "status": "single_image_saved_resized",
                "file": image_paths[0].name,
                "missing_files": missing_files,
            })
            continue

        # Alinear secuencia completa (estricto)
        try:
            logs = align_sequence(
                image_paths,
                out_dir=out_folder,
                out_size=out_size,
                akaze_threshold=akaze_threshold,
            )

            # Guardar log por lesión
            pd.DataFrame(logs).to_csv(out_folder / "alignment_log.csv", index=False)

            # Resumen global
            ok = sum(1 for r in logs if r.get("status") == "ok")
            global_rows.append({
                "patient_id": patient_id,
                "marktext": marktext,
                "folder": fold,
                "status": "processed",
                "n_images_excel": len(wanted),
                "n_images_found": len(image_paths),
                "missing_files": missing_files,
                "n_ok_pairs": ok,
                "log_path": str(out_folder / "alignment_log.csv"),
            })

        except Exception as e:
            global_rows.append({
                "patient_id": patient_id,
                "marktext": marktext,
                "folder": fold,
                "status": "exception",
                "error": repr(e),
            })

        if gi % 50 == 0 or gi == total_groups:
            print(f"[{gi}/{total_groups}] procesados...")

    # Guardar log global
    if save_global_log:
        out_root.mkdir(parents=True, exist_ok=True)
        log_dir = out_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(global_rows).to_csv(log_dir / "preprocess_global_log.csv", index=False)
        print(f"Log global: {log_dir / 'preprocess_global_log.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, default="./data", help="Directorio de entrada (data).")
    ap.add_argument("--out_root", type=str, default="./data_processed", help="Directorio de salida (data_processed).")
    ap.add_argument("--excel", type=str, default="Combined_Database_Unified.xlsx", help="Excel maestro.")
    ap.add_argument("--w", type=int, default=400, help="Ancho resize.")
    ap.add_argument("--h", type=int, default=320, help="Alto resize.")
    ap.add_argument("--akaze_threshold", type=float, default=1e-5, help="Umbral AKAZE (más bajo -> más keypoints).")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    process_all(
        in_root=in_root,
        out_root=out_root,
        excel_name=args.excel,
        out_size=(args.w, args.h),
        akaze_threshold=args.akaze_threshold,
        save_global_log=True,
    )


if __name__ == "__main__":
    main()