from __future__ import annotations
import Metadata.utils as ut
import numpy as np
import pandas as pd
from Metadata.encode import encode_slide_wsi
import os, shutil, torch, time, tempfile, glob
from Metadata.process import process_manifest, process_clinical

                  
TARGET_MAG     = 20 # Magnification factor
TILE_SIZE      = 512 # Image block size
STEP           = 512 # pitch
MIN_TISSUE     = 0.40 # Minimum organisational ratio
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Device

# Tool function for exporting NPZ files for each patient
def export_npz_for_patient(
    manifest_txt: str,
    base_dir: str,
    out_npz_dir: str,
    time_event: dict,
    clinical_features_map: dict,
    model,
    vision_preprocess,
    only_diagnostic: bool = True,
    target_mag: int = 10,
    tile_size: int = 384,
    step: int = 384,
    min_tissue: float = 0.2,
    device: str = "cuda",
):
    """
    Export NPZ files for each patient based on the manifest and clinical data.
    
    Parameters:
        manifest_txt: Path to the manifest text file.
        clinical_json: Path to the clinical JSON file.
        base_dir: Base directory where WSI files are located.
        out_npz_dir: Output directory for NPZ files.
        model: Pre-loaded MUSK model for feature extraction.
        vision_preprocess: Preprocessing function for vision model.
        tokenizer: Tokenizer for text processing (optional).
        endpoint: Survival endpoint ("PFS" or "OS").
        only_diagnostic: Whether to only include diagnostic slides.
        target_mag: Target magnification for WSI processing.
        tile_size: Tile size for patch extraction.
        step: Step size for patch extraction.
        min_tissue: Minimum tissue proportion for patch selection.
        device: Device to run the model on ("cuda" or "cpu").
    
    Returns: summary: DataFrame containing the processing summary for each patient.
    """

    os.makedirs(out_npz_dir, exist_ok=True)

    df_meta = process_manifest(manifest_txt, base_dir, only_diagnostic=only_diagnostic)
    df_meta["time"] = df_meta["pid"].map(lambda pid: time_event.get(pid, (np.nan, np.nan))[0])
    df_meta["event"] = df_meta["pid"].map(lambda pid: time_event.get(pid, (np.nan, np.nan))[1])
    df_meta = df_meta.dropna(subset=["time", "event"]).reset_index(drop=True)
    
    if len(df_meta) == 0:
        return pd.DataFrame()

    pids = list(df_meta["pid"].unique())
    records = []

    for pid in ut.pbar(pids, desc="Patients", leave=False):
        dfp = df_meta[df_meta["pid"] == pid]

        feats_all, coords_all = [], []
        slide_idxs_all = []
        slide_files_list = []
        
        slide_paths = dfp["slide_path"].tolist()

        for i, sp in enumerate(ut.pbar(slide_paths, desc=f"{pid} | Slides", leave=False)):
            if not os.path.exists(sp):
                print(f"[MISS] {sp}")
                continue
            try:
                feats, coords = encode_slide_wsi(
                    sp, target_mag, tile_size, step, min_tissue,
                    model, vision_preprocess, device=device
                )
                if len(feats) > 0:
                    feats_all.append(feats)
                    coords_all.append(coords)
                    slide_idxs_all.append(np.full(len(feats), i, dtype=np.int32))
                    slide_files_list.append(os.path.basename(sp))
                    
            except Exception as e:
                print(f"[ERR] {pid} | {os.path.basename(sp)}: {e}")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

        if not feats_all:
            records.append({"pid": pid, "saved": False, "reason": "no_feats", "n_tiles": 0})
            continue

        feats  = np.concatenate(feats_all, axis=0)
        coords = np.concatenate(coords_all, axis=0)
        slide_idxs = np.concatenate(slide_idxs_all, axis=0)
        
        try:
            time_v  = float(dfp["time"].iloc[0])
            event_v = int(dfp["event"].iloc[0])
        except:
            records.append({"pid": pid, "saved": False, "reason": "no_time"})
            continue
            
        patient_clin_dict = clinical_features_map.get(pid, {})

        out_path = os.path.join(out_npz_dir, f"{pid}.npz")
        try:
            np.savez_compressed(
                out_path,
                pid=np.array(pid),
                feats=feats, 
                coords=coords,
                slide_idxs=slide_idxs,                    
                slide_files=np.array(slide_files_list),   
                clinical_features=patient_clin_dict,      
                time=np.array(time_v, dtype=np.float32),  
                event=np.array(event_v, dtype=np.int64)
            )
            
            records.append({
                "pid": pid, "saved": True, "reason": "", 
                "n_tiles": feats.shape[0], "npz_path": out_path
            })
        except Exception as e:
            print(f"[ERR] save {pid}: {e}")
            records.append({"pid": pid, "saved": False, "reason": str(e)})

    return pd.DataFrame.from_records(records)

# Tool function for exporting NPZ for a cancer type with resume capability
def export_npz_for_cancer(
    cancer_code, GDC_ROOT, GDC_CLIENT,
    DOWNLOAD_PARENT, OUT_PARENT, model, vision_preprocess, 
    ENDPOINT="PFS", ONLY_DIAGNOSTIC=True, DELETE_RAW_AFTER=False,
    *,
    MAX_PASSES=3,
    VERIFY_MD5=True,
    SLEEP_INITIAL=15,
    EXTRA_GDC_ARGS=None,
    BATCH_SIZE_FILES=30,
    DELETE_RAW_AFTER_EACH_BATCH=True
):
    """
    For a given cancer type, download WSI files from GDC in batches according to the manifest,
    and export features and metadata to NPZ files. Supports resume from interruption.
    
    This version:
    1. Maintains stable batch ordering by sorting files by ID
    2. Checks for existing NPZ files to avoid reprocessing completed patients
    3. Still checks downloaded raw files to avoid re-downloading
    4. Supports resuming from any point of interruption
    
    Parameters:
        cancer_code: TCGA cancer code
        GDC_ROOT: Root directory of GDC data
        GDC_CLIENT: Path to gdc-client executable
        DOWNLOAD_PARENT: Parent directory for downloading raw files
        OUT_PARENT: Parent directory for output NPZ files
        model: Pre-loaded MUSK model
        vision_preprocess: Preprocessing function for vision model
        tokenizer: Tokenizer for text processing
        ENDPOINT: Survival endpoint ("PFS" or "OS")
        ONLY_DIAGNOSTIC: Whether to only include diagnostic slides
        DELETE_RAW_AFTER: Whether to delete raw downloaded files after processing
        MAX_PASSES: Maximum number of download attempts per batch
        VERIFY_MD5: Whether to verify MD5 checksums after download
        SLEEP_INITIAL: Initial sleep time for exponential backoff
        EXTRA_GDC_ARGS: Extra arguments to pass to gdc-client
        BATCH_SIZE_FILES: Number of files to process per batch
        DELETE_RAW_AFTER_EACH_BATCH: Whether to delete raw files after each batch
    
    Return: summary DataFrame containing the processing summary for each patient
    """
    # ------ Location for generating manifest/clinical files ------
    cancer_dir = os.path.join(GDC_ROOT, cancer_code)
    if not os.path.isdir(cancer_dir):
        print(f"[SKIP] cancer_dir not found: {cancer_dir}")
        return None
    
    # Tool function for finding the first matching file in a directory
    def _find_one(glob_dir, pattern):
        """
        Recursively search for the first matching file within the specified directory.
        
        Parameters:
            glob_dir (str): The directory to search within.
            pattern (str): The glob pattern to match files.
        
        Returns: str or None - The path of the first matching file, or None if not found
        """
        return next((p for p in glob.iglob(os.path.join(glob_dir, pattern), recursive=True)), None)
    
    manifest = _find_one(cancer_dir, f"gdc_manifest.{cancer_code}.txt") \
           or _find_one(cancer_dir, "gdc_manifest*.txt")
    clinical = _find_one(cancer_dir, f"clinical.cart.{cancer_code}.json") \
           or _find_one(cancer_dir, "*clinical*.json")
    if not manifest:
        print(f"[ERR] manifest not found in {cancer_dir}")
        return None
    if not clinical:
        print(f"[ERR] clinical json not found in {cancer_dir}")
        return None

    download_dir = os.path.join(DOWNLOAD_PARENT, f"TCGA_{cancer_code}")
    out_npz_dir  = os.path.join(OUT_PARENT, cancer_code)
    os.makedirs(out_npz_dir, exist_ok=True)
    
    try:
        time_event, clinical_features_map, KP, skipped_pids = process_clinical(clinical, endpoint=ENDPOINT)
    except Exception as e:
        print(f"[ERR] Failed to process clinical data: {e}")
        return None
    
    # Read all rows from manifest
    all_rows, header = ut.load_manifest_minimal(manifest)
    if not all_rows:
        print(f"[ERR] no rows in manifest: {manifest}")
        return None

    # Sort all rows by file ID to ensure stable ordering
    all_rows = sorted(all_rows, key=lambda r: r["id"])
    print(f"[INFO] Total files in manifest: {len(all_rows)}")
    
    print(f"[INFO] Removing {len(skipped_pids)} bad PID(s) from file list...")
    # print(skipped_pids)
    
    all_rows_cleaned = [
        r for r in all_rows
        if r["filename"][:12] not in skipped_pids
    ]

    print(f"[INFO] Files before cleaning: {len(all_rows)}")
    print(f"[INFO] Files after  cleaning: {len(all_rows_cleaned)}")
    print(f"[INFO] Removed {len(all_rows) - len(all_rows_cleaned)} files from bad PIDs")
    
    # Check for already completed NPZ files
    print(f"\n===== [{cancer_code}] Check existing NPZ files in: {out_npz_dir} =====")
    completed_patients = ut.get_completed_patients(out_npz_dir, cancer_code)
    print(f"[NPZ CHECK] Found {len(completed_patients)} patients with existing NPZ files")
    
    # Filter out files from completed patients
    all_rows_filtered = ut.filter_by_completed_patients(all_rows_cleaned, completed_patients)
    print(f"[NPZ CHECK] Files after filtering completed patients: {len(all_rows_filtered)}")

    # Check downloaded raw files
    print(f"\n===== [{cancer_code}] Inspect local cache in: {download_dir} =====")
    subset_tmp = os.path.join(out_npz_dir, f"__probe_{cancer_code}.tsv")
    ut.write_subset_manifest(all_rows_filtered, header, subset_tmp)
    _, missing_all, bad_all = ut.check_completeness_subset(subset_tmp, download_dir, verify_md5=False)
    if os.path.exists(subset_tmp): os.remove(subset_tmp)
    
    need_ids = set([fid for fid, _ in missing_all] + [fid for fid, _ in bad_all])
    
    # Keep only files that need downloading, maintain sorted order
    to_process = [r for r in all_rows_filtered if r["id"] in need_ids] if need_ids else []
    already_downloaded = [r for r in all_rows_filtered if r["id"] not in need_ids]
    
    total = len(all_rows_cleaned)
    total_filtered = len(all_rows_filtered)
    print(f"[CHECK@start] total={total}, already_processed_to_npz={total-total_filtered}")
    print(f"[CHECK@start] remaining={total_filtered}, already_downloaded={len(already_downloaded)}, to_download={len(to_process)}")

    files_to_process_ordered = already_downloaded + to_process
    
    if not files_to_process_ordered:
        print(f"[INFO] All files already processed for {cancer_code}.")
        return None

    # ---------- Split into batches ----------
    batch_plan = []
    for i in range(0, len(files_to_process_ordered), BATCH_SIZE_FILES):
        batch = files_to_process_ordered[i:i+BATCH_SIZE_FILES]
        batch_plan.append(batch)

    print(f"[BATCH PLAN] Total batches: {len(batch_plan)}")
    for i, batch in enumerate(batch_plan, 1):
        need_dl = sum(1 for r in batch if r["id"] in need_ids)
        print(f"  Batch {i}: {len(batch)} files ({need_dl} to download, {len(batch)-need_dl} already downloaded)")
        
    all_summ = []
    # ---------- Download and process in batches ----------
    for bi, batch_rows in enumerate(ut.pbar(batch_plan, desc=f"{cancer_code} | Batches", leave=False), 1):
        print(f"\n===== [{cancer_code}] Batch {bi}/{len(batch_plan)} | size={len(batch_rows)} =====")

        with tempfile.TemporaryDirectory(dir=out_npz_dir) as tmpd:
            sub_manifest = os.path.join(tmpd, f"gdc_manifest.{cancer_code}.batch{bi}.txt")
            ut.write_subset_manifest(batch_rows, header, sub_manifest)

            batch_need_ids = set(r["id"] for r in batch_rows if r["id"] in need_ids)
            needs_download = bool(batch_need_ids)

            if needs_download:
                print(f"[BATCH {bi}] {len(batch_need_ids)}/{len(batch_rows)} files need downloading")
                ok = False
                wait = SLEEP_INITIAL
                with ut.progress_task(MAX_PASSES, desc=f"{cancer_code} | Batch {bi} download", leave=False) as bar:
                    for attempt in range(1, MAX_PASSES + 1):
                        _, missing, bad = ut.check_completeness_subset(sub_manifest, download_dir, verify_md5=False)
                        missing_in_batch = [fid for fid, _ in missing if fid in batch_need_ids]
                        bad_in_batch = [fid for fid, _ in bad if fid in batch_need_ids]
                        
                        if not missing_in_batch and not bad_in_batch:
                            ok = True
                            bar.update(MAX_PASSES - (attempt - 1))
                            break

                        print(f"[PASS {attempt}/{MAX_PASSES}] gdc-client for batch {bi}")
                        rc = ut.run_gdc_client_once(GDC_CLIENT, sub_manifest, download_dir, extra=EXTRA_GDC_ARGS)
                        if rc != 0:
                            print(f"[WARN] gdc-client returned {rc}.")
                        bar.update(1)
                        if attempt < MAX_PASSES:
                            time.sleep(min(wait, 120))
                            wait *= 2

                if not ok:
                    print(f"[ERR] batch {bi} incomplete for {cancer_code}")
                    return None

                if VERIFY_MD5:
                    rows_sub, _ = ut.load_manifest_minimal(sub_manifest)
                    missing_f, bad_f = [], []
                    for r in ut.pbar(rows_sub, desc=f"{cancer_code} | Batch {bi} md5", leave=False):
                        if r["id"] not in batch_need_ids: continue
                        fid, fname = r["id"], r["filename"]
                        p = os.path.join(download_dir, fid, fname)
                        if not os.path.exists(p) or os.path.getsize(p) <= 0:
                            missing_f.append((fid, fname))
                            continue
                        md5 = r.get("md5") or ""
                        if md5:
                            try:
                                if ut.md5sum(p) != md5: bad_f.append((fid, fname))
                            except: bad_f.append((fid, fname))
                    
                    if missing_f or bad_f:
                        print(f"[ERR] batch {bi} md5 check failed.")
                        return None
                    print(f"[OK] batch {bi} md5 check passed.")
            else:
                print(f"[BATCH {bi}] All files already downloaded, proceeding to NPZ export")
            print(f"[NPZ] Export batch {bi} -> {out_npz_dir}")
            try:
                summ = export_npz_for_patient(
                    manifest_txt=sub_manifest,
                    base_dir=download_dir, 
                    out_npz_dir=out_npz_dir,
                    time_event=time_event, 
                    clinical_features_map=clinical_features_map, 
                    model=model, 
                    vision_preprocess=vision_preprocess, 
                    only_diagnostic=ONLY_DIAGNOSTIC,
                    target_mag=TARGET_MAG, tile_size=TILE_SIZE, 
                    step=STEP, min_tissue=MIN_TISSUE,
                    device=DEVICE
                )
            except Exception as e:
                print(f"[ERR] export npz failed (batch {bi}): {e}")
                import traceback
                traceback.print_exc()
                return None

            if isinstance(summ, pd.DataFrame):
                all_summ.append(summ)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if DELETE_RAW_AFTER_EACH_BATCH:
                del_ids = list({r["id"] for r in batch_rows})
                removed = 0
                for fid in ut.pbar(del_ids, desc=f"{cancer_code} | Batch {bi} cleanup", leave=False):
                    d = os.path.join(download_dir, fid)
                    if os.path.isdir(d):
                        shutil.rmtree(d, ignore_errors=True)
                        removed += 1
                print(f"[CLEAN] batch {bi}: removed {removed}/{len(del_ids)} file_id dirs.")

    # ---------- Final cleanup ----------
    summary = (pd.concat(all_summ, ignore_index=True) if all_summ else None)
    if DELETE_RAW_AFTER and not DELETE_RAW_AFTER_EACH_BATCH and summary is not None and (summary["saved"].sum() > 0):
        try:
            print(f"\n===== [{cancer_code}] Cleanup all raw dir: {download_dir} =====")
            shutil.rmtree(download_dir, ignore_errors=True)
            print("[CLEAN] removed:", download_dir)
        except Exception as e:
            print(f"[WARN] cleanup failed: {e}")

    return summary
