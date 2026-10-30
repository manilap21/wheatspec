import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

# Base directory is /Users/manuliperera/Documents/THESIS DATA/RHEOLOGY
BASE_DIR = Path(".")
DATA_SUBDIR = BASE_DIR / "data_rheology"
OUT_DIR = BASE_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 6 Primary Dough Rheology Parameters
TRAIT_PATTERNS = {
    "Protein": [
        r"^protein",
        r"^pr$",
        r"crude_protein",
        r"grain_protein",
        r"flour_protein",
    ],
    "Absorption": [
        r"^absorp",
        r"water_abs",
        r"^wa$",
        r"water\s*absorption",
        r"fabsorp",
    ],
    "Extensibility": [r"^extensib", r"^ext$", r"^e$", r"extensograph_ext"],
    "Rmax": [
        r"^rmax",
        r"^r_max",
        r"max_res",
        r"maximum_resistance",
        r"resistance",
    ],
    "DDT": [r"^ddt", r"dough_dev", r"development_time", r"dev_time"],
    "Stability": [
        r"^stabil",
        r"^stab$",
        r"dough_stab",
        r"farinograph_stability",
    ],
}


def find_file(filename):
  """Checks data_rheology/, the parent folder, and subfolders flexibly."""
  candidates = [
      DATA_SUBDIR / filename,
      BASE_DIR / filename,
      BASE_DIR / "rheology_data" / filename,
  ]
  for c in candidates:
    if c.exists() and c.is_file():
      return c

  # Fallback: case-insensitive search across folders
  for folder in [DATA_SUBDIR, BASE_DIR]:
    if folder.exists() and folder.is_dir():
      for f in folder.iterdir():
        if f.name.lower() == filename.lower():
          return f
  return None


def get_numeric_wavenumbers(df):
  wn_cols = []
  for c in df.columns:
    try:
      val = float(str(c).replace("-", "").strip())
      if 400.0 <= val <= 5000.0:
        wn_cols.append(c)
    except ValueError:
      pass
  return sorted(wn_cols, key=lambda x: float(str(x).replace("-", "").strip()))


def find_trait_column(columns, patterns):
  for p in patterns:
    for c in columns:
      if re.search(p, str(c).strip(), re.IGNORECASE):
        return c
  return None


def find_common_key(df_spec, df_ref):
  """Locates common identifier column like ID, Sample, or Name."""
  spec_cols = set(df_spec.columns)
  ref_cols = set(df_ref.columns)
  candidates = [
      "ID",
      "Sample_ID",
      "Sample",
      "SampleID",
      "Name",
      "Bio_Rep_ID",
      "Variety",
  ]
  for cand in candidates:
    if cand in spec_cols and cand in ref_cols:
      return cand
  common = [
      c
      for c in spec_cols.intersection(ref_cols)
      if not str(c).replace(".", "", 1).isdigit()
  ]
  if common:
    return common[0]
  return None


def merge_cohort(spec_filename, ref_filename, cohort_name):
  print(f"\n[+] Searching files for Cohort: {cohort_name}")
  spec_path = find_file(spec_filename)
  ref_path = find_file(ref_filename)

  if not spec_path:
    print(f"    [!] Missing spectra file: '{spec_filename}'")
    return None
  if not ref_path:
    print(f"    [!] Missing reference file: '{ref_filename}'")
    return None

  print(f"    Found spectra: {spec_path}")
  print(f"    Found ref:     {ref_path}")

  df_spec = pd.read_csv(spec_path, encoding="latin1")
  df_ref = pd.read_csv(ref_path, encoding="latin1")

  wn_cols = get_numeric_wavenumbers(df_spec)
  print(
      f"    Spectra: {df_spec.shape[0]} rows x {df_spec.shape[1]} cols ({len(wn_cols)} channels)"
  )
  print(
      f"    Ref:     {df_ref.shape[0]} rows x {df_ref.shape[1]} cols |"
      f" Columns: {list(df_ref.columns)}"
  )

  key = find_common_key(df_spec, df_ref)
  if key:
    print(f"    Matching datasets on shared key: '{key}'")
    df_spec[key] = df_spec[key].astype(str).str.strip().str.upper()
    df_ref[key] = df_ref[key].astype(str).str.strip().str.upper()
    merged = pd.merge(df_ref, df_spec, on=key, how="inner")
  else:
    print(
        "    [!] No shared identifier string detected. Merging 1:1 by"
        " positional row order."
    )
    merged = pd.concat(
        [df_ref.reset_index(drop=True), df_spec.reset_index(drop=True)], axis=1
    )

  print(f"    Successfully matched: {merged.shape[0]} samples.")
  merged["__cohort__"] = cohort_name
  return merged


def compile_master_rheology():
  print("=" * 75)
  print("STEP 1: COMPILING MASTER RHEOLOGY DATASET (SSA + SSB)")
  print("=" * 75)

  df_ssa = merge_cohort("SSA_FTIR_data.csv", "SSA_reference_data.csv", "SSA")
  df_ssb = merge_cohort("SSB_FTIR_data.csv", "SSB_reference_data.csv", "SSB")

  cohorts = [df for df in [df_ssa, df_ssb] if df is not None]
  if not cohorts:
    raise FileNotFoundError("Could not load SSA or SSB from 'data_rheology/'.")

  df_raw = pd.concat(cohorts, axis=0, ignore_index=True)
  print(f"\n[i] Combined raw matched sample count: {len(df_raw)} records")

  clean_df = pd.DataFrame()
  clean_df["Cohort"] = df_raw["__cohort__"]

  id_col = next(
      (
          c
          for c in ["ID", "Sample_ID", "Sample", "Variety", "Name"]
          if c in df_raw.columns
      ),
      None,
  )
  if id_col:
    clean_df["Sample_ID"] = df_raw[id_col]

  print("\n[i] Extracting 6 Physical Dough Rheology Traits:")
  traits_detected = []
  for trait, patterns in TRAIT_PATTERNS.items():
    matched = find_trait_column(df_raw.columns, patterns)
    if matched:
      clean_df[trait] = pd.to_numeric(df_raw[matched], errors="coerce")
      valid_n = clean_df[trait].notna().sum()
      print(
          f"    - Trait '{trait:<14}' <- Column '{matched}' (Valid readings:"
          f" {valid_n})"
      )
      traits_detected.append(trait)
    else:
      clean_df[trait] = np.nan
      print(f"    - Trait '{trait:<14}' <- NOT DETECTED (set to NaN)")

  wn_channels = get_numeric_wavenumbers(df_raw)
  print(
      f"\n[+] Extracting {len(wn_channels)} spectral channels ({wn_channels[0]}"
      f" to {wn_channels[-1]} cm⁻¹)"
  )
  for c in wn_channels:
    clean_df[str(c)] = pd.to_numeric(df_raw[c], errors="coerce")

  clean_df = clean_df.drop_duplicates(subset=[str(c) for c in wn_channels[:20]])
  print(f"[i] Samples after spectral deduplication: {len(clean_df)}")

  if traits_detected:
    has_trait = clean_df[traits_detected].notna().any(axis=1)
    final_df = clean_df[has_trait].copy()
  else:
    final_df = clean_df.copy()

  out_file = OUT_DIR / "master_rheology_dataset.csv"
  final_df.to_csv(out_file, index=False)

  print("=" * 75)
  print(f"SUCCESS: Master Rheology Dataset saved to: {out_file.resolve()}")
  print(
      f"Final Matrix Shape: {final_df.shape[0]} clean samples x"
      f" {final_df.shape[1]} columns"
  )
  for t in traits_detected:
    print(
        f"  * {t:<14}: {final_df[t].notna().sum()} valid reference measurements"
    )
  print("=" * 75)


if __name__ == "__main__":
  compile_master_rheology()