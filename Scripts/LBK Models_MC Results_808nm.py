"""
Lower Back (Lumbar L3-S1) STL Mesh → 3D Voxel Volume + pmcx Fluence Overlay (808 nm)
---------------------------------------------------------------------------------------
Pipeline (shared logic lives in pbm_mc_core; see that package's README for the
full stage list and the tissue-label convention this script's `tissues` dict
follows).

Tissue hierarchy (highest label wins when meshes overlap):
  1  L3-bone             L3 vertebral body
  2  L4-bone             L4 vertebral body (primary PBM target)
  3  L5-bone             L5 vertebral body
  4  S1-bone             Sacrum (superior segment)
  5  L3L4-disc           L3-L4 disc annulus fibrosus (fibrocartilage)
  6  L4L5-disc           L4-L5 disc annulus fibrosus (primary target)
  7  L5S1-disc           L5-S1 disc annulus fibrosus
  14 nucleus             Disc nucleus pulposus (synthesised from disc dilation,
                         or loaded from separate STL if available - water-like)
  11 muscle              Synthesised - concentric dilation (paraspinals)
  12 adipose             Synthesised - concentric dilation
  13 skin                Synthesised - concentric dilation
  15 epidermis           Synthesised - outermost 1-voxel skin ring

Wrapping note:
  MUSCLE_THICK_MM = 35 mm models the paraspinal muscle bulk (erector spinae +
  multifidus) at L4/L5.  This is the dominant depth-limiting factor for lumbar
  PBM and is larger than any other Kineon target.  Subcutaneous fat is also
  thicker posteriorly (~8-12 mm) in typical subjects.

Source positions (default):
  +Y = anterior (ventral),  -Y = posterior (dorsal),  +Z = superior (cranial)
  Move+ lumbar placement uses bilateral posterior pads flanking the spinous
  process.  Three posterior sources are used: one central (over the midline)
  and two bilateral (+/-30 mm lateral to spinous process).

Dependencies:
    pip install numpy trimesh pmcx plotly scipy
    pip install git+https://github.com/CLB-GH2026/pbm-mc-core.git@v0.1.1
"""

import numpy as np
import time
from pathlib import Path
from datetime import datetime

from pbm_mc_core import (
    opt, EPIDERMIS_LABEL, build_melanin_conditions,
    build_label_volume,
    add_synovial_fluid, add_wrapping_layers, add_epidermis_layer,
    find_joint_line_z, find_surface_source_positions,
    optimize_source_positions_reciprocity,
    run_pmcx,
    analyze_fluence_absorption, analyze_penetration_depth, plot_depth_histogram,
    target_depth_zone,
    results_to_csv, melanin_comparison_to_csv,
)

# Lower-back anatomy depth references (approximate, posterior/paraspinal
# access) — NOT knee's zone; the disc/nucleus target sits much deeper than
# knee's joint line, so this must be passed explicitly to
# pbm_mc_core.plot_depth_histogram (see its docstring — depth_refs/zone_lo/
# zone_hi default to knee's much shallower values otherwise).
_LBK_DEPTH_REFS = [(1.0, 'Skin/Adipose'), (4.5, 'Paraspinals'), (6.5, 'L4/L5 Disc')]
_LBK_ZONE_LO, _LBK_ZONE_HI = 4.5, 6.5

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

start_time = time.perf_counter()

WAVELENGTH_M  = 808e-9
WAVELENGTH_NM = 808

# Epidermal optical properties by melanin condition at 808 nm.
# True (unscaled) values; build_melanin_conditions() applies the epidermis
# thickness-correction scale (0.2 mm physical / 1 mm voxel).
_MELANIN_RAW_808NM = {
    #        µa      µs'    g     n
    'fair':  (0.008, 1.50, 0.80, 1.40),  # Fitzpatrick I-II
    'olive': (0.025, 1.60, 0.80, 1.40),  # Fitzpatrick III-IV
    'dark':  (0.075, 1.70, 0.80, 1.40),  # Fitzpatrick V-VI
}

# ── Source optimiser ──────────────────────────────────────────────────────────
OPTIMIZE_SOURCES = False   # True → per-subject reciprocity scan before main run
OPT_N_SOURCES    = 3
OPT_MIN_SEP_MM   = 25.0
OPT_NPHOTON      = 1e6

# ── Grid / voxel ─────────────────────────────────────────────────────────────
VOXEL_SIZE    = 1.0               # mm per voxel
GRID_DIMS_MM  = (200, 180, 280)   # x, y, z - lumbar stack is tall; depth ~18 cm posterior to disc
VOXEL_RES     = tuple(int(round(d / VOXEL_SIZE)) for d in GRID_DIMS_MM)
AUTO_ORIENT   = True              # auto-correct Z-axis inversion (L3 above S1 check)
FLUENCE_OUTPUT = None             # None = run pmcx; path string = load saved .npy

# ── Soft-tissue wrapping (mm) ─────────────────────────────────────────────────
MUSCLE_THICK_MM  = 30   # paraspinal (erector spinae + multifidus) at L4/L5; trimmed from 35 (isotropic wrap overestimates mean depth)
ADIPOSE_THICK_MM =  8   # posterior subcutaneous fat - thicker than knee/shoulder
SKIN_THICK_MM    =  2

# ── Source power ──────────────────────────────────────────────────────────────
SOURCE_POWER_MW   = 50
SOURCE_DUTY_CYCLE = 0.75
SOURCE_OPT_EFF    = 0.85
CONE_ANGLE_DEG    = 20             # source cone full angle

MELANIN_CONDITIONS = build_melanin_conditions(_MELANIN_RAW_808NM, voxel_size_mm=VOXEL_SIZE)

# ─────────────────────────────────────────────────────────────────────────────
# TISSUE GROUPS (lower-back anatomy: no synovial fluid — disc annulus and
# nucleus pulposus stand in for cartilage/synovial fluid) — passed into
# analyze_fluence_absorption / results_to_csv / melanin_comparison_to_csv,
# which are anatomy-agnostic in pbm_mc_core.
# ─────────────────────────────────────────────────────────────────────────────
GROUPS = {
    'Bone':    lambda n: 'bone'    in n,
    'Disc':    lambda n: 'disc'    in n,
    'Nucleus': lambda n: 'nucleus' in n,
    'Muscle':  lambda n: 'muscle'  in n,
    'Adipose': lambda n: 'adipose' in n,
    'Skin+Epidermis': lambda n: ('skin' in n) or ('epidermis' in n),
}
DOSE_GROUPS = {
    'Disc':    lambda n: 'disc'    in n,
    'Nucleus': lambda n: 'nucleus' in n,
    'Muscle':  lambda n: 'muscle'  in n,
}
COMP_GROUPS = {
    'Disc':           lambda n: 'disc'     in n,
    'Nucleus':        lambda n: 'nucleus'  in n,
    'Muscle':         lambda n: 'muscle'   in n,
    'Bone':           lambda n: 'bone'     in n,
    'Skin+Epidermis': lambda n: 'skin' in n or 'epidermis' in n,
}

# Lower back has no "synovial"/"cartilage" tissue names — disc annulus and
# nucleus pulposus are this anatomy's joint-space-fluid-equivalent targets.
TARGET_MATCH_FN = lambda name: ('disc' in name) or ('nucleus' in name)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PER-SUBJECT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_subject(subject_id, mesh_dir_base, output_dir, melanin_condition='fair'):
    """Run the full pipeline for a single lower-back (lumbar L3-S1) subject."""

    mesh_dir = Path(mesh_dir_base) / f"Raw_Mesh_Files_{subject_id}"
    if not mesh_dir.exists():
        print(f"  Skipping {subject_id} - directory not found: {mesh_dir}")
        return None

    print(f"\n{'=' * 60}")
    print(f"  Processing {subject_id}  [{melanin_condition}]")
    print(f"{'=' * 60}")

    # ── Tissue table ─────────────────────────────────────────────────────────
    # Optical properties at 808 nm (ua, us', g, n).
    tissues = {
        "nucleus":   (None,                                            14, opt(0.0002, 0.005, 0.90, 1.35)),  # nucleus pulposus - near-water
        "skin":      (None,                                            13, opt(0.003,  1.22,  0.79, 1.40)),
        "adipose":   (None,                                            12, opt(0.0013, 1.00,  0.90, 1.44)),
        "muscle":    (None,                                            11, opt(0.0180, 0.55,  0.93, 1.37)),
        "L5S1-disc": (mesh_dir / "L5S1_disc_raw.stl",                  7, opt(0.006,  1.80,  0.90, 1.37)),  # fibrocartilage
        "L4L5-disc": (mesh_dir / "L4L5_disc_raw.stl",                  6, opt(0.006,  1.80,  0.90, 1.37)),  # fibrocartilage - primary target
        "L3L4-disc": (mesh_dir / "L3L4_disc_raw.stl",                  5, opt(0.006,  1.80,  0.90, 1.37)),  # fibrocartilage
        "S1-bone":   (mesh_dir / "S1_raw.stl",                         4, opt(0.040,  2.50,  0.92, 1.37)),
        "L5-bone":   (mesh_dir / "L5_raw.stl",                         3, opt(0.040,  2.50,  0.92, 1.37)),
        "L4-bone":   (mesh_dir / "L4_raw.stl",                         2, opt(0.040,  2.50,  0.92, 1.37)),
        "L3-bone":   (mesh_dir / "L3_raw.stl",                         1, opt(0.040,  2.50,  0.92, 1.37)),
    }
    tissues["epidermis"] = (None, EPIDERMIS_LABEL, MELANIN_CONDITIONS[melanin_condition])

    try:
        # ── Step 1: Build label volume ────────────────────────────────────
        vol, origin, mesh_center = build_label_volume(
            tissues, VOXEL_RES, VOXEL_SIZE,
            auto_orient=AUTO_ORIENT,
            orient_ref_a='L3-bone', orient_ref_b='S1-bone',
        )

        # ── Step 2: Add nucleus pulposus and wrapping layers ─────────────
        bone_labels      = [t[1] for name, t in tissues.items() if "bone" in name]
        cartilage_labels = [t[1] for name, t in tissues.items() if "disc" in name]

        # Nucleus pulposus: dilate disc annulus inward to fill nucleus space.
        # Uses label 14 (same label convention as synovial fluid elsewhere —
        # add_synovial_fluid() is generic and works for any joint-space-fluid
        # analog).
        vol = add_synovial_fluid(
            vol,
            cartilage_labels=cartilage_labels,
            bone_labels=bone_labels,
            fluid_label=tissues["nucleus"][1],
            dilation_vox=2
        )

        layer_configs_vox = [
            (tissues["muscle"][1],  int(round(MUSCLE_THICK_MM  / VOXEL_SIZE))),
            (tissues["adipose"][1], int(round(ADIPOSE_THICK_MM / VOXEL_SIZE))),
            (tissues["skin"][1],    int(round(SKIN_THICK_MM    / VOXEL_SIZE))),
        ]
        vol = add_wrapping_layers(vol, layer_configs_vox)
        vol = add_epidermis_layer(vol, skin_label=tissues["skin"][1],
                                   epidermis_label=EPIDERMIS_LABEL)

        # ── Step 2b: Locate disc-level Z (this anatomy's "joint line") ───
        jl_z = find_joint_line_z(vol, tissues, origin, VOXEL_SIZE, mesh_center,
                                  target_match_fn=TARGET_MATCH_FN)

        # ── Step 3: Compute source directions and place on epidermis surface
        _colors = ['red', 'green', 'blue', 'orange', 'purple']
        if OPTIMIZE_SOURCES:
            print("\n--- Reciprocity source position optimisation ---")
            opt_positions = optimize_source_positions_reciprocity(
                vol, tissues, origin, mesh_center, VOXEL_SIZE,
                OPT_N_SOURCES, OPT_MIN_SEP_MM, OPT_NPHOTON,
                epidermis_label=EPIDERMIS_LABEL,
                target_match_fn=TARGET_MATCH_FN,
            )
            if opt_positions:
                src_configs = [
                    {'name': f'Opt-{i+1}', 'world_pos': pos, 'color': _colors[i % len(_colors)]}
                    for i, pos in enumerate(opt_positions)
                ]
            else:
                print("  [OPT] Falling back to default positions")
                src_configs = _default_src_configs(jl_z)
        else:
            src_configs = _default_src_configs(jl_z)

        for cfg in src_configs:
            d = np.array([0, 0, jl_z]) - np.array(cfg['world_pos'])
            cfg['srcdir'] = (d / np.linalg.norm(d)).tolist()

        pmcx_source_plus = find_surface_source_positions(
            vol, origin, VOXEL_SIZE, mesh_center, src_configs
        )
        pmcx_source = [{'srcpos': s['srcpos'], 'srcdir': s['srcdir']}
                       for s in pmcx_source_plus]

        # ── Step 4: Run pmcx ──────────────────────────────────────────────
        fluence_combined, fluence_list = run_pmcx(
            vol, tissues, pmcx_source,
            wavelength_m=WAVELENGTH_M,
            source_power_mw=SOURCE_POWER_MW,
            duty_cycle=SOURCE_DUTY_CYCLE,
            opt_eff=SOURCE_OPT_EFF,
            cone_angle_deg=CONE_ANGLE_DEG,
            voxel_size_mm=VOXEL_SIZE,
        )

        # ── Step 5: Absorption analysis ───────────────────────────────────
        results = analyze_fluence_absorption(
            fluence_combined, vol, tissues, VOXEL_SIZE,
            pmcx_source=pmcx_source,
            groups=GROUPS,
            source_power_mw=SOURCE_POWER_MW,
            duty_cycle=SOURCE_DUTY_CYCLE,
            opt_eff=SOURCE_OPT_EFF,
        )

        # ── Step 6: Save subject outputs ──────────────────────────────────
        subj_dir = Path(output_dir) / melanin_condition / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        disc_names   = [n for n in results if 'disc'     in n]
        disc_vox     = sum(results[n]['n_voxels'] for n in disc_names)
        disc_flu_mw  = (sum(results[n]['mean_flu'] * results[n]['n_voxels']
                            for n in disc_names) / disc_vox) if disc_vox > 0 else 0.0

        nucleus_names   = [n for n in results if 'nucleus'  in n]
        nucleus_vox     = sum(results[n]['n_voxels'] for n in nucleus_names)
        nucleus_flu_mw  = (sum(results[n]['mean_flu'] * results[n]['n_voxels']
                               for n in nucleus_names) / nucleus_vox) if nucleus_vox > 0 else 0.0

        print("\n=== Penetration depth analysis ===")
        bin_centers, mean_flu, max_depth = analyze_penetration_depth(
            fluence_combined, vol, VOXEL_SIZE, mesh_center, origin
        )
        # Data-driven dose zone from the actual target (disc/nucleus) depth.
        z_lo, z_hi, z_med = target_depth_zone(vol, tissues, VOXEL_SIZE, TARGET_MATCH_FN)
        if z_lo is None:
            z_lo, z_hi, z_med = _LBK_ZONE_LO, _LBK_ZONE_HI, 0.5 * (_LBK_ZONE_LO + _LBK_ZONE_HI)
        print(f"  Target depth zone: {z_lo:.2f}-{z_hi:.2f} cm (median {z_med:.2f} cm)")
        fig_depth = plot_depth_histogram(
            bin_centers, mean_flu, subject_id, WAVELENGTH_NM,
            depth_refs=[(z_med, 'Disc/nucleus (targets)')],
            zone_lo=z_lo, zone_hi=z_hi,
            group_flu_mw={
                'Disc': disc_flu_mw,
                'Nucleus Pulposus': nucleus_flu_mw,
            },
        )
        depth_html = str(subj_dir / f"depth_histogram_{subject_id}_{melanin_condition}.html")
        fig_depth.write_html(depth_html)
        print(f"  Saved: {depth_html}")

        np.save(subj_dir / "label_volume.npy", vol)
        np.save(subj_dir / "fluence_combined.npy", fluence_combined)
        for i, flu in enumerate(fluence_list):
            np.save(subj_dir / f"fluence_src{i + 1}.npy", flu)

        return subject_id, results

    except Exception as e:
        print(f"  ERROR processing {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _default_src_configs(jl_z):
    """
    Default source positions for the lumbar spine at 808 nm.

    Coordinate convention:
      +Y = anterior (ventral),  -Y = posterior (dorsal),  +Z = superior (cranial)

    Move+ lumbar placement uses posterior pads flanking the spinous process:
      Centre:      directly posterior over the spinous process (Y ~ -80 mm).
      Left/Right:  bilateral, ~30 mm lateral to spinous process (X ~ +/-30 mm).

    All Z values are auto-set to jl_z (disc centroid height).
    """
    return [
        {'name': 'Posterior (C)',   'world_pos': [  0, -80, jl_z], 'color': 'red'  },
        {'name': 'Posterior (L)',   'world_pos': [-30, -75, jl_z], 'color': 'green'},
        {'name': 'Posterior (R)',   'world_pos': [ 30, -75, jl_z], 'color': 'blue' },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Subject list ─────────────────────────────────────────────────────────
    # Populate once STL files are available.
    # Expected directory name format: Raw_Mesh_Files_LBK001, LBK002, …
    # Required STL files per subject (see TISSUE TABLE above):
    #   L3_raw.stl, L4_raw.stl, L5_raw.stl, S1_raw.stl,
    #   L3L4_disc_raw.stl, L4L5_disc_raw.stl, L5S1_disc_raw.stl
    SUBJECT_IDS = ["LBK001"]   # e.g. ["LBK001", "LBK002"]

    BASE_DIR   = Path(".")
    RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = Path(f"results_lowerback_808nm_{RUN_ID}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Lower Back MC Simulation - 808 nm")
    print(f"Subjects: {SUBJECT_IDS if SUBJECT_IDS else '(none configured - add to SUBJECT_IDS)'}")
    print(f"Output:   {OUTPUT_DIR}")

    if not SUBJECT_IDS:
        print("\n⚠  No subjects configured.  Add subject IDs to SUBJECT_IDS and place "
              "STL files in Raw_Mesh_Files_LBK### directories.")
        raise SystemExit(0)

    all_condition_results = {}
    for condition in MELANIN_CONDITIONS:
        print(f"\n{'=' * 60}\n  Melanin: {condition.upper()}\n{'=' * 60}")
        (OUTPUT_DIR / condition).mkdir(exist_ok=True)
        cond_results = []
        for subject_id in SUBJECT_IDS:
            result = run_subject(subject_id, BASE_DIR, OUTPUT_DIR,
                                 melanin_condition=condition)
            if result is not None:
                cond_results.append(result)
        all_condition_results[condition] = cond_results
        if cond_results:
            results_to_csv(
                cond_results,
                groups=GROUPS,
                dose_groups=DOSE_GROUPS,
                source_power_mw=SOURCE_POWER_MW,
                duty_cycle=SOURCE_DUTY_CYCLE,
                opt_eff=SOURCE_OPT_EFF,
                n_sources=3,
                output_path=str(OUTPUT_DIR / f"MC_LowerBack_808nm_{condition}.csv"),
            )

    melanin_comparison_to_csv(
        all_condition_results,
        groups=COMP_GROUPS,
        output_path=str(OUTPUT_DIR / "MC_LowerBack_Melanin_Comparison_808nm.csv"),
        wavelength_nm=WAVELENGTH_NM,
    )
    print(f"\nDone.")
