#!/usr/bin/env python3
"""
AutoDock Vina pipeline: screen all 20 standard amino acids against
the AlphaFold-predicted structure of a given UniProt target (default C6XH98).

Usage:
    python dock_amino_acids.py [--uniprot C6XH98] [--pdb path/to/structure.pdb]
                               [--exhaustiveness 16] [--output results/]
"""

import argparse
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Amino acid definitions -- pH 7.4 physiological protonation states
# All are zwitterions (alpha-NH3+, alpha-COO-); side chains set by pKa:
#   Arg +1 (guanidinium pKa ~12.5), Lys +1 (e-NH3+ pKa ~10.5),
#   Asp -1 (carboxylate pKa ~3.9),  Glu -1 (carboxylate pKa ~4.1),
#   His neutral (imidazole pKa ~6.0 -> ~10% protonated at pH 7.4),
#   Cys neutral (thiol pKa ~8.3 -> mostly SH at pH 7.4)
# ---------------------------------------------------------------------------
AMINO_ACIDS = {
    "GLY": ("Glycine",       "[NH3+]CC(=O)[O-]"),
    "ALA": ("Alanine",       "[NH3+][C@@H](C)C(=O)[O-]"),
    "VAL": ("Valine",        "[NH3+][C@@H](C(C)C)C(=O)[O-]"),
    "LEU": ("Leucine",       "[NH3+][C@@H](CC(C)C)C(=O)[O-]"),
    "ILE": ("Isoleucine",    "[NH3+][C@@H]([C@@H](CC)C)C(=O)[O-]"),
    "PRO": ("Proline",       "[O-]C(=O)[C@@H]1CCC[NH2+]1"),
    "PHE": ("Phenylalanine", "[NH3+][C@@H](Cc1ccccc1)C(=O)[O-]"),
    "TRP": ("Tryptophan",    "[NH3+][C@@H](Cc1c[nH]c2ccccc12)C(=O)[O-]"),
    "MET": ("Methionine",    "[NH3+][C@@H](CCSC)C(=O)[O-]"),
    "SER": ("Serine",        "[NH3+][C@@H](CO)C(=O)[O-]"),
    "THR": ("Threonine",     "[NH3+][C@@H]([C@@H](O)C)C(=O)[O-]"),
    "CYS": ("Cysteine",      "[NH3+][C@@H](CS)C(=O)[O-]"),
    "TYR": ("Tyrosine",      "[NH3+][C@@H](Cc1ccc(O)cc1)C(=O)[O-]"),
    "ASN": ("Asparagine",    "[NH3+][C@@H](CC(N)=O)C(=O)[O-]"),
    "GLN": ("Glutamine",     "[NH3+][C@@H](CCC(N)=O)C(=O)[O-]"),
    "ASP": ("Aspartate",     "[NH3+][C@@H](CC(=O)[O-])C(=O)[O-]"),
    "GLU": ("Glutamate",     "[NH3+][C@@H](CCC(=O)[O-])C(=O)[O-]"),
    "LYS": ("Lysine",        "[NH3+][C@@H](CCCC[NH3+])C(=O)[O-]"),
    "ARG": ("Arginine",      "[NH3+][C@@H](CCCNC(=[NH2+])N)C(=O)[O-]"),
    "HIS": ("Histidine",     "[NH3+][C@@H](Cc1cnc[nH]1)C(=O)[O-]"),
}

SINGLE_LETTER = {
    "GLY": "G", "ALA": "A", "VAL": "V", "LEU": "L", "ILE": "I",
    "PRO": "P", "PHE": "F", "TRP": "W", "MET": "M", "SER": "S",
    "THR": "T", "CYS": "C", "TYR": "Y", "ASN": "N", "GLN": "Q",
    "ASP": "D", "GLU": "E", "LYS": "K", "ARG": "R", "HIS": "H",
}


# ---------------------------------------------------------------------------
# Structure download
# ---------------------------------------------------------------------------

def download_alphafold(uniprot_id: str, out_path: Path) -> Path:
    """Download AlphaFold predicted structure from EBI."""
    import urllib.request
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"[download] saved -> {out_path}  ({out_path.stat().st_size:,} bytes)")
    return out_path


# ---------------------------------------------------------------------------
# Protein preparation
# ---------------------------------------------------------------------------

def clean_pdb(pdb_in: Path, pdb_out: Path) -> Path:
    """Keep only ATOM records; strip HETATM/waters from PDB."""
    lines = []
    with open(pdb_in) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "ATOM":
                lines.append(line)
            elif rec in ("TER", "END"):
                lines.append(line)
    with open(pdb_out, "w") as fh:
        fh.writelines(lines)
    print(f"[prep]     cleaned PDB -> {pdb_out}  ({len(lines)} lines)")
    return pdb_out


def pdb_to_pdbqt_receptor(pdb_path: Path, pdbqt_path: Path) -> Path:
    """Convert cleaned PDB to PDBQT using OpenBabel."""
    if pdbqt_path.exists():
        print(f"[prep]     receptor PDBQT cached -> {pdbqt_path}")
        return pdbqt_path

    from openbabel import openbabel

    ob = openbabel.OBConversion()
    ob.SetInAndOutFormats("pdb", "pdbqt")
    ob.AddOption("h", openbabel.OBConversion.GENOPTIONS)
    ob.AddOption("xr", openbabel.OBConversion.OUTOPTIONS)
    mol = openbabel.OBMol()
    if not ob.ReadFile(mol, str(pdb_path)):
        raise RuntimeError(f"OpenBabel could not read {pdb_path}")
    mol.AddPolarHydrogens()
    ob.WriteFile(mol, str(pdbqt_path))

    # OpenBabel sometimes emits ROOT/BRANCH/TORSDOF lines (ligand-style PDBQT).
    # Vina rejects these in a rigid receptor -- strip them.
    raw = pdbqt_path.read_text()
    kept = [l for l in raw.splitlines()
            if l[:6].strip() in ("ATOM", "HETATM", "TER", "END", "REMARK") or l.strip() == ""]
    pdbqt_path.write_text("\n".join(kept) + "\n")

    print(f"[prep]     receptor PDBQT -> {pdbqt_path}  ({len(kept)} lines)")
    return pdbqt_path


def get_protein_box(pdb_path: Path, padding: float = 4.0):
    """Return (center, size) of protein bounding box with padding, A."""
    coords = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                except ValueError:
                    pass
    if not coords:
        raise ValueError("No ATOM coordinates found in PDB")
    arr = np.array(coords)
    lo = arr.min(axis=0) - padding
    hi = arr.max(axis=0) + padding
    center = ((lo + hi) / 2).tolist()
    size   = (hi - lo).tolist()
    size = [min(s, 126.0) for s in size]
    print(f"[box]      center={[f'{c:.1f}' for c in center]}  size={[f'{s:.1f}' for s in size]}")
    return center, size


def detect_binding_pocket(pdb_path: Path, grid_spacing: float = 2.0,
                           inner_r: float = 8.0, outer_r: float = 20.0,
                           box_size: float = 28.0):
    """
    Grid-based buriedness pocket detection (simplified LIGSITE).
    Scans a 3-D grid; pocket candidates have 15-200 protein atoms within 8 A
    (partially buried but not core) and highest protein context within 20 A.
    Returns (center, [box_size]*3) or (None, None) on failure.
    """
    from scipy.spatial import cKDTree

    coords = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    pass
    if not coords:
        return None, None

    arr = np.array(coords)
    tree = cKDTree(arr)

    lo, hi = arr.min(0) - 1.0, arr.max(0) + 1.0
    axes = [np.arange(lo[i], hi[i], grid_spacing) for i in range(3)]
    GX, GY, GZ = np.meshgrid(*axes, indexing="ij")
    grid = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
    print(f"[pocket]   scanning {len(grid):,} grid points ...")

    inner_counts = np.array(tree.query_ball_point(grid, inner_r, return_length=True))

    mask = (inner_counts >= 15) & (inner_counts <= 200)
    if not mask.any():
        print("[pocket]   pocket detection found no candidates; falling back to global box")
        return None, None

    outer_counts = np.array(tree.query_ball_point(grid[mask], outer_r, return_length=True))
    scores = inner_counts[mask].astype(float) * outer_counts

    candidates = grid[mask]
    best_pt = candidates[int(np.argmax(scores))]
    near = np.linalg.norm(candidates - best_pt, axis=1) < 12.0
    pocket_center = candidates[near].mean(axis=0).tolist()

    bs = [box_size] * 3
    print(f"[pocket]   pocket center: {[f'{c:.1f}' for c in pocket_center]}  box: {bs}")
    return pocket_center, bs


# ---------------------------------------------------------------------------
# Ligand preparation
# ---------------------------------------------------------------------------

def prepare_amino_acid_pdbqt(three_letter: str, smiles: str,
                              out_dir: Path) -> Path:
    """Generate 3-D structure and PDBQT for a single amino acid."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    pdbqt_path = out_dir / f"{three_letter}.pdbqt"
    if pdbqt_path.exists():
        return pdbqt_path

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES for {three_letter}: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    prep = MoleculePreparation(merge_these_atom_types=[])
    mol_setup_list = prep.prepare(mol)
    if not mol_setup_list:
        raise RuntimeError(f"Meeko could not prepare {three_letter}")

    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setup_list[0])
    if not is_ok:
        raise RuntimeError(f"Meeko PDBQT write failed for {three_letter}: {error_msg}")

    pdbqt_path.write_text(pdbqt_string)
    return pdbqt_path


# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------

def dock_ligand(receptor_pdbqt: Path, ligand_pdbqt: Path,
                center, box_size, poses_dir: Path,
                name: str, exhaustiveness: int = 16, n_poses: int = 9):
    """
    Run AutoDock Vina and return (best_affinity_kcal_mol, [all_affinities]).
    Saves top pose to poses_dir/name.pdbqt.
    """
    from vina import Vina

    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(str(receptor_pdbqt))
    v.set_ligand_from_file(str(ligand_pdbqt))
    v.compute_vina_maps(center=center, box_size=box_size)
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

    energies = v.energies(n_poses=n_poses)
    best = float(energies[0][0])
    all_scores = [float(e[0]) for e in energies]

    out_pose = poses_dir / f"{name}_best_pose.pdbqt"
    v.write_poses(str(out_pose), n_poses=1, overwrite=True)

    return best, all_scores


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def save_results(results: list, out_dir: Path, uniprot_id: str):
    """Save CSV, JSON, and a human-readable ranked table."""
    import csv

    ranked = sorted(results, key=lambda r: r["best_affinity_kcal_mol"])

    csv_path = out_dir / f"{uniprot_id}_docking_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "rank", "three_letter", "one_letter", "name",
            "best_affinity_kcal_mol", "all_affinities_kcal_mol"])
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            writer.writerow({
                "rank": rank,
                "three_letter": row["three_letter"],
                "one_letter":   row["one_letter"],
                "name":         row["name"],
                "best_affinity_kcal_mol": f"{row['best_affinity_kcal_mol']:.2f}",
                "all_affinities_kcal_mol": "|".join(f"{s:.2f}" for s in row["all_affinities"]),
            })

    json_path = out_dir / f"{uniprot_id}_docking_results.json"
    json_path.write_text(json.dumps(ranked, indent=2))

    table_path = out_dir / f"{uniprot_id}_docking_ranked.txt"
    lines = [
        f"AutoDock Vina Amino Acid Screening -- Target: {uniprot_id}",
        "=" * 65,
        f"{'Rank':<5} {'AA':<4} {'1L':<3} {'Name':<18} {'Best dG (kcal/mol)':>20}",
        "-" * 65,
    ]
    for rank, row in enumerate(ranked, 1):
        lines.append(
            f"{rank:<5} {row['three_letter']:<4} {row['one_letter']:<3} "
            f"{row['name']:<18} {row['best_affinity_kcal_mol']:>20.2f}"
        )
    lines.append("=" * 65)
    lines.append("(More negative dG = stronger predicted binder)")
    table_path.write_text("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\n[results]  CSV  -> {csv_path}")
    print(f"[results]  JSON -> {json_path}")
    print(f"[results]  table-> {table_path}")
    return ranked


def plot_results(ranked: list, out_dir: Path, uniprot_id: str):
    """Bar chart of binding affinities."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names  = [r["three_letter"] for r in ranked]
        scores = [r["best_affinity_kcal_mol"] for r in ranked]
        colors = ["#d62728" if s == min(scores) else "#1f77b4" for s in scores]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(names, scores, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Amino acid", fontsize=12)
        ax.set_ylabel("Best dG (kcal/mol)", fontsize=12)
        ax.set_title(f"AutoDock Vina: amino acid binding affinity to {uniprot_id}", fontsize=13)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        best_idx = scores.index(min(scores))
        ax.annotate(
            f"Best: {names[best_idx]}\n{min(scores):.2f} kcal/mol",
            xy=(best_idx, min(scores)),
            xytext=(best_idx + 1.5, min(scores) + 0.5),
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red", fontsize=10,
        )

        plt.tight_layout()
        png_path = out_dir / f"{uniprot_id}_docking_chart.png"
        fig.savefig(png_path, dpi=150)
        print(f"[results]  chart-> {png_path}")
        plt.close()
    except Exception as exc:
        print(f"[warn] Could not generate chart: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniprot",        default="C6XH98",  help="UniProt accession")
    parser.add_argument("--pdb",            default=None,       help="Path to pre-downloaded PDB (skips download)")
    parser.add_argument("--exhaustiveness", type=int, default=16, help="Vina exhaustiveness (default 16)")
    parser.add_argument("--n-poses",        type=int, default=9, help="Poses per amino acid (default 9)")
    parser.add_argument("--output",         default="results",  help="Output directory")
    parser.add_argument("--ligands-dir",    default="ligands",  help="Directory for ligand PDBQTs")
    parser.add_argument("--poses-dir",      default="poses",    help="Directory for output poses")
    args = parser.parse_args()

    out_dir     = Path(args.output);     out_dir.mkdir(parents=True, exist_ok=True)
    ligands_dir = Path(args.ligands_dir); ligands_dir.mkdir(parents=True, exist_ok=True)
    poses_dir   = Path(args.poses_dir);  poses_dir.mkdir(parents=True, exist_ok=True)

    if args.pdb:
        raw_pdb = Path(args.pdb)
        print(f"[struct]   using provided PDB: {raw_pdb}")
    else:
        raw_pdb = out_dir / f"{args.uniprot}_alphafold.pdb"
        if not raw_pdb.exists():
            download_alphafold(args.uniprot, raw_pdb)
        else:
            print(f"[struct]   found cached PDB: {raw_pdb}")

    clean    = out_dir / f"{args.uniprot}_clean.pdb"
    receptor = out_dir / f"{args.uniprot}_receptor.pdbqt"
    clean_pdb(raw_pdb, clean)
    pdb_to_pdbqt_receptor(clean, receptor)

    center, box_size = detect_binding_pocket(clean)
    if center is None:
        print("[box]      falling back to global protein box")
        center, box_size = get_protein_box(clean)

    print(f"\n[ligands]  preparing {len(AMINO_ACIDS)} amino acid ligands ...")
    ligand_pdbqts = {}
    failed = []
    for code, (full_name, smiles) in AMINO_ACIDS.items():
        try:
            p = prepare_amino_acid_pdbqt(code, smiles, ligands_dir)
            ligand_pdbqts[code] = p
            print(f"[ligands]  {code} ({full_name}) ok")
        except Exception as exc:
            print(f"[ligands]  {code} FAILED: {exc}")
            failed.append(code)

    print(f"\n[docking]  exhaustiveness={args.exhaustiveness}  n_poses={args.n_poses}")
    results = []
    for code, (full_name, _) in AMINO_ACIDS.items():
        if code in failed:
            continue
        sys.stdout.write(f"[docking]  {code} ... ")
        sys.stdout.flush()
        try:
            best, all_scores = dock_ligand(
                receptor, ligand_pdbqts[code],
                center, box_size,
                poses_dir, name=code,
                exhaustiveness=args.exhaustiveness,
                n_poses=args.n_poses,
            )
            print(f"{best:.2f} kcal/mol")
            results.append({
                "three_letter": code,
                "one_letter":   SINGLE_LETTER[code],
                "name":         full_name,
                "best_affinity_kcal_mol": best,
                "all_affinities": all_scores,
            })
        except Exception as exc:
            print(f"ERROR: {exc}")

    if not results:
        print("[error]    No successful docking runs. Exiting.")
        sys.exit(1)

    ranked = save_results(results, out_dir, args.uniprot)
    plot_results(ranked, out_dir, args.uniprot)

    best = ranked[0]
    print(f"\n*  Best predicted binder: {best['name']} ({best['three_letter']}) "
          f"dG = {best['best_affinity_kcal_mol']:.2f} kcal/mol")


if __name__ == "__main__":
    main()
