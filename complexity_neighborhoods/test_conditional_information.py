"""Pilot conditional-neighborhood test using public CAMELS halo catalogs.

The test tracks resolved subhalos across adjacent late-time snapshots, measures
new k-nearest-neighbor relations conditional on the previous neighborhood, and
compares that novelty with the physical-volume expansion of the same local
Lagrangian patch.  Dense, satellite-rich regions serve as the bound-environment
comparison and low-density central subhalos as the field comparison.

This is a test inside a LambdaCDM simulation, not evidence that information
causes expansion.  It tests the narrower local prediction that relational
novelty should accompany expansion and be suppressed by binding.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.integrate import quad
from scipy.stats import spearmanr


SNAPSHOTS = (86, 88, 90)
BASE_ROOT = "https://users.flatironinstitute.org/~camels/Sims"
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
MIN_PARTICLES = 100
NEIGHBORS = 16
KERNEL_CANDIDATES = 64
MAX_MATCH_DISTANCE_MPC_H = 0.50
MAX_MASS_RATIO = 3.0


@dataclass
class Catalog:
    snapshot: int
    scale_factor: float
    redshift: float
    box_mpc_h: float
    position: np.ndarray
    velocity: np.ndarray
    mass: np.ndarray
    anchor_id: np.ndarray
    satellite: np.ndarray
    omega_m: float
    omega_lambda: float
    hubble_parameter: float


def cache_directory(
    realization: int, suite: str = "IllustrisTNG", simulation_set: str = "CV"
) -> Path:
    # Keep compatibility with the already-downloaded pilot data.
    legacy = DATA_ROOT / "camels_cv0"
    if suite == "IllustrisTNG" and simulation_set == "CV" and realization == 0 and legacy.exists():
        return legacy
    return DATA_ROOT / "camels" / suite / simulation_set / f"{simulation_set}_{realization}"


def suite_url(suite: str, simulation_set: str) -> str:
    # The current public tree retains an extra generation component for the
    # gravity-only CV directories, but not for the accessible LH directories.
    if suite == "IllustrisTNG_DM" and simulation_set == "CV":
        return f"{BASE_ROOT}/{suite}/L25n256/{simulation_set}"
    return f"{BASE_ROOT}/{suite}/{simulation_set}"


def ensure_data(
    realization: int, suite: str = "IllustrisTNG", simulation_set: str = "CV"
) -> None:
    cache = cache_directory(realization, suite, simulation_set)
    cache.mkdir(parents=True, exist_ok=True)
    for snapshot in SNAPSHOTS:
        target = cache / f"groups_{snapshot:03d}.hdf5"
        if not target.exists() or target.stat().st_size == 0:
            print(f"Downloading {suite}/{simulation_set}_{realization}/{target.name} ...")
            urlretrieve(
                f"{suite_url(suite, simulation_set)}/{simulation_set}_{realization}/{target.name}",
                target,
            )


def load_catalog(
    snapshot: int,
    realization: int,
    suite: str = "IllustrisTNG",
    simulation_set: str = "CV",
) -> Catalog:
    path = cache_directory(realization, suite, simulation_set) / f"groups_{snapshot:03d}.hdf5"
    with h5py.File(path, "r") as handle:
        header = handle["Header"].attrs
        lengths = handle["Subhalo/SubhaloLen"][:]
        keep = lengths >= MIN_PARTICLES
        group_number = handle["Subhalo/SubhaloGrNr"][:]
        first_subhalo = handle["Group/GroupFirstSub"][:]
        subhalo_index = np.arange(len(lengths))
        central = subhalo_index == first_subhalo[group_number]
        return Catalog(
            snapshot=snapshot,
            scale_factor=float(header["Time"]),
            redshift=float(header["Redshift"]),
            box_mpc_h=float(header["BoxSize"]) / 1000.0,
            position=handle["Subhalo/SubhaloPos"][:][keep] / 1000.0,
            velocity=handle["Subhalo/SubhaloVel"][:][keep],
            mass=handle["Subhalo/SubhaloMass"][:][keep],
            anchor_id=handle["Subhalo/SubhaloIDMostbound"][:][keep],
            satellite=(~central)[keep],
            omega_m=float(header["Omega0"]),
            omega_lambda=float(header["OmegaLambda"]),
            hubble_parameter=float(header["HubbleParam"]),
        )


def periodic_delta(delta: np.ndarray, box: float) -> np.ndarray:
    return delta - box * np.rint(delta / box)


def match_catalogs(old: Catalog, new: Catalog) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mutual-nearest positional match with a broad mass-consistency cut."""
    old_tree = cKDTree(old.position, boxsize=old.box_mpc_h)
    new_tree = cKDTree(new.position, boxsize=new.box_mpc_h)
    distance, old_to_new = new_tree.query(old.position, k=1)
    _back_distance, new_to_old = old_tree.query(new.position, k=1)
    old_index = np.arange(len(old.position))
    mutual = new_to_old[old_to_new] == old_index
    ratio = new.mass[old_to_new] / old.mass
    mass_ok = (ratio >= 1.0 / MAX_MASS_RATIO) & (ratio <= MAX_MASS_RATIO)
    keep = mutual & mass_ok & (distance <= MAX_MATCH_DISTANCE_MPC_H)
    return old_index[keep], old_to_new[keep], distance[keep]


def local_log_volume(relative_vectors: np.ndarray) -> np.ndarray:
    """Log of sqrt(det(covariance)), a three-volume scale for each patch."""
    covariance = np.einsum("nki,nkj->nij", relative_vectors, relative_vectors) / relative_vectors.shape[1]
    sign, logdet = np.linalg.slogdet(covariance)
    if np.any(sign <= 0):
        raise ValueError("Degenerate local covariance encountered")
    return 0.5 * logdet


def ballistic_position(old: Catalog, new: Catalog, old_idx: np.ndarray) -> tuple[np.ndarray, float]:
    """Propagate the old peculiar velocity through the FLRW interval.

    SubhaloVel is the peculiar proper velocity in km/s and SubhaloPos is in
    comoving Mpc/h.  Holding the measured old velocity fixed gives
    dx = (v/100) integral[da/(a^2 E(a))] in comoving Mpc/h.
    """
    if not math.isclose(old.omega_m, new.omega_m) or not math.isclose(
        old.omega_lambda, new.omega_lambda
    ):
        raise ValueError("Cosmology changed between snapshots")

    def inverse_a2_e(a: float) -> float:
        e = math.sqrt(old.omega_m / a**3 + old.omega_lambda)
        return 1.0 / (a * a * e)

    displacement_integral = quad(
        inverse_a2_e, old.scale_factor, new.scale_factor, epsabs=1.0e-12
    )[0]
    predicted = old.position[old_idx] + old.velocity[old_idx] * (displacement_integral / 100.0)
    return np.mod(predicted, old.box_mpc_h), displacement_integral


def gaussian_weights(
    focal_position: np.ndarray,
    candidate_positions: np.ndarray,
    bandwidth: float,
    box: float,
) -> np.ndarray:
    delta = periodic_delta(candidate_positions - focal_position, box)
    radius = np.linalg.norm(delta, axis=1)
    weight = np.exp(-0.5 * (radius / bandwidth) ** 2)
    return weight / weight.sum()


def js_bits(first: np.ndarray, second: np.ndarray) -> float:
    midpoint = 0.5 * (first + second)
    return float(
        0.5
        * (
            np.sum(first * np.log2(first / midpoint))
            + np.sum(second * np.log2(second / midpoint))
        )
    )


def analyze_pair(old: Catalog, new: Catalog) -> dict[str, float]:
    old_idx, new_idx, match_distance = match_catalogs(old, new)
    old_pos = old.position[old_idx]
    new_pos = new.position[new_idx]
    predicted_pos, displacement_integral = ballistic_position(old, new, old_idx)
    count = len(old_idx)
    if count <= NEIGHBORS + 1:
        raise ValueError("Too few matched subhalos")

    old_tree = cKDTree(old_pos, boxsize=old.box_mpc_h)
    new_tree = cKDTree(new_pos, boxsize=new.box_mpc_h)
    predicted_tree = cKDTree(predicted_pos, boxsize=old.box_mpc_h)
    old_dist, old_neighbors = old_tree.query(old_pos, k=NEIGHBORS + 1)
    _new_dist, new_neighbors = new_tree.query(new_pos, k=NEIGHBORS + 1)
    old_neighbors = old_neighbors[:, 1:]
    new_neighbors = new_neighbors[:, 1:]

    retained = np.asarray(
        [len(set(before).intersection(after)) for before, after in zip(old_neighbors, new_neighbors)],
        dtype=float,
    )
    new_count = NEIGHBORS - retained
    retention_rate = float(retained.sum() / (count * NEIGHBORS))
    new_edge_probability = NEIGHBORS * (1.0 - retention_rate) / (count - 1 - NEIGHBORS)
    conditional_new_bits = new_count * -math.log2(new_edge_probability)

    # Continuous robustness estimator: Jensen-Shannon change of Gaussian
    # neighborhood weights over the union of nearby candidates at both times.
    old_kernel_distance, old_kernel_neighbors = old_tree.query(old_pos, k=KERNEL_CANDIDATES + 1)
    _new_kernel_distance, new_kernel_neighbors = new_tree.query(new_pos, k=KERNEL_CANDIDATES + 1)
    _predicted_kernel_distance, predicted_kernel_neighbors = predicted_tree.query(
        predicted_pos, k=KERNEL_CANDIDATES + 1
    )
    old_kernel_neighbors = old_kernel_neighbors[:, 1:]
    new_kernel_neighbors = new_kernel_neighbors[:, 1:]
    predicted_kernel_neighbors = predicted_kernel_neighbors[:, 1:]
    kernel_js_bits = np.empty(count)
    residual_js_bits = np.empty(count)
    for i in range(count):
        candidates = np.union1d(
            np.union1d(old_kernel_neighbors[i], new_kernel_neighbors[i]),
            predicted_kernel_neighbors[i],
        )
        candidates = candidates[candidates != i]
        bandwidth = max(old_kernel_distance[i, NEIGHBORS], 1.0e-8)
        old_weight = gaussian_weights(old_pos[i], old_pos[candidates], bandwidth, old.box_mpc_h)
        new_weight = gaussian_weights(new_pos[i], new_pos[candidates], bandwidth, new.box_mpc_h)
        predicted_weight = gaussian_weights(
            predicted_pos[i], predicted_pos[candidates], bandwidth, old.box_mpc_h
        )
        kernel_js_bits[i] = js_bits(old_weight, new_weight)
        residual_js_bits[i] = js_bits(predicted_weight, new_weight)

    focal = np.arange(count)[:, None]
    old_relative = periodic_delta(old_pos[old_neighbors] - old_pos[focal], old.box_mpc_h)
    # Follow the same old neighbors into the new snapshot: a Lagrangian patch.
    new_relative_same = periodic_delta(new_pos[old_neighbors] - new_pos[focal], new.box_mpc_h)
    log_volume_old = local_log_volume(old_relative)
    log_volume_new = local_log_volume(new_relative_same)
    background_log_volume_growth = 3.0 * math.log(new.scale_factor / old.scale_factor)
    physical_log_volume_growth = log_volume_new - log_volume_old + background_log_volume_growth
    expansion_ratio = physical_log_volume_growth / background_log_volume_growth

    # The old kNN radius is an inverse-density environment proxy.
    density_proxy = NEIGHBORS / np.maximum(old_dist[:, -1], 1.0e-12) ** 3
    low_cut, high_cut = np.quantile(density_proxy, [0.25, 0.75])
    low_density = density_proxy <= low_cut
    high_density = density_proxy >= high_cut
    satellite = old.satellite[old_idx]
    field = low_density & ~satellite
    bound_proxy = high_density | satellite

    rho_expansion, p_expansion = spearmanr(conditional_new_bits, expansion_ratio)
    rho_density, p_density = spearmanr(conditional_new_bits, density_proxy)
    rho_kernel_expansion, p_kernel_expansion = spearmanr(kernel_js_bits, expansion_ratio)
    rho_kernel_density, p_kernel_density = spearmanr(kernel_js_bits, density_proxy)
    rho_residual_expansion, p_residual_expansion = spearmanr(residual_js_bits, expansion_ratio)
    rho_residual_density, p_residual_density = spearmanr(residual_js_bits, density_proxy)
    exact_anchor = old.anchor_id[old_idx] == new.anchor_id[new_idx]
    trackable_anchor = np.isin(old.anchor_id[old_idx], new.anchor_id)
    anchor_recovery = float(exact_anchor[trackable_anchor].mean()) if np.any(trackable_anchor) else float("nan")

    def median(mask: np.ndarray, values: np.ndarray) -> float:
        return float(np.median(values[mask])) if np.any(mask) else float("nan")

    return {
        "matched": float(count),
        "median_match_distance": float(np.median(match_distance)),
        "exact_anchor_fraction": float(exact_anchor.mean()),
        "trackable_anchor_fraction": float(trackable_anchor.mean()),
        "anchor_recovery": anchor_recovery,
        "retention_rate": retention_rate,
        "displacement_integral": displacement_integral,
        "rho_novelty_expansion": float(rho_expansion),
        "p_novelty_expansion": float(p_expansion),
        "rho_novelty_density": float(rho_density),
        "p_novelty_density": float(p_density),
        "rho_kernel_expansion": float(rho_kernel_expansion),
        "p_kernel_expansion": float(p_kernel_expansion),
        "rho_kernel_density": float(rho_kernel_density),
        "p_kernel_density": float(p_kernel_density),
        "rho_residual_expansion": float(rho_residual_expansion),
        "p_residual_expansion": float(p_residual_expansion),
        "rho_residual_density": float(rho_residual_density),
        "p_residual_density": float(p_residual_density),
        "field_count": float(field.sum()),
        "bound_count": float(bound_proxy.sum()),
        "field_novelty": median(field, conditional_new_bits),
        "bound_novelty": median(bound_proxy, conditional_new_bits),
        "field_expansion": median(field, expansion_ratio),
        "bound_expansion": median(bound_proxy, expansion_ratio),
        "satellite_novelty": median(satellite, conditional_new_bits),
        "central_novelty": median(~satellite, conditional_new_bits),
        "field_kernel_js": median(field, kernel_js_bits),
        "bound_kernel_js": median(bound_proxy, kernel_js_bits),
        "field_residual_js": median(field, residual_js_bits),
        "bound_residual_js": median(bound_proxy, residual_js_bits),
        "all_kernel_js": median(np.ones(count, dtype=bool), kernel_js_bits),
        "all_residual_js": median(np.ones(count, dtype=bool), residual_js_bits),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--realizations", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--suite", choices=("IllustrisTNG", "IllustrisTNG_DM"), default="IllustrisTNG"
    )
    parser.add_argument("--simulation-set", choices=("CV", "LH"), default="CV")
    args = parser.parse_args()
    print("CAMELS phase-space-conditioned neighborhood test")
    print(f"Selection: SubhaloLen >= {MIN_PARTICLES}; k={NEIGHBORS}")
    print("Bound proxy: satellite or top-quartile local density")
    print()
    summaries = []
    for realization in args.realizations:
        ensure_data(realization, args.suite, args.simulation_set)
        catalogs = [
            load_catalog(snapshot, realization, args.suite, args.simulation_set)
            for snapshot in SNAPSHOTS
        ]
        for old, new in zip(catalogs, catalogs[1:]):
            result = analyze_pair(old, new)
            result["realization"] = float(realization)
            result["pair"] = f"{old.snapshot}->{new.snapshot}"
            summaries.append(result)
            print(
                f"{args.suite}/{args.simulation_set}_{realization} "
                f"snapshots {old.snapshot}->{new.snapshot}: "
                f"z={old.redshift:.4f}->{new.redshift:.4f}"
            )
            print(
                f"  matched={int(result['matched'])}, median displacement={result['median_match_distance']:.4f} Mpc/h, "
                f"persistent anchors={result['trackable_anchor_fraction']:.1%}, "
                f"match recovery={result['anchor_recovery']:.1%}"
            )
            print(f"  neighbor retention={result['retention_rate']:.1%}")
            print(
                f"  novelty vs local expansion: rho={result['rho_novelty_expansion']:+.3f}, "
                f"p={result['p_novelty_expansion']:.3g}"
            )
            print(
                f"  novelty vs density:         rho={result['rho_novelty_density']:+.3f}, "
                f"p={result['p_novelty_density']:.3g}"
            )
            print(
                f"  raw kernel change vs expansion: rho={result['rho_kernel_expansion']:+.3f}; "
                f"vs density: rho={result['rho_kernel_density']:+.3f}"
            )
            print(
                f"  phase-conditioned residual:     rho(expansion)={result['rho_residual_expansion']:+.3f}; "
                f"rho(density)={result['rho_residual_density']:+.3f}"
            )
            print(
                f"  field (n={int(result['field_count'])}): median novelty={result['field_novelty']:.3f} bits, "
                f"expansion/global={result['field_expansion']:.3f}"
            )
            print(
                f"  bound (n={int(result['bound_count'])}): median novelty={result['bound_novelty']:.3f} bits, "
                f"expansion/global={result['bound_expansion']:.3f}"
            )
            print(
                f"  raw field/bound JS:      {result['field_kernel_js']:.6f}/{result['bound_kernel_js']:.6f} bits"
            )
            print(
                f"  residual field/bound JS: {result['field_residual_js']:.6f}/{result['bound_residual_js']:.6f} bits"
            )
            print(
                f"  overall raw/residual JS: {result['all_kernel_js']:.6f}/{result['all_residual_js']:.6f} bits"
            )
            print()

    if len(args.realizations) > 1:
        print("Replication summary (mean +/- sample SD across intervals and realizations)")
        for key in (
            "field_expansion",
            "bound_expansion",
            "field_kernel_js",
            "bound_kernel_js",
            "field_residual_js",
            "bound_residual_js",
            "rho_residual_expansion",
            "rho_residual_density",
        ):
            values = np.asarray([row[key] for row in summaries])
            print(f"  {key}: {values.mean():.6g} +/- {values.std(ddof=1):.6g}")


if __name__ == "__main__":
    main()
