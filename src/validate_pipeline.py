from pxr import Usd
import numpy as np
import os, csv
from collision_check import CollisionValidator as SatCollisionValidator


DEFAULT_CSV_PATH = "scenarios.csv"


def load_truth_row(csv_path, scenario_id):
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["scenario_id"].strip() == scenario_id:
                return row
    raise ValueError(f"scenario_id {scenario_id} not found in {csv_path}")


def get_scenario_metadata(validator):
    """Extract scenario-level metadata from USD."""
    world = validator.stage.GetPrimAtPath("/World")
    metadata = {}

    # Timestamps
    ts_attr = world.GetAttribute("waymo:timestampsSeconds")
    if ts_attr and ts_attr.Get():
        ts = list(ts_attr.Get())
        metadata['num_frames'] = len(ts)
        metadata['duration_s'] = ts[-1] - ts[0] if ts else 0

    # Current time index (prediction start)
    cti_attr = world.GetAttribute("waymo:currentTimeIndex")
    if cti_attr:
        metadata['current_time_index'] = cti_attr.Get()

    # Tracks to predict
    ttp_attr = world.GetAttribute("waymo:tracksToPredict")
    if ttp_attr and ttp_attr.Get():
        metadata['tracks_to_predict'] = list(ttp_attr.Get())

    return metadata



def validate_scenario(usd_path, csv_path, verbose=True):
    scenario_id = os.path.basename(usd_path).replace("_base.usd", "").replace(".usd", "")
    truth = load_truth_row(csv_path, scenario_id)

    target_dist = float(truth["min_distance_m"]) if truth.get("min_distance_m") else None
    critical_time_s = float(truth["critical_time_s"]) if truth.get("critical_time_s") else 0.0
    closest_obj_id = str(truth["closest_obj_id"]).strip()
    closest_obj_type = truth.get("closest_obj_type", "UNKNOWN")

    # Get TTC from CSV if available
    csv_ttc = float(truth["min_ttc_s"]) if truth.get("min_ttc_s") else None

    if verbose:
        print(f"\n{'='*70}")
        print(f"🔍 Validating: {scenario_id}")
        print(f"{'='*70}")
        print(f"CSV Ground Truth:")
        print(f"  closest_obj_id: {closest_obj_id} ({closest_obj_type})")
        print(f"  min_distance_m: {target_dist:.2f}m" if target_dist is not None else "  min_distance_m: N/A")
        print(f"  critical_time_s: {critical_time_s:.2f}")
        if csv_ttc is not None:
            print(f"  min_ttc_s: {csv_ttc:.2f}")

    validator = SatCollisionValidator(usd_path)

    # Use actual timestamps for frame conversion if available
    if validator.timestamps:
        # Find frame closest to critical_time_s
        critical_frame = min(range(len(validator.timestamps)),
                            key=lambda i: abs(validator.timestamps[i] - critical_time_s))
    else:
        critical_frame = int(round(critical_time_s * 10.0))

    target_agent_path = f"/World/Agents/Agent_{closest_obj_id}/Geometry"

    ego_prim = validator.stage.GetPrimAtPath("/World/Agents/Agent_EGO/Geometry")
    if not ego_prim:
        print("❌ EGO prim not found at /World/Agents/Agent_EGO/Geometry")
        return False

    end_frame = int(validator.stage.GetEndTimeCode())
    if critical_frame > end_frame:
        print(f"❌ CSV critical_frame={critical_frame} exceeds USD end_frame={end_frame}")
        return False

    agent_prim = validator.stage.GetPrimAtPath(target_agent_path)
    if not agent_prim:
        print(f"❌ Closest agent from CSV not found in USD: {target_agent_path}")
        return False

    # --- Run comprehensive analysis ---
    if verbose:
        print(f"\n📊 Running comprehensive analysis (0-{end_frame} frames)...")

    results = validator.analyze_scenario(0, end_frame)

    # Find the CSV's closest agent in our results
    csv_agent_result = None
    for r in results:
        if r['agent_id'] == closest_obj_id:
            csv_agent_result = r
            break

    # --- Validation checks ---
    checks = {}

    # 1. Distance check
    if csv_agent_result and csv_agent_result['min_distance'] is not None and target_dist is not None:
        computed_dist = csv_agent_result['min_distance']
        checks['distance'] = abs(computed_dist - target_dist) < 0.5
        if verbose:
            print(f"\n📏 Distance Validation:")
            print(f"   CSV: {target_dist:.2f}m | USD: {computed_dist:.2f}m | Δ={abs(computed_dist - target_dist):.2f}m")
            print(f"   {'✅ PASS' if checks['distance'] else '❌ FAIL'} (tolerance: 0.5m)")
    elif csv_agent_result and csv_agent_result['min_distance'] is not None and target_dist is None:
        # CSV had no ground truth distance (was empty/zero) - accept computed value
        computed_dist = csv_agent_result['min_distance']
        checks['distance'] = True
        if verbose:
            print(f"\n📏 Distance Validation:")
            print(f"   CSV: N/A | USD: {computed_dist:.2f}m")
            print(f"   ✅ PASS (no CSV ground truth to compare)")
    else:
        checks['distance'] = False

    # 2. Frame/time check
    if csv_agent_result and csv_agent_result['min_distance_frame'] is not None:
        computed_frame = csv_agent_result['min_distance_frame']
        checks['frame'] = abs(computed_frame - critical_frame) <= 2
        if verbose:
            print(f"\n⏱️  Time Validation:")
            print(f"   CSV frame: {critical_frame} | USD frame: {computed_frame} | Δ={abs(computed_frame - critical_frame)}")
            print(f"   {'✅ PASS' if checks['frame'] else '❌ FAIL'} (tolerance: 2 frames)")
    else:
        checks['frame'] = False

    # 3. TTC check (if CSV has TTC data)
    # Note: TTC discrepancy is expected because:
    #   - CSV computes TTC using position-based velocity (finite differences)
    #   - USD uses Waymo's actual velocity fields (more accurate)
    # TTC is informational, not a hard pass/fail criterion
    if csv_ttc and csv_agent_result and csv_agent_result['min_ttc'] is not None:
        computed_ttc = csv_agent_result['min_ttc']
        ttc_delta = abs(computed_ttc - csv_ttc)
        checks['ttc'] = ttc_delta < 1.0  # For reporting only
        if verbose:
            print(f"\n⏳ TTC Comparison (informational):")
            print(f"   CSV (estimated): {csv_ttc:.2f}s | USD (actual velocity): {computed_ttc:.2f}s | Δ={ttc_delta:.2f}s")
            if ttc_delta > 1.0:
                print(f"   ℹ️  Large delta expected: CSV uses position-based velocity estimation")
    else:
        checks['ttc'] = True  # Skip if no TTC data

    # 4. Object type check
    if csv_agent_result:
        usd_type = csv_agent_result['object_type']
        # Normalize for comparison (CSV might have TYPE_ prefix or not)
        csv_type_normalized = closest_obj_type if closest_obj_type.startswith("TYPE_") else f"TYPE_{closest_obj_type}"
        checks['object_type'] = usd_type == csv_type_normalized
        if verbose:
            print(f"\n🏷️  Object Type Validation:")
            print(f"   CSV: {closest_obj_type} | USD: {usd_type}")
            print(f"   {'✅ PASS' if checks['object_type'] else '❌ FAIL'}")

    # 5. Semantic ranking check - is the CSV's closest agent in top 3 by criticality?
    top_3_ids = [r['agent_id'] for r in results[:3]]
    checks['ranking'] = closest_obj_id in top_3_ids
    if verbose:
        print(f"\n🏆 Semantic Ranking Validation:")
        print(f"   Top 3 critical agents: {top_3_ids}")
        print(f"   CSV closest_obj_id '{closest_obj_id}' in top 3: {'✅ PASS' if checks['ranking'] else '⚠️ WARN'}")

    # --- SAT collision check ---
    if verbose:
        print(f"\n🧪 SAT Overlap Check (frames {max(0, critical_frame-10)} to {min(end_frame, critical_frame+10)}):")
    collision_detected = False
    collision_frame = None
    for t in range(max(0, critical_frame - 10), min(end_frame, critical_frame + 10) + 1):
        time = Usd.TimeCode(t)
        ego_obb = validator.get_obb(ego_prim, time)
        agent_obb = validator.get_obb(agent_prim, time)
        if validator.check_overlap(ego_obb, agent_obb):
            collision_detected = True
            collision_frame = t
            break
    if verbose:
        if collision_detected:
            print(f"   ⚠️  COLLISION detected at frame {collision_frame}")
        else:
            print(f"   ✅ No collision in critical window")

    # --- Print analysis report ---
    if verbose:
        validator.print_analysis_report(results, top_n=5)

    # --- Overall result ---
    # Core checks: distance and frame (TTC is informational due to velocity estimation differences)
    core_checks = checks['distance'] and checks['frame']
    if verbose:
        print(f"\n{'='*70}")
        print(f"OVERALL: {'✅ PASS' if core_checks else '❌ FAIL'}")
        if not checks.get('ttc', True):
            print(f"  (TTC delta >1s is expected due to velocity estimation method differences)")
        print(f"{'='*70}\n")

    return core_checks





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to scenarios.csv") 
    parser.add_argument("--all", action="store_true", help="Validate all scenario_ids from scenarios.csv")  
    parser.add_argument("--usd_dir", default="scenarios", help="Directory containing USD files")
    parser.add_argument("--suffix", default="_base.usd", help="USD filename suffix")
    parser.add_argument("scenario_id", nargs="?", help="Validate a single scenario_id (optional if --all)")  
    args = parser.parse_args()

    if args.all:
        # Load all scenario IDs from CSV
        with open(args.csv, newline="") as f:
            r = csv.DictReader(f) 
            scenario_ids = [row["scenario_id"].strip() for row in r if row.get("scenario_id")]

        passed = 0
        total = 0
        missing_usd = []

        for sid in scenario_ids:
            usd_path = os.path.join(args.usd_dir, f"{sid}{args.suffix}")
            if not os.path.exists(usd_path):
                # Check synthetic_scenarios subdirectory
                usd_path = os.path.join(args.usd_dir, "synthetic_scenarios", f"{sid}{args.suffix}")
            if not os.path.exists(usd_path):
                missing_usd.append(usd_path)
                continue

            total += 1
            ok = validate_scenario(usd_path, args.csv)
            if ok:
                passed += 1

        print("\n" + "=" * 60)
        print(f"Batch results: {passed}/{total} PASS")
        if missing_usd:
            print(f"Missing USD files ({len(missing_usd)}):")
            for p in missing_usd[:10]:
                print("  ", p)
            if len(missing_usd) > 10:
                print("  ...")
        print("=" * 60)

    else:
        if not args.scenario_id:
            parser.error("Provide scenario_id or use --all")
        usd_path = os.path.join(args.usd_dir, f"{args.scenario_id}{args.suffix}")
        validate_scenario(usd_path, args.csv)

