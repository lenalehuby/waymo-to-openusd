"""
intervention_layer.py - USD Intervention Layer for Counterfactual Analysis

Creates intervention.usd layers that override base.usd to deactivate specific actors.
Uses USD's composition arcs for sparse overrides without rewriting the base scene.

Usage:
    # Create intervention for single actor removal
    python intervention_layer.py --base scenarios/abc123_base.usd --remove 717

    # Create intervention for multiple actor removal (Tier 2)
    python intervention_layer.py --base scenarios/abc123_base.usd --remove 717 823

    # Batch create interventions for all scenarios
    python intervention_layer.py --batch --removal_set 717
"""

from pxr import Usd, Sdf, UsdGeom
import os
import argparse
from datetime import datetime
from typing import Set, Optional, List, Dict, Any


def create_intervention_layer(
    base_usd_path: str,
    removal_set: Set,
    output_path: Optional[str] = None,
    collision_threshold_m: Optional[float] = None,
    intervention_reason: Optional[str] = None,
):
    """
    Create an intervention USD layer that deactivates specified actors.

    USD Composition Strategy:
    - intervention.usd sublayers base.usd (base is stronger by default)
    - We use 'over' prims to override specific actors
    - Setting 'active = false' removes actors from stage traversal

    Also stores intervention metadata in /World customData["intervention"]
    following the RLxUSD pattern of self-describing artifacts.

    Args:
        base_usd_path: Path to the base USD file (e.g., scenario_base.usd)
        removal_set: Set of agent IDs to deactivate (e.g., {717, 823})
        output_path: Output path for intervention USD (default: auto-generated)
        collision_threshold_m: Optional collision threshold used (for metadata)
        intervention_reason: Optional reason/description for the intervention

    Returns:
        Path to created intervention USD file
    """
    if not os.path.exists(base_usd_path):
        raise FileNotFoundError(f"Base USD not found: {base_usd_path}")

    # Generate output path if not specified
    if output_path is None:
        base_dir = os.path.dirname(base_usd_path)
        base_name = os.path.basename(base_usd_path).replace("_base.usd", "")
        removal_suffix = "_".join(str(x) for x in sorted(removal_set))
        output_path = os.path.join(base_dir, f"{base_name}_intervention_{removal_suffix}.usd")

    # Create a new layer for the intervention
    intervention_layer = Sdf.Layer.CreateNew(output_path)

    # Add the base USD as a sublayer (intervention layer is stronger/on top)
    # The base path should be relative for portability
    base_filename = os.path.basename(base_usd_path)
    intervention_layer.subLayerPaths.append(f"./{base_filename}")

    # Open the base stage to verify agent paths exist
    base_stage = Usd.Stage.Open(base_usd_path)
    if not base_stage:
        raise RuntimeError(f"Failed to open base stage: {base_usd_path}")

    # Create override prims for each agent to deactivate
    # Also collect object types for metadata
    deactivated = []
    removal_set_types = []

    for agent_id in removal_set:
        # Handle both numeric and string IDs
        safe_id = str(agent_id).replace("-", "_")
        agent_path = f"/World/Agents/Agent_{safe_id}"

        # Verify agent exists in base
        base_prim = base_stage.GetPrimAtPath(agent_path)
        if not base_prim:
            print(f"  Warning: Agent {agent_id} not found at {agent_path}, skipping")
            continue

        # Get object type from base prim for metadata
        obj_type_attr = base_prim.GetAttribute("waymo:objectTypeString")
        obj_type = str(obj_type_attr.Get()) if obj_type_attr and obj_type_attr.Get() else "TYPE_UNKNOWN"
        removal_set_types.append(obj_type)

        # Create 'over' prim in intervention layer (doesn't define, just overrides)
        over_prim_spec = Sdf.CreatePrimInLayer(intervention_layer, agent_path)
        over_prim_spec.specifier = Sdf.SpecifierOver

        # Set active = false to deactivate the agent
        # This removes the prim and children from stage traversal
        over_prim_spec.SetInfo("active", False)

        # Also add a custom attribute documenting the deactivation
        # This helps with debugging and provenance tracking
        attr_spec = Sdf.AttributeSpec(
            over_prim_spec,
            "waymo:deactivatedBy",
            Sdf.ValueTypeNames.String
        )
        attr_spec.default = "intervention_layer"

        deactivated.append(agent_id)

    # Add metadata to the root
    root_spec = Sdf.CreatePrimInLayer(intervention_layer, "/World")
    root_spec.specifier = Sdf.SpecifierOver

    # Store removal set as metadata on /World
    removal_attr = Sdf.AttributeSpec(
        root_spec,
        "waymo:removalSet",
        Sdf.ValueTypeNames.IntArray
    )
    removal_attr.default = list(int(x) for x in removal_set if str(x).isdigit())

    # Store removal set as string array (handles non-numeric IDs like "EGO")
    removal_str_attr = Sdf.AttributeSpec(
        root_spec,
        "waymo:removalSetStr",
        Sdf.ValueTypeNames.StringArray
    )
    removal_str_attr.default = [str(x) for x in removal_set]

    # --- Store intervention metadata in customData["intervention"] ---
    # Following RLxUSD pattern of self-describing artifacts
    # Note: USD customData doesn't support Python lists directly in crate format
    # Convert lists to comma-separated strings for compatibility
    intervention_metadata = {
        "avxusd_version": "0.1",
        "base_scenario": base_filename,
        "removal_set": ",".join(str(x) for x in sorted(removal_set, key=str)),
        "removal_set_types": ",".join(removal_set_types),
        "created_at": datetime.now().isoformat(),
        "num_removed": len(deactivated),
    }

    # Add optional fields if provided
    if collision_threshold_m is not None:
        intervention_metadata["collision_threshold_m"] = float(collision_threshold_m)

    if intervention_reason is not None:
        intervention_metadata["intervention_reason"] = str(intervention_reason)

    # To set customData on a prim spec in Sdf, we need to use SetInfo
    # customData is stored as a dictionary
    root_spec.SetInfo("customData", {"intervention": intervention_metadata})

    # Save the layer
    intervention_layer.Save()

    print(f"Created intervention layer: {output_path}")
    print(f"  Base: {base_filename}")
    print(f"  Deactivated agents: {deactivated}")

    return output_path


def load_scenario_with_intervention(base_usd_path, intervention_usd_path=None, removal_set=None):
    """
    Load a scenario stage with intervention layer applied.

    Can either:
    1. Use an existing intervention USD file
    2. Create a temporary intervention for a given removal set

    Args:
        base_usd_path: Path to base USD
        intervention_usd_path: Path to existing intervention USD (optional)
        removal_set: Set of agent IDs to remove (used if no intervention_usd_path)

    Returns:
        Usd.Stage with intervention applied
    """
    if intervention_usd_path and os.path.exists(intervention_usd_path):
        # Load the intervention layer directly (it sublayers the base)
        stage = Usd.Stage.Open(intervention_usd_path)
    elif removal_set:
        # Create temporary intervention
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_intervention = os.path.join(temp_dir, "temp_intervention.usd")

        # Copy base to temp dir for relative path resolution
        import shutil
        temp_base = os.path.join(temp_dir, os.path.basename(base_usd_path))
        shutil.copy(base_usd_path, temp_base)

        create_intervention_layer(temp_base, removal_set, temp_intervention)
        stage = Usd.Stage.Open(temp_intervention)
    else:
        # No intervention, just open base
        stage = Usd.Stage.Open(base_usd_path)

    return stage


def verify_intervention(intervention_usd_path):
    """
    Verify an intervention layer is correctly configured.

    Checks:
    1. Sublayer reference to base exists
    2. Deactivated agents are not traversable
    3. Other agents remain active
    4. Intervention metadata in customData
    """
    print(f"\nVerifying: {intervention_usd_path}")

    stage = Usd.Stage.Open(intervention_usd_path)
    if not stage:
        print("  ERROR: Failed to open stage")
        return False

    # Get the root layer
    root_layer = stage.GetRootLayer()
    print(f"  Sublayers: {root_layer.subLayerPaths}")

    # Check removal set metadata (legacy attributes)
    world = stage.GetPrimAtPath("/World")
    if world:
        removal_attr = world.GetAttribute("waymo:removalSetStr")
        if removal_attr:
            removal_set = list(removal_attr.Get())
            print(f"  Removal set (attr): {removal_set}")

    # Check intervention metadata in customData
    intervention_meta = load_intervention_metadata(stage)
    if intervention_meta:
        print(f"  Intervention metadata (customData):")
        print(f"    avxusd_version: {intervention_meta.get('avxusd_version')}")
        print(f"    base_scenario: {intervention_meta.get('base_scenario')}")
        # removal_set and removal_set_types are stored as comma-separated strings
        removal_set_str = intervention_meta.get('removal_set', '')
        removal_types_str = intervention_meta.get('removal_set_types', '')
        print(f"    removal_set: [{removal_set_str}]")
        print(f"    removal_set_types: [{removal_types_str}]")
        print(f"    created_at: {intervention_meta.get('created_at')}")
        if 'collision_threshold_m' in intervention_meta:
            print(f"    collision_threshold_m: {intervention_meta.get('collision_threshold_m')}")
        if 'intervention_reason' in intervention_meta:
            print(f"    intervention_reason: {intervention_meta.get('intervention_reason')}")
    else:
        print("  Warning: No intervention metadata in customData")

    # Check counterfactual results in customData
    counterfactual_results = load_counterfactual_results(stage)
    if counterfactual_results:
        status = "PASSED" if counterfactual_results.get('test_passed') else "FAILED"
        print(f"  Counterfactual results (customData):")
        print(f"    test_passed: {status}")
        if 'new_min_distance_m' in counterfactual_results:
            print(f"    new_min_distance_m: {counterfactual_results.get('new_min_distance_m'):.2f}")
        if 'new_min_ttc_s' in counterfactual_results:
            print(f"    new_min_ttc_s: {counterfactual_results.get('new_min_ttc_s'):.2f}")
        if 'is_oracle_minimal' in counterfactual_results:
            print(f"    is_oracle_minimal: {counterfactual_results.get('is_oracle_minimal')}")
        if 'strategy_used' in counterfactual_results:
            print(f"    strategy_used: {counterfactual_results.get('strategy_used')}")
        if 'agents_tested' in counterfactual_results:
            print(f"    agents_tested: {counterfactual_results.get('agents_tested')}")
        if 'distance_improvement_m' in counterfactual_results:
            print(f"    distance_improvement_m: {counterfactual_results.get('distance_improvement_m'):.2f}")
        if 'tested_at' in counterfactual_results:
            print(f"    tested_at: {counterfactual_results.get('tested_at')}")

    # Count active vs inactive agents
    agents_root = stage.GetPrimAtPath("/World/Agents")
    if agents_root:
        active_count = 0
        inactive_count = 0

        for child in agents_root.GetChildren():
            if child.IsActive():
                active_count += 1
            else:
                inactive_count += 1
                print(f"  Deactivated: {child.GetPath()}")

        print(f"  Active agents: {active_count}")
        print(f"  Inactive agents: {inactive_count}")

    return True


def load_intervention_metadata(stage: Usd.Stage) -> Optional[Dict[str, Any]]:
    """
    Load intervention metadata from /World customData["intervention"].

    Returns:
        Dictionary with intervention metadata, or None if not found.
    """
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        return None

    custom_data = world_prim.GetCustomData()
    if not custom_data:
        return None

    intervention_data = custom_data.get("intervention")
    if intervention_data:
        # Convert VtDictionary to regular dict
        return dict(intervention_data)
    return None


def write_counterfactual_results(
    intervention_path: str,
    test_passed: bool,
    new_min_distance_m: Optional[float] = None,
    new_min_ttc_s: Optional[float] = None,
    is_oracle_minimal: Optional[bool] = None,
    agents_tested: Optional[int] = None,
    strategy_used: Optional[str] = None,
    baseline_collision: Optional[bool] = None,
    baseline_min_distance_m: Optional[float] = None,
    baseline_min_ttc_s: Optional[float] = None,
):
    """
    Write counterfactual test results to intervention layer.

    Adds to customData["counterfactual"]:
    - test_passed: bool (collision avoided?)
    - new_min_distance_m: float
    - new_min_ttc_s: float
    - is_oracle_minimal: bool (if known)
    - agents_tested: int (how many agents were tested before this one)
    - strategy_used: str (ranking strategy that found this solution)
    - tested_at: str (ISO timestamp)

    Optional baseline comparison:
    - baseline_collision: bool
    - baseline_min_distance_m: float
    - baseline_min_ttc_s: float

    This makes intervention files self-documenting with their outcomes.

    Args:
        intervention_path: Path to the intervention USD file
        test_passed: Whether the intervention eliminated the collision
        new_min_distance_m: Minimum distance after intervention
        new_min_ttc_s: Minimum TTC after intervention
        is_oracle_minimal: Whether this is a minimal oracle solution
        agents_tested: Number of agents tested before finding this solution
        strategy_used: Ranking strategy used (e.g., "semantic", "ttc")
        baseline_collision: Whether baseline had collision
        baseline_min_distance_m: Baseline minimum distance
        baseline_min_ttc_s: Baseline minimum TTC
    """
    if not os.path.exists(intervention_path):
        print(f"Error: Intervention file not found: {intervention_path}")
        return

    # Open the intervention layer
    stage = Usd.Stage.Open(intervention_path)
    if not stage:
        print(f"Error: Failed to open stage: {intervention_path}")
        return

    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        print("Error: /World prim not found")
        return

    # Build counterfactual results dictionary
    counterfactual_results = {
        "test_passed": bool(test_passed),
        "tested_at": datetime.now().isoformat(),
    }

    # Add optional metrics
    if new_min_distance_m is not None:
        counterfactual_results["new_min_distance_m"] = float(new_min_distance_m)

    if new_min_ttc_s is not None:
        # Handle infinity - use sentinel value
        ttc_value = float(new_min_ttc_s) if new_min_ttc_s != float('inf') else 9999.0
        counterfactual_results["new_min_ttc_s"] = ttc_value

    if is_oracle_minimal is not None:
        counterfactual_results["is_oracle_minimal"] = bool(is_oracle_minimal)

    if agents_tested is not None:
        counterfactual_results["agents_tested"] = int(agents_tested)

    if strategy_used is not None:
        counterfactual_results["strategy_used"] = str(strategy_used)

    # Add baseline comparison metrics if provided
    if baseline_collision is not None:
        counterfactual_results["baseline_collision"] = bool(baseline_collision)

    if baseline_min_distance_m is not None:
        counterfactual_results["baseline_min_distance_m"] = float(baseline_min_distance_m)

    if baseline_min_ttc_s is not None:
        ttc_value = float(baseline_min_ttc_s) if baseline_min_ttc_s != float('inf') else 9999.0
        counterfactual_results["baseline_min_ttc_s"] = ttc_value

    # Calculate improvement metrics if we have both baseline and new values
    if baseline_min_distance_m is not None and new_min_distance_m is not None:
        counterfactual_results["distance_improvement_m"] = float(new_min_distance_m - baseline_min_distance_m)

    # Get existing customData and merge
    existing_custom_data = world_prim.GetCustomData()
    custom_data_dict = dict(existing_custom_data) if existing_custom_data else {}
    custom_data_dict["counterfactual"] = counterfactual_results

    # Set the updated customData
    world_prim.SetCustomData(custom_data_dict)

    # Save the stage
    stage.GetRootLayer().Save()

    status = "PASSED" if test_passed else "FAILED"
    print(f"Wrote counterfactual results to {intervention_path}: {status}")


def write_counterfactual_results_from_diagnosis(
    intervention_path: str,
    diagnosis_result,
    baseline_result=None,
    is_oracle_minimal: Optional[bool] = None,
):
    """
    Convenience function to write counterfactual results from a DiagnosisResult object.

    Args:
        intervention_path: Path to the intervention USD file
        diagnosis_result: DiagnosisResult from diagnosis_engine
        baseline_result: Optional DiagnosisResult for baseline comparison
        is_oracle_minimal: Whether this is a minimal oracle solution
    """
    # Extract values from diagnosis_result
    test_passed = not diagnosis_result.collision
    new_min_distance_m = diagnosis_result.margin_dist if diagnosis_result.margin_dist >= 0 else None
    new_min_ttc_s = diagnosis_result.margin_ttc if diagnosis_result.margin_ttc >= 0 else None
    agents_tested = diagnosis_result.agents_tested if hasattr(diagnosis_result, 'agents_tested') else None
    strategy_used = diagnosis_result.strategy_used if hasattr(diagnosis_result, 'strategy_used') else None

    # Extract baseline values if provided
    baseline_collision = None
    baseline_min_distance_m = None
    baseline_min_ttc_s = None

    if baseline_result is not None:
        baseline_collision = baseline_result.collision
        baseline_min_distance_m = baseline_result.margin_dist if baseline_result.margin_dist >= 0 else None
        baseline_min_ttc_s = baseline_result.margin_ttc if baseline_result.margin_ttc >= 0 else None

    write_counterfactual_results(
        intervention_path=intervention_path,
        test_passed=test_passed,
        new_min_distance_m=new_min_distance_m,
        new_min_ttc_s=new_min_ttc_s,
        is_oracle_minimal=is_oracle_minimal,
        agents_tested=agents_tested,
        strategy_used=strategy_used,
        baseline_collision=baseline_collision,
        baseline_min_distance_m=baseline_min_distance_m,
        baseline_min_ttc_s=baseline_min_ttc_s,
    )


def load_counterfactual_results(stage: Usd.Stage) -> Optional[Dict[str, Any]]:
    """
    Load counterfactual results from /World customData["counterfactual"].

    Returns:
        Dictionary with counterfactual test results, or None if not found.
    """
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        return None

    custom_data = world_prim.GetCustomData()
    if not custom_data:
        return None

    counterfactual_data = custom_data.get("counterfactual")
    if counterfactual_data:
        # Convert VtDictionary to regular dict
        return dict(counterfactual_data)
    return None


def batch_create_interventions(scenarios_dir, removal_sets, suffix="_base.usd"):
    """
    Create intervention layers for multiple scenarios and removal sets.

    Args:
        scenarios_dir: Directory containing base USD files
        removal_sets: List of removal sets to create (e.g., [{717}, {717, 823}])
        suffix: Suffix for base USD files

    Returns:
        List of created intervention file paths
    """
    import glob

    base_files = sorted(glob.glob(os.path.join(scenarios_dir, f"*{suffix}")))
    print(f"Found {len(base_files)} base USD files")

    created = []
    for base_path in base_files:
        for removal_set in removal_sets:
            try:
                intervention_path = create_intervention_layer(base_path, removal_set)
                created.append(intervention_path)
            except Exception as e:
                print(f"  Error creating intervention for {base_path}: {e}")

    return created


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create USD intervention layers for counterfactual analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single actor removal
  python intervention_layer.py --base scenarios/abc123_base.usd --remove 717

  # Multiple actors (Tier 2 pairwise)
  python intervention_layer.py --base scenarios/abc123_base.usd --remove 717 823

  # Verify an intervention layer
  python intervention_layer.py --verify scenarios/abc123_intervention_717.usd
        """
    )

    parser.add_argument("--base", help="Path to base USD file")
    parser.add_argument("--remove", nargs="+", help="Agent IDs to deactivate")
    parser.add_argument("--output", help="Output path (default: auto-generated)")
    parser.add_argument("--verify", help="Verify an existing intervention layer")
    parser.add_argument("--batch", action="store_true", help="Batch mode for all scenarios")
    parser.add_argument("--scenarios_dir", default="scenarios", help="Directory with base USDs")

    args = parser.parse_args()

    if args.verify:
        verify_intervention(args.verify)

    elif args.batch:
        if not args.remove:
            parser.error("--batch requires --remove to specify removal set")
        removal_set = set(args.remove)
        batch_create_interventions(args.scenarios_dir, [removal_set])

    elif args.base and args.remove:
        removal_set = set(args.remove)
        create_intervention_layer(args.base, removal_set, args.output)

    else:
        parser.print_help()
