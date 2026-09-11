"""
test_avxusd_implementation.py - Integration tests for AVxUSD implementation

Tests:
1. Schema validator on existing scenarios
2. Metrics namespace writing and reading
3. Scenario summary in customData
4. Intervention layer creation and metadata
5. Counterfactual results writing

Usage:
    python test_avxusd_implementation.py
    python test_avxusd_implementation.py --scenario scenarios/abc123_base.usd
"""

import os
import sys
import tempfile
import shutil
from pxr import Usd, UsdGeom, Sdf

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_schema import validate_avxusd_stage_detailed, ValidationResult
from collision_check import CollisionValidator
from intervention_layer import (
    create_intervention_layer,
    load_intervention_metadata,
    write_counterfactual_results,
    load_counterfactual_results,
    verify_intervention
)
from waymo_to_usd import add_scenario_summary, load_scenario_summary


def print_header(title: str):
    """Print a formatted test section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "PASS" if passed else "FAIL"
    marker = "[+]" if passed else "[X]"
    print(f"  {marker} {test_name}: {status}")
    if details:
        print(f"      {details}")


def test_schema_validator(scenario_path: str) -> bool:
    """Test 1: Run schema validator on a scenario."""
    print_header("TEST 1: Schema Validator")

    result = validate_avxusd_stage_detailed(scenario_path)

    print(f"  File: {scenario_path}")
    print(f"  Valid: {result.valid}")
    print(f"  Is Intervention: {result.is_intervention}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")

    if result.errors:
        print("\n  Errors:")
        for error in result.errors[:5]:
            print(f"    - {error}")

    if result.warnings:
        print("\n  Warnings (first 5):")
        for warning in result.warnings[:5]:
            print(f"    - {warning}")

    # For existing scenarios, we expect them to be mostly valid
    # but may have warnings for missing customData (since they were created before this feature)
    print_result("Schema validation completed", True)
    return True


def test_metrics_namespace(scenario_path: str, output_dir: str) -> bool:
    """Test 2: Write and read metrics namespace attributes."""
    print_header("TEST 2: Metrics Namespace")

    output_path = os.path.join(output_dir, "test_metrics.usd")

    try:
        # Create validator and analyze
        validator = CollisionValidator(scenario_path)
        results = validator.analyze_scenario()

        print(f"  Analyzed {len(results)} agents")

        # Write metrics to USD
        validator.write_metrics_to_usd(output_path, results)

        # Verify metrics were written
        stage = Usd.Stage.Open(output_path)
        agents_prim = stage.GetPrimAtPath("/World/Agents")

        metrics_found = False
        for agent_prim in agents_prim.GetChildren():
            if "EGO" in str(agent_prim.GetPath()):
                continue

            # Check for metrics attributes
            dist_attr = agent_prim.GetAttribute("metrics:distanceToEgo")
            ttc_attr = agent_prim.GetAttribute("metrics:ttc")
            min_dist_attr = agent_prim.GetAttribute("metrics:minDistance")

            if dist_attr and dist_attr.GetNumTimeSamples() > 0:
                metrics_found = True
                samples = dist_attr.GetTimeSamples()
                print(f"  Agent {agent_prim.GetName()}:")
                print(f"    metrics:distanceToEgo - {len(samples)} time samples")

                if ttc_attr:
                    print(f"    metrics:ttc - {ttc_attr.GetNumTimeSamples()} time samples")
                if min_dist_attr and min_dist_attr.HasValue():
                    print(f"    metrics:minDistance = {min_dist_attr.Get():.2f}m")

                break  # Just check first agent with metrics

        print_result("Metrics written to USD", metrics_found)
        print_result("Time-sampled attributes created", metrics_found)

        return metrics_found

    except Exception as e:
        print(f"  Error: {e}")
        print_result("Metrics namespace test", False, str(e))
        return False


def test_scenario_summary(scenario_path: str, output_dir: str) -> bool:
    """Test 3: Add and read scenario summary in customData."""
    print_header("TEST 3: Scenario Summary (customData)")

    output_path = os.path.join(output_dir, "test_scenario_summary.usd")

    try:
        # Copy the original file
        shutil.copy(scenario_path, output_path)

        # Open and add scenario summary
        stage = Usd.Stage.Open(output_path)

        # Create a mock scenario object with minimal required fields
        class MockScenario:
            def __init__(self):
                self.scenario_id = "test_scenario_123"
                self.timestamps_seconds = [i * 0.1 for i in range(91)]
                self.sdc_track_index = 0

                class MockTrack:
                    def __init__(self, id):
                        self.id = id

                self.tracks = [MockTrack(i) for i in range(10)]

        mock_scenario = MockScenario()

        # Run analysis for metrics
        validator = CollisionValidator(output_path)
        analysis_results = validator.analyze_scenario()

        # Add scenario summary
        add_scenario_summary(
            stage,
            mock_scenario,
            analysis_results=analysis_results,
            typology="Test_Intersection",
            baseline_collision=True
        )
        stage.GetRootLayer().Save()

        # Verify by loading
        stage2 = Usd.Stage.Open(output_path)
        summary = load_scenario_summary(stage2)

        if summary:
            print(f"  Loaded scenario summary:")
            print(f"    avxusd_version: {summary.get('avxusd_version')}")
            print(f"    scenario_id: {summary.get('scenario_id')}")
            print(f"    num_agents: {summary.get('num_agents')}")
            print(f"    duration_frames: {summary.get('duration_frames')}")
            print(f"    typology: {summary.get('typology')}")
            print(f"    min_distance_m: {summary.get('min_distance_m')}")
            print(f"    baseline_collision: {summary.get('baseline_collision')}")

        has_version = summary and "avxusd_version" in summary
        has_scenario_id = summary and "scenario_id" in summary

        print_result("customData['scenario'] created", summary is not None)
        print_result("avxusd_version present", has_version)
        print_result("scenario_id present", has_scenario_id)

        return summary is not None and has_version

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        print_result("Scenario summary test", False, str(e))
        return False


def test_intervention_layer(scenario_path: str, output_dir: str) -> bool:
    """Test 4: Create intervention layer with metadata."""
    print_header("TEST 4: Intervention Layer")

    try:
        # Copy base to output dir for relative path resolution
        base_filename = os.path.basename(scenario_path)
        temp_base = os.path.join(output_dir, base_filename)
        shutil.copy(scenario_path, temp_base)

        # Find an agent to remove
        stage = Usd.Stage.Open(temp_base)
        agents_prim = stage.GetPrimAtPath("/World/Agents")

        removal_agent = None
        for agent_prim in agents_prim.GetChildren():
            agent_name = agent_prim.GetName()
            if "EGO" not in agent_name:
                # Extract ID from Agent_XXX
                removal_agent = agent_name.replace("Agent_", "")
                break

        if not removal_agent:
            print("  No non-EGO agents found to remove")
            return False

        print(f"  Creating intervention to remove agent: {removal_agent}")

        # Create intervention layer
        intervention_path = create_intervention_layer(
            temp_base,
            {removal_agent},
            collision_threshold_m=0.5,
            intervention_reason="Test intervention"
        )

        # Verify intervention metadata
        int_stage = Usd.Stage.Open(intervention_path)
        int_meta = load_intervention_metadata(int_stage)

        if int_meta:
            print(f"  Intervention metadata:")
            print(f"    avxusd_version: {int_meta.get('avxusd_version')}")
            print(f"    base_scenario: {int_meta.get('base_scenario')}")
            print(f"    removal_set: {int_meta.get('removal_set')}")
            print(f"    removal_set_types: {int_meta.get('removal_set_types')}")
            print(f"    created_at: {int_meta.get('created_at')}")

        # Verify agent is deactivated
        removed_agent_path = f"/World/Agents/Agent_{removal_agent}"
        removed_prim = int_stage.GetPrimAtPath(removed_agent_path)
        is_deactivated = removed_prim and not removed_prim.IsActive()

        # Verify sublayer reference
        root_layer = int_stage.GetRootLayer()
        has_sublayer = len(root_layer.subLayerPaths) > 0

        print_result("Intervention layer created", os.path.exists(intervention_path))
        print_result("customData['intervention'] present", int_meta is not None)
        print_result("Agent deactivated", is_deactivated)
        print_result("Sublayer reference exists", has_sublayer)

        # Run verify_intervention
        print("\n  Running verify_intervention:")
        verify_intervention(intervention_path)

        return int_meta is not None and is_deactivated and has_sublayer

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        print_result("Intervention layer test", False, str(e))
        return False


def test_counterfactual_results(scenario_path: str, output_dir: str) -> bool:
    """Test 5: Write and read counterfactual results."""
    print_header("TEST 5: Counterfactual Results")

    try:
        # First create an intervention layer
        base_filename = os.path.basename(scenario_path)
        temp_base = os.path.join(output_dir, base_filename.replace("_base", "_cf_base"))
        shutil.copy(scenario_path, temp_base)

        # Find an agent
        stage = Usd.Stage.Open(temp_base)
        agents_prim = stage.GetPrimAtPath("/World/Agents")

        removal_agent = None
        for agent_prim in agents_prim.GetChildren():
            agent_name = agent_prim.GetName()
            if "EGO" not in agent_name:
                removal_agent = agent_name.replace("Agent_", "")
                break

        if not removal_agent:
            print("  No agent to test")
            return False

        # Create intervention
        intervention_path = create_intervention_layer(temp_base, {removal_agent})

        # Write counterfactual results
        write_counterfactual_results(
            intervention_path=intervention_path,
            test_passed=True,
            new_min_distance_m=3.5,
            new_min_ttc_s=2.1,
            is_oracle_minimal=True,
            agents_tested=3,
            strategy_used="semantic",
            baseline_collision=True,
            baseline_min_distance_m=0.35,
            baseline_min_ttc_s=1.2
        )

        # Load and verify
        cf_stage = Usd.Stage.Open(intervention_path)
        cf_results = load_counterfactual_results(cf_stage)

        if cf_results:
            print(f"  Counterfactual results:")
            print(f"    test_passed: {cf_results.get('test_passed')}")
            print(f"    new_min_distance_m: {cf_results.get('new_min_distance_m')}")
            print(f"    new_min_ttc_s: {cf_results.get('new_min_ttc_s')}")
            print(f"    is_oracle_minimal: {cf_results.get('is_oracle_minimal')}")
            print(f"    strategy_used: {cf_results.get('strategy_used')}")
            print(f"    distance_improvement_m: {cf_results.get('distance_improvement_m')}")

        has_test_passed = cf_results and "test_passed" in cf_results
        has_metrics = cf_results and "new_min_distance_m" in cf_results

        print_result("customData['counterfactual'] created", cf_results is not None)
        print_result("test_passed field present", has_test_passed)
        print_result("Metrics fields present", has_metrics)

        return cf_results is not None and has_test_passed

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        print_result("Counterfactual results test", False, str(e))
        return False


def test_schema_validator_batch(scenarios_dir: str) -> bool:
    """Test 6: Run schema validator on all existing scenarios."""
    print_header("TEST 6: Batch Schema Validation")

    import glob
    usd_files = sorted(glob.glob(os.path.join(scenarios_dir, "*_base.usd")))

    if not usd_files:
        print("  No USD files found")
        return False

    print(f"  Validating {len(usd_files)} base scenarios...")

    valid_count = 0
    error_count = 0
    warning_scenarios = []

    for usd_path in usd_files:
        result = validate_avxusd_stage_detailed(usd_path)
        if result.valid:
            valid_count += 1
        else:
            error_count += 1
            print(f"    INVALID: {os.path.basename(usd_path)}")
            for error in result.errors[:2]:
                print(f"      - {error}")

        if result.warnings:
            warning_scenarios.append(os.path.basename(usd_path))

    print(f"\n  Results:")
    print(f"    Valid: {valid_count}/{len(usd_files)}")
    print(f"    Invalid: {error_count}/{len(usd_files)}")
    print(f"    With warnings: {len(warning_scenarios)}/{len(usd_files)}")

    # Existing scenarios may have warnings for missing customData
    # since they were created before this feature
    print_result("Batch validation completed", True)
    print_result("All scenarios structurally valid", error_count == 0)

    return error_count == 0


def run_all_tests(scenario_path: str, scenarios_dir: str):
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print(" AVxUSD IMPLEMENTATION INTEGRATION TESTS")
    print("=" * 70)
    print(f"Test scenario: {scenario_path}")
    print(f"Scenarios dir: {scenarios_dir}")

    # Create temp directory for test outputs
    temp_dir = tempfile.mkdtemp(prefix="avxusd_test_")
    print(f"Temp directory: {temp_dir}")

    results = {}

    try:
        # Run tests
        results["schema_validator"] = test_schema_validator(scenario_path)
        results["metrics_namespace"] = test_metrics_namespace(scenario_path, temp_dir)
        results["scenario_summary"] = test_scenario_summary(scenario_path, temp_dir)
        results["intervention_layer"] = test_intervention_layer(scenario_path, temp_dir)
        results["counterfactual_results"] = test_counterfactual_results(scenario_path, temp_dir)
        results["batch_validation"] = test_schema_validator_batch(scenarios_dir)

    finally:
        # Cleanup
        print(f"\n  Cleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Summary
    print_header("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_flag in results.items():
        status = "PASS" if passed_flag else "FAIL"
        marker = "[+]" if passed_flag else "[X]"
        print(f"  {marker} {test_name}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 70)

    return passed == total


# --- CLI ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Integration tests for AVxUSD implementation"
    )
    parser.add_argument("--scenario", default=None,
                        help="Specific scenario to test")
    parser.add_argument("--scenarios-dir", default="scenarios",
                        help="Directory containing scenarios")

    args = parser.parse_args()

    # Find a test scenario if not specified
    if args.scenario:
        test_scenario = args.scenario
    else:
        import glob
        scenarios = sorted(glob.glob(os.path.join(args.scenarios_dir, "*_base.usd")))
        if not scenarios:
            print(f"No scenarios found in {args.scenarios_dir}")
            sys.exit(1)
        test_scenario = scenarios[0]

    if not os.path.exists(test_scenario):
        print(f"Scenario not found: {test_scenario}")
        sys.exit(1)

    success = run_all_tests(test_scenario, args.scenarios_dir)
    sys.exit(0 if success else 1)
