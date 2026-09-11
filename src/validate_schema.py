"""
validate_schema.py - AVxUSD Schema Validator

Validates USD files conform to AVxUSD v0.1 schema.
Inspired by RLxUSD Section 4.4 validation rules.

Usage:
    python validate_schema.py scenarios/abc123_base.usd
    python validate_schema.py --all --scenarios-dir scenarios
    python validate_schema.py --intervention scenarios/abc123_intervention_717.usd
"""

from pxr import Usd, UsdGeom, Sdf
import os
import argparse
import glob
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    """Validation issue severity levels."""
    ERROR = "ERROR"      # Schema violation - file is invalid
    WARNING = "WARNING"  # Recommended but not required
    INFO = "INFO"        # Informational note


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: Severity
    rule_id: str
    message: str
    path: Optional[str] = None

    def __str__(self):
        path_str = f" at {self.path}" if self.path else ""
        return f"[{self.severity.value}] {self.rule_id}: {self.message}{path_str}"


@dataclass
class ValidationResult:
    """Complete validation result for a USD file."""
    file_path: str
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    is_intervention: bool = False
    schema_version: Optional[str] = None

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def summary(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        return (f"{status}: {self.file_path} "
                f"({len(self.errors)} errors, {len(self.warnings)} warnings)")


# Valid object types per schema
VALID_OBJECT_TYPES = {0, 1, 2, 3, 4}
VALID_OBJECT_TYPE_STRINGS = {
    "TYPE_UNSET", "TYPE_VEHICLE", "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST", "TYPE_OTHER", "TYPE_UNKNOWN"
}


def validate_avxusd_stage(usd_path: str) -> Tuple[bool, List[str]]:
    """
    Validate a USD stage against AVxUSD v0.1 schema.

    Returns:
        (valid: bool, issues: List[str])

    Checks:
    - Required prims exist (/World, /World/Agents, Agent_EGO)
    - Required attributes present on agents
    - Time sampling consistency
    - customData["scenario"] present and valid
    - For intervention layers: sublayer reference exists
    """
    result = validate_avxusd_stage_detailed(usd_path)
    issues = [str(i) for i in result.issues]
    return result.valid, issues


def validate_avxusd_stage_detailed(usd_path: str) -> ValidationResult:
    """
    Detailed validation returning structured ValidationResult.

    This is the main validation entry point with full diagnostics.
    """
    result = ValidationResult(file_path=usd_path, valid=True)

    # Check file exists
    if not os.path.exists(usd_path):
        result.issues.append(ValidationIssue(
            Severity.ERROR, "FILE-001",
            f"File does not exist: {usd_path}"
        ))
        result.valid = False
        return result

    # Try to open stage
    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as e:
        result.issues.append(ValidationIssue(
            Severity.ERROR, "FILE-002",
            f"Failed to open USD stage: {e}"
        ))
        result.valid = False
        return result

    if not stage:
        result.issues.append(ValidationIssue(
            Severity.ERROR, "FILE-002",
            "Failed to open USD stage (returned None)"
        ))
        result.valid = False
        return result

    # Detect if this is an intervention layer
    root_layer = stage.GetRootLayer()
    result.is_intervention = len(root_layer.subLayerPaths) > 0

    # Run all validation checks
    _validate_stage_metadata(stage, result)
    _validate_world_prim(stage, result)
    _validate_agents_container(stage, result)
    _validate_ego_agent(stage, result)
    _validate_agent_prims(stage, result)
    _validate_time_sampling(stage, result)
    _validate_custom_data(stage, result)

    if result.is_intervention:
        _validate_intervention_layer(stage, root_layer, result)

    # Determine overall validity (any ERROR = invalid)
    result.valid = len(result.errors) == 0

    return result


def _validate_stage_metadata(stage: Usd.Stage, result: ValidationResult):
    """Validate stage-level metadata (Rules 1-5)."""

    # Rule 1: Up axis must be Z
    up_axis = UsdGeom.GetStageUpAxis(stage)
    if up_axis != UsdGeom.Tokens.z:
        result.issues.append(ValidationIssue(
            Severity.ERROR, "STAGE-001",
            f"Stage upAxis must be 'Z', found '{up_axis}'"
        ))

    # Rule 2: Meters per unit must be 1.0
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if abs(meters_per_unit - 1.0) > 0.001:
        result.issues.append(ValidationIssue(
            Severity.ERROR, "STAGE-002",
            f"Stage metersPerUnit must be 1.0, found {meters_per_unit}"
        ))

    # Rule 3: Start time code must be defined
    if not stage.HasAuthoredTimeCodeRange():
        result.issues.append(ValidationIssue(
            Severity.ERROR, "STAGE-003",
            "Stage must define startTimeCode and endTimeCode"
        ))
    else:
        start = stage.GetStartTimeCode()
        end = stage.GetEndTimeCode()
        if start > end:
            result.issues.append(ValidationIssue(
                Severity.ERROR, "STAGE-004",
                f"startTimeCode ({start}) > endTimeCode ({end})"
            ))

    # Rule 5: Frames per second should be 10
    fps = stage.GetFramesPerSecond()
    if fps and fps != 10:
        result.issues.append(ValidationIssue(
            Severity.WARNING, "STAGE-005",
            f"framesPerSecond should be 10 for Waymo data, found {fps}"
        ))


def _validate_world_prim(stage: Usd.Stage, result: ValidationResult):
    """Validate /World prim exists (Rule 6)."""
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        result.issues.append(ValidationIssue(
            Severity.ERROR, "PRIM-001",
            "/World prim must exist",
            path="/World"
        ))
        return

    # Check it's an Xform
    if not world_prim.IsA(UsdGeom.Xform):
        result.issues.append(ValidationIssue(
            Severity.WARNING, "PRIM-002",
            "/World should be a UsdGeom.Xform",
            path="/World"
        ))


def _validate_agents_container(stage: Usd.Stage, result: ValidationResult):
    """Validate /World/Agents container exists (Rule 7)."""
    agents_prim = stage.GetPrimAtPath("/World/Agents")
    if not agents_prim:
        result.issues.append(ValidationIssue(
            Severity.ERROR, "PRIM-003",
            "/World/Agents prim must exist",
            path="/World/Agents"
        ))


def _validate_ego_agent(stage: Usd.Stage, result: ValidationResult):
    """Validate Agent_EGO exists (Rule 8)."""
    ego_prim = stage.GetPrimAtPath("/World/Agents/Agent_EGO")
    if not ego_prim:
        result.issues.append(ValidationIssue(
            Severity.ERROR, "PRIM-004",
            "/World/Agents/Agent_EGO must exist for collision analysis",
            path="/World/Agents/Agent_EGO"
        ))
        return

    # Check EGO has geometry child (Rule 9)
    ego_geom = stage.GetPrimAtPath("/World/Agents/Agent_EGO/Geometry")
    if not ego_geom:
        result.issues.append(ValidationIssue(
            Severity.ERROR, "PRIM-005",
            "Agent_EGO must have /Geometry child",
            path="/World/Agents/Agent_EGO/Geometry"
        ))


def _validate_agent_prims(stage: Usd.Stage, result: ValidationResult):
    """Validate all agent prims have required attributes (Rules 9-13)."""
    agents_prim = stage.GetPrimAtPath("/World/Agents")
    if not agents_prim:
        return

    for agent_prim in agents_prim.GetChildren():
        if not agent_prim.IsActive():
            # Skip deactivated agents (intervention layers)
            continue

        agent_path = str(agent_prim.GetPath())

        # Rule 9: Must have Geometry child
        geom_path = f"{agent_path}/Geometry"
        geom_prim = stage.GetPrimAtPath(geom_path)
        if not geom_prim:
            result.issues.append(ValidationIssue(
                Severity.ERROR, "AGENT-001",
                "Agent must have /Geometry child",
                path=agent_path
            ))

        # Rule 10: Must have waymo:objectType
        obj_type_attr = agent_prim.GetAttribute("waymo:objectType")
        if not obj_type_attr or not obj_type_attr.HasValue():
            result.issues.append(ValidationIssue(
                Severity.ERROR, "AGENT-002",
                "Agent must have waymo:objectType attribute",
                path=agent_path
            ))
        else:
            obj_type = obj_type_attr.Get()
            if obj_type not in VALID_OBJECT_TYPES:
                result.issues.append(ValidationIssue(
                    Severity.WARNING, "AGENT-003",
                    f"waymo:objectType value {obj_type} not in standard range (0-4)",
                    path=agent_path
                ))

        # Rule 11: Must have waymo:objectTypeString
        obj_type_str_attr = agent_prim.GetAttribute("waymo:objectTypeString")
        if not obj_type_str_attr or not obj_type_str_attr.HasValue():
            result.issues.append(ValidationIssue(
                Severity.ERROR, "AGENT-004",
                "Agent must have waymo:objectTypeString attribute",
                path=agent_path
            ))
        else:
            obj_type_str = obj_type_str_attr.Get()
            if obj_type_str not in VALID_OBJECT_TYPE_STRINGS:
                result.issues.append(ValidationIssue(
                    Severity.WARNING, "AGENT-005",
                    f"waymo:objectTypeString '{obj_type_str}' not recognized",
                    path=agent_path
                ))

        # Rule 12: Must have time-sampled velocity attributes
        vel_x_attr = agent_prim.GetAttribute("waymo:velocityX")
        vel_y_attr = agent_prim.GetAttribute("waymo:velocityY")

        if not vel_x_attr:
            result.issues.append(ValidationIssue(
                Severity.ERROR, "AGENT-006",
                "Agent must have waymo:velocityX attribute",
                path=agent_path
            ))
        elif not vel_x_attr.GetNumTimeSamples() > 0:
            result.issues.append(ValidationIssue(
                Severity.WARNING, "AGENT-007",
                "waymo:velocityX should be time-sampled",
                path=agent_path
            ))

        if not vel_y_attr:
            result.issues.append(ValidationIssue(
                Severity.ERROR, "AGENT-008",
                "Agent must have waymo:velocityY attribute",
                path=agent_path
            ))
        elif not vel_y_attr.GetNumTimeSamples() > 0:
            result.issues.append(ValidationIssue(
                Severity.WARNING, "AGENT-009",
                "waymo:velocityY should be time-sampled",
                path=agent_path
            ))

        # Rule 13: Should have validity frame attributes
        first_valid_attr = agent_prim.GetAttribute("waymo:firstValidFrame")
        last_valid_attr = agent_prim.GetAttribute("waymo:lastValidFrame")

        if not first_valid_attr or not first_valid_attr.HasValue():
            result.issues.append(ValidationIssue(
                Severity.WARNING, "AGENT-010",
                "Agent should have waymo:firstValidFrame attribute",
                path=agent_path
            ))

        if not last_valid_attr or not last_valid_attr.HasValue():
            result.issues.append(ValidationIssue(
                Severity.WARNING, "AGENT-011",
                "Agent should have waymo:lastValidFrame attribute",
                path=agent_path
            ))


def _validate_time_sampling(stage: Usd.Stage, result: ValidationResult):
    """Validate time sampling consistency (Rules 14-16)."""
    if not stage.HasAuthoredTimeCodeRange():
        return

    start_time = int(stage.GetStartTimeCode())
    end_time = int(stage.GetEndTimeCode())
    expected_samples = set(range(start_time, end_time + 1))

    agents_prim = stage.GetPrimAtPath("/World/Agents")
    if not agents_prim:
        return

    # Check a sample of agents for time consistency
    checked_agents = 0
    for agent_prim in agents_prim.GetChildren():
        if not agent_prim.IsActive():
            continue

        checked_agents += 1
        if checked_agents > 5:  # Only check first 5 for performance
            break

        agent_path = str(agent_prim.GetPath())

        # Check transform has time samples
        xform = UsdGeom.Xformable(agent_prim)
        xform_ops = xform.GetOrderedXformOps()

        has_animated_transform = False
        for op in xform_ops:
            attr = op.GetAttr()
            if attr.GetNumTimeSamples() > 0:
                has_animated_transform = True
                break

        if not has_animated_transform:
            result.issues.append(ValidationIssue(
                Severity.WARNING, "TIME-001",
                "Agent transform should be time-sampled",
                path=agent_path
            ))


def _validate_custom_data(stage: Usd.Stage, result: ValidationResult):
    """Validate customData structures (Rules 21-24)."""
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        return

    custom_data = world_prim.GetCustomData()

    # Check for scenario metadata (Rule 21-22)
    if "scenario" in custom_data:
        scenario_data = dict(custom_data["scenario"])
        result.schema_version = scenario_data.get("avxusd_version")

        # Rule 21: Must include avxusd_version
        if "avxusd_version" not in scenario_data:
            result.issues.append(ValidationIssue(
                Severity.ERROR, "DATA-001",
                "customData['scenario'] must include 'avxusd_version'",
                path="/World"
            ))

        # Rule 22: Must include scenario_id
        if "scenario_id" not in scenario_data:
            result.issues.append(ValidationIssue(
                Severity.ERROR, "DATA-002",
                "customData['scenario'] must include 'scenario_id'",
                path="/World"
            ))

        # Check for recommended fields
        recommended_fields = [
            "sdc_track_id", "num_agents", "duration_frames",
            "duration_seconds", "frames_per_second"
        ]
        for field in recommended_fields:
            if field not in scenario_data:
                result.issues.append(ValidationIssue(
                    Severity.WARNING, "DATA-003",
                    f"customData['scenario'] should include '{field}'",
                    path="/World"
                ))

    elif not result.is_intervention:
        # Base scenarios should have scenario metadata
        result.issues.append(ValidationIssue(
            Severity.WARNING, "DATA-004",
            "Base scenario should have customData['scenario']",
            path="/World"
        ))

    # Check intervention metadata (Rules 23-24)
    if "intervention" in custom_data:
        intervention_data = dict(custom_data["intervention"])

        # Rule 23: Must include base_scenario
        if "base_scenario" not in intervention_data:
            result.issues.append(ValidationIssue(
                Severity.ERROR, "DATA-005",
                "customData['intervention'] must include 'base_scenario'",
                path="/World"
            ))

        # Rule 24: Must include removal_set
        if "removal_set" not in intervention_data:
            result.issues.append(ValidationIssue(
                Severity.ERROR, "DATA-006",
                "customData['intervention'] must include 'removal_set'",
                path="/World"
            ))


def _validate_intervention_layer(stage: Usd.Stage, root_layer: Sdf.Layer,
                                  result: ValidationResult):
    """Validate intervention layer specific rules (Rules 17-20)."""

    # Rule 17: Must sublayer base USD
    if len(root_layer.subLayerPaths) == 0:
        result.issues.append(ValidationIssue(
            Severity.ERROR, "INTV-001",
            "Intervention layer must sublayer a base USD"
        ))
        return

    # Rule 18: Should use relative paths
    for sublayer_path in root_layer.subLayerPaths:
        if os.path.isabs(sublayer_path):
            result.issues.append(ValidationIssue(
                Severity.WARNING, "INTV-002",
                f"Sublayer should use relative path: {sublayer_path}"
            ))

    # Check for deactivated agents
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim:
        # Rule 19: Check deactivated agents have active=false
        removal_attr = world_prim.GetAttribute("waymo:removalSetStr")
        if removal_attr and removal_attr.HasValue():
            removal_set = list(removal_attr.Get())

            for agent_id in removal_set:
                safe_id = str(agent_id).replace("-", "_")
                agent_path = f"/World/Agents/Agent_{safe_id}"
                agent_prim = stage.GetPrimAtPath(agent_path)

                if agent_prim and agent_prim.IsActive():
                    result.issues.append(ValidationIssue(
                        Severity.ERROR, "INTV-003",
                        f"Agent in removal set should be deactivated",
                        path=agent_path
                    ))

        # Rule 20: Should have intervention customData
        custom_data = world_prim.GetCustomData()
        if "intervention" not in custom_data:
            result.issues.append(ValidationIssue(
                Severity.WARNING, "INTV-004",
                "Intervention layer should have customData['intervention']",
                path="/World"
            ))


def validate_directory(
    scenarios_dir: str,
    pattern: str = "*.usd",
    verbose: bool = False
) -> Dict[str, ValidationResult]:
    """
    Validate all USD files in a directory.

    Returns:
        Dictionary mapping file paths to ValidationResults
    """
    results = {}

    usd_files = sorted(glob.glob(os.path.join(scenarios_dir, pattern)))
    print(f"Found {len(usd_files)} USD files to validate")

    for usd_path in usd_files:
        if verbose:
            print(f"\nValidating: {usd_path}")

        result = validate_avxusd_stage_detailed(usd_path)
        results[usd_path] = result

        if verbose:
            print(f"  {result.summary()}")
            for issue in result.issues:
                print(f"    {issue}")

    return results


def print_validation_report(results: Dict[str, ValidationResult]):
    """Print a summary report of validation results."""
    total = len(results)
    valid = sum(1 for r in results.values() if r.valid)
    invalid = total - valid

    total_errors = sum(len(r.errors) for r in results.values())
    total_warnings = sum(len(r.warnings) for r in results.values())

    print("\n" + "=" * 70)
    print("AVXUSD SCHEMA VALIDATION REPORT")
    print("=" * 70)
    print(f"Files validated:  {total}")
    print(f"Valid:            {valid}")
    print(f"Invalid:          {invalid}")
    print(f"Total errors:     {total_errors}")
    print(f"Total warnings:   {total_warnings}")
    print("-" * 70)

    if invalid > 0:
        print("\nINVALID FILES:")
        for path, result in results.items():
            if not result.valid:
                print(f"\n  {path}")
                for error in result.errors:
                    print(f"    {error}")

    print("=" * 70)


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate USD files against AVxUSD v0.1 schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a single file
  python validate_schema.py scenarios/abc123_base.usd

  # Validate all files in a directory
  python validate_schema.py --all --scenarios-dir scenarios

  # Validate intervention layers only
  python validate_schema.py --all --scenarios-dir scenarios --pattern "*_intervention_*.usd"

  # Verbose output
  python validate_schema.py scenarios/abc123_base.usd --verbose

Validation Rules:
  See AVXUSD_SCHEMA.md for complete schema documentation.

Exit Codes:
  0 - All files valid
  1 - One or more files invalid
  2 - Error (file not found, etc.)
        """
    )

    parser.add_argument("file", nargs="?", help="USD file to validate")
    parser.add_argument("--all", action="store_true",
                        help="Validate all USD files in directory")
    parser.add_argument("--scenarios-dir", default="scenarios",
                        help="Directory containing USD files")
    parser.add_argument("--pattern", default="*.usd",
                        help="Glob pattern for USD files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")

    args = parser.parse_args()

    if args.all:
        # Validate all files in directory
        results = validate_directory(args.scenarios_dir, args.pattern, args.verbose)
        print_validation_report(results)

        # Determine exit code
        if args.strict:
            all_clean = all(len(r.issues) == 0 for r in results.values())
        else:
            all_clean = all(r.valid for r in results.values())

        exit(0 if all_clean else 1)

    elif args.file:
        # Validate single file
        result = validate_avxusd_stage_detailed(args.file)

        print(f"\n{result.summary()}")
        print("-" * 60)

        if result.issues:
            for issue in result.issues:
                print(f"  {issue}")
        else:
            print("  No issues found.")

        print("-" * 60)

        if args.json:
            import json
            output = {
                "file": result.file_path,
                "valid": result.valid,
                "is_intervention": result.is_intervention,
                "schema_version": result.schema_version,
                "errors": [str(e) for e in result.errors],
                "warnings": [str(w) for w in result.warnings],
            }
            print(json.dumps(output, indent=2))

        # Exit code based on validity
        if args.strict:
            exit(0 if len(result.issues) == 0 else 1)
        else:
            exit(0 if result.valid else 1)

    else:
        parser.print_help()
        exit(2)
