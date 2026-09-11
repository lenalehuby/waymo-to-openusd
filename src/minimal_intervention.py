"""
minimal_intervention.py - Minimal-Intervention Counterfactual Explanations

Implements physically-plausible ego vehicle interventions to find the minimum
modification needed to avoid collision or achieve safe distance margins.

Intervention Types:
1. Uniform Speed Reduction: Scale all ego velocities by factor (e.g., 0.95 = 5% slower)
2. Targeted Braking: Apply deceleration starting at critical time
3. Emergency Braking: Apply maximum physically-plausible deceleration

Physical Constraints (based on vehicle dynamics research):
- Comfortable braking: 2-3 m/s² (passenger comfort threshold)
- Firm braking: 4-5 m/s² (noticeable but controlled)
- Hard braking: 6-7 m/s² (aggressive, may activate ABS)
- Emergency braking: 8-10 m/s² (maximum friction-limited, dry pavement)

Usage:
    python minimal_intervention.py --scenario scenarios/abc123_base.usd --mode uniform
    python minimal_intervention.py --scenario scenarios/abc123_base.usd --mode braking --threshold 3.0
"""

import os
import math
import tempfile
import shutil
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pxr import Usd, Sdf, UsdGeom, Gf
import numpy as np

from collision_check import CollisionValidator


class InterventionMode(Enum):
    """Types of ego vehicle interventions."""
    UNIFORM_SPEED_REDUCTION = "uniform"       # Scale all velocities uniformly
    TARGETED_BRAKING = "braking"              # Apply braking at specific time
    EMERGENCY_BRAKING = "emergency"           # Maximum deceleration
    SPEED_LIMIT = "limit"                     # Cap maximum speed


class BrakingProfile(Enum):
    """Predefined braking intensity profiles."""
    COMFORTABLE = "comfortable"    # 3.0 m/s² - passenger comfort
    FIRM = "firm"                  # 5.0 m/s² - controlled braking
    HARD = "hard"                  # 7.0 m/s² - aggressive braking
    EMERGENCY = "emergency"        # 9.0 m/s² - maximum friction-limited


# Physical constants for braking deceleration (m/s²)
BRAKING_DECELERATION = {
    BrakingProfile.COMFORTABLE: 3.0,
    BrakingProfile.FIRM: 5.0,
    BrakingProfile.HARD: 7.0,
    BrakingProfile.EMERGENCY: 9.0,
}

# Maximum physically achievable deceleration (dry pavement, good tires)
MAX_DECELERATION = 9.81  # Approximately 1g


@dataclass
class InterventionResult:
    """Result of a single intervention test."""
    intervention_type: str
    intervention_value: float  # Speed factor, deceleration, etc.
    intervention_description: str

    # Pre-intervention metrics
    original_min_distance: float
    original_min_ttc: float
    original_collision: bool

    # Post-intervention metrics
    new_min_distance: float
    new_min_ttc: float
    new_collision: bool

    # Success metrics
    collision_avoided: bool
    threshold_achieved: bool
    safety_margin_gained: float  # new_min_distance - original_min_distance


@dataclass
class MinimalInterventionExplanation:
    """Complete minimal intervention explanation for a scenario."""
    scenario_id: str
    scenario_path: str
    safety_threshold_m: float

    # Original scenario state
    original_min_distance: float
    original_min_ttc: float
    original_collision: bool
    critical_time_s: float
    critical_frame: int

    # Minimal intervention found
    minimal_intervention: Optional[InterventionResult]
    intervention_mode: str

    # All interventions tested (for analysis)
    all_interventions: List[InterventionResult] = field(default_factory=list)

    # Natural language explanation
    explanation: str = ""

    # Physical feasibility assessment
    physically_feasible: bool = True
    feasibility_notes: str = ""


class EgoInterventionEngine:
    """
    Engine for applying and testing ego vehicle interventions.

    Creates modified USD stages with altered ego trajectories and
    validates whether the intervention achieves safety goals.
    """

    def __init__(self, base_usd_path: str, safety_threshold: float = 2.5):
        """
        Initialize the intervention engine.

        Args:
            base_usd_path: Path to base USD scenario
            safety_threshold: Minimum safe distance in meters
        """
        self.base_usd_path = base_usd_path
        self.safety_threshold = safety_threshold
        self.scenario_id = os.path.basename(base_usd_path).replace("_base.usd", "")

        # Load base stage for reading
        self.base_stage = Usd.Stage.Open(base_usd_path)
        if not self.base_stage:
            raise FileNotFoundError(f"Failed to open: {base_usd_path}")

        # Get ego prim
        self.ego_xform = self.base_stage.GetPrimAtPath("/World/Agents/Agent_EGO")
        if not self.ego_xform:
            raise ValueError("EGO agent not found in scenario")

        # Get timestamps
        world_prim = self.base_stage.GetPrimAtPath("/World")
        ts_attr = world_prim.GetAttribute("waymo:timestampsSeconds")
        self.timestamps = list(ts_attr.Get()) if ts_attr and ts_attr.Get() else None

        # Cache original ego trajectory
        self._cache_ego_trajectory()

        # Compute baseline metrics
        self._compute_baseline()

    def _cache_ego_trajectory(self):
        """Cache the original ego trajectory for modification."""
        self.ego_positions = {}  # frame -> (x, y, z)
        self.ego_velocities = {}  # frame -> (vx, vy)
        self.ego_headings = {}   # frame -> degrees

        vx_attr = self.ego_xform.GetAttribute("waymo:velocityX")
        vy_attr = self.ego_xform.GetAttribute("waymo:velocityY")

        xformable = UsdGeom.Xformable(self.ego_xform)

        start_frame = int(self.base_stage.GetStartTimeCode())
        end_frame = int(self.base_stage.GetEndTimeCode())

        for frame in range(start_frame, end_frame + 1):
            time_code = Usd.TimeCode(frame)

            # Get position from transform
            world_xform = xformable.ComputeLocalToWorldTransform(time_code)
            pos = world_xform.ExtractTranslation()
            self.ego_positions[frame] = (pos[0], pos[1], pos[2])

            # Get velocity
            vx = vx_attr.Get(time_code) if vx_attr else 0.0
            vy = vy_attr.Get(time_code) if vy_attr else 0.0
            if vx is not None and vy is not None:
                self.ego_velocities[frame] = (float(vx), float(vy))
            else:
                self.ego_velocities[frame] = (0.0, 0.0)

            # Compute heading from velocity if non-zero
            vx, vy = self.ego_velocities[frame]
            if abs(vx) > 0.01 or abs(vy) > 0.01:
                self.ego_headings[frame] = math.degrees(math.atan2(vy, vx))
            else:
                self.ego_headings[frame] = self.ego_headings.get(frame - 1, 0.0)

        self.start_frame = start_frame
        self.end_frame = end_frame

    def _compute_baseline(self):
        """Compute baseline metrics for the original scenario."""
        validator = CollisionValidator(self.base_usd_path)
        results = validator.analyze_scenario(self.start_frame, self.end_frame)

        self.baseline_min_dist = float('inf')
        self.baseline_min_ttc = float('inf')
        self.baseline_collision = False
        self.critical_frame = self.start_frame
        self.critical_time = 0.0

        # Find minimum distance across all agents
        for r in results:
            if r['agent_id'] == 'EGO':
                continue
            if r['min_distance'] is not None and r['min_distance'] < self.baseline_min_dist:
                self.baseline_min_dist = r['min_distance']
                self.critical_frame = r['min_distance_frame']
                if r['min_distance_time'] is not None:
                    self.critical_time = r['min_distance_time']
            if r['min_ttc'] is not None and r['min_ttc'] < self.baseline_min_ttc:
                self.baseline_min_ttc = r['min_ttc']

        # Check for actual collision (distance below collision threshold)
        if self.baseline_min_dist < 0.5:  # Actual overlap
            self.baseline_collision = True

    def get_ego_speed_at_frame(self, frame: int) -> float:
        """Get ego vehicle speed magnitude at a frame."""
        vx, vy = self.ego_velocities.get(frame, (0.0, 0.0))
        return math.sqrt(vx**2 + vy**2)

    def get_average_ego_speed(self) -> float:
        """Get average ego speed over the scenario."""
        speeds = [self.get_ego_speed_at_frame(f) for f in range(self.start_frame, self.end_frame + 1)]
        return sum(speeds) / len(speeds) if speeds else 0.0

    def create_modified_stage(
        self,
        velocity_scale: float = 1.0,
        braking_start_frame: Optional[int] = None,
        deceleration: float = 0.0,
        speed_limit: Optional[float] = None
    ) -> str:
        """
        Create a modified USD stage with altered ego trajectory.

        Args:
            velocity_scale: Factor to multiply all velocities (0.9 = 10% slower)
            braking_start_frame: Frame to start braking (None = no braking)
            deceleration: Deceleration magnitude in m/s²
            speed_limit: Maximum speed cap in m/s (None = no limit)

        Returns:
            Path to the modified USD file
        """
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="ego_intervention_")

        # Copy base USD to temp
        base_filename = os.path.basename(self.base_usd_path)
        temp_base = os.path.join(temp_dir, base_filename)
        shutil.copy(self.base_usd_path, temp_base)

        # Create intervention layer
        intervention_filename = base_filename.replace("_base.usd", "_ego_intervention.usd")
        intervention_path = os.path.join(temp_dir, intervention_filename)

        # Create new layer
        intervention_layer = Sdf.Layer.CreateNew(intervention_path)
        intervention_layer.subLayerPaths.append(f"./{base_filename}")

        # Create override for EGO agent
        ego_path = "/World/Agents/Agent_EGO"
        ego_spec = Sdf.CreatePrimInLayer(intervention_layer, ego_path)
        ego_spec.specifier = Sdf.SpecifierOver

        # Compute modified velocities and positions
        dt = 0.1  # 10 fps
        current_speed = None

        velocity_x_samples = {}
        velocity_y_samples = {}
        position_samples = {}

        # Start from original first position
        current_x, current_y, current_z = self.ego_positions[self.start_frame]

        for frame in range(self.start_frame, self.end_frame + 1):
            orig_vx, orig_vy = self.ego_velocities[frame]
            orig_speed = math.sqrt(orig_vx**2 + orig_vy**2)

            # Initialize current speed tracking
            if current_speed is None:
                current_speed = orig_speed * velocity_scale

            # Apply interventions
            new_speed = orig_speed * velocity_scale

            # Apply braking if specified and past braking start
            if braking_start_frame is not None and frame >= braking_start_frame:
                # Reduce speed based on deceleration
                frames_since_braking = frame - braking_start_frame
                time_braking = frames_since_braking * dt
                speed_reduction = deceleration * time_braking
                new_speed = max(0.0, current_speed - speed_reduction)
                current_speed = new_speed

            # Apply speed limit if specified
            if speed_limit is not None:
                new_speed = min(new_speed, speed_limit)

            # Compute new velocity components (maintain direction)
            if orig_speed > 0.01:
                scale = new_speed / orig_speed
                new_vx = orig_vx * scale
                new_vy = orig_vy * scale
                dir_x = orig_vx / orig_speed
                dir_y = orig_vy / orig_speed
            else:
                new_vx, new_vy = 0.0, 0.0
                dir_x, dir_y = 0.0, 0.0

            velocity_x_samples[frame] = float(new_vx)
            velocity_y_samples[frame] = float(new_vy)

            # Integrate position (for frames after the first)
            if frame > self.start_frame:
                current_x += dir_x * new_speed * dt
                current_y += dir_y * new_speed * dt
                # Keep original z
                _, _, current_z = self.ego_positions[frame]

            position_samples[frame] = Gf.Vec3d(current_x, current_y, current_z)

        # Create velocity attributes with time samples
        vx_spec = Sdf.AttributeSpec(ego_spec, "waymo:velocityX", Sdf.ValueTypeNames.Float)
        vy_spec = Sdf.AttributeSpec(ego_spec, "waymo:velocityY", Sdf.ValueTypeNames.Float)

        # Set time samples using layer's SetTimeSample method
        vx_path = Sdf.Path(f"{ego_path}.waymo:velocityX")
        vy_path = Sdf.Path(f"{ego_path}.waymo:velocityY")
        translate_path = Sdf.Path(f"{ego_path}.xformOp:translate")

        for frame, vx in velocity_x_samples.items():
            intervention_layer.SetTimeSample(vx_path, frame, vx)
        for frame, vy in velocity_y_samples.items():
            intervention_layer.SetTimeSample(vy_path, frame, vy)

        # Create translate override
        translate_spec = Sdf.AttributeSpec(ego_spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
        for frame, pos in position_samples.items():
            intervention_layer.SetTimeSample(translate_path, frame, pos)

        # Add metadata
        world_spec = Sdf.CreatePrimInLayer(intervention_layer, "/World")
        world_spec.specifier = Sdf.SpecifierOver

        meta_attr = Sdf.AttributeSpec(world_spec, "waymo:interventionType", Sdf.ValueTypeNames.String)
        meta_attr.default = f"velocity_scale={velocity_scale},braking_frame={braking_start_frame},decel={deceleration}"

        intervention_layer.Save()

        return intervention_path


    def validate_intervention(self, intervention_usd_path: str) -> Tuple[float, float, bool]:
        """
        Validate an intervention by computing new safety metrics.

        Returns:
            (min_distance, min_ttc, collision_detected)
        """
        try:
            validator = CollisionValidator(intervention_usd_path)
            results = validator.analyze_scenario(self.start_frame, self.end_frame)

            min_dist = float('inf')
            min_ttc = float('inf')

            for r in results:
                if r['agent_id'] == 'EGO':
                    continue
                if r['min_distance'] is not None and r['min_distance'] < min_dist:
                    min_dist = r['min_distance']
                if r['min_ttc'] is not None and r['min_ttc'] < min_ttc:
                    min_ttc = r['min_ttc']

            collision = min_dist < 0.5

            return min_dist, min_ttc, collision

        except Exception as e:
            print(f"Validation error: {e}")
            return 0.0, 0.0, True
        finally:
            # Cleanup temp directory
            temp_dir = os.path.dirname(intervention_usd_path)
            if temp_dir.startswith(tempfile.gettempdir()):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_uniform_speed_reduction(self, reduction_percent: float) -> InterventionResult:
        """
        Test uniform speed reduction intervention.

        Args:
            reduction_percent: Speed reduction as percentage (5 = 5% reduction)

        Returns:
            InterventionResult with metrics
        """
        velocity_scale = 1.0 - (reduction_percent / 100.0)

        intervention_path = self.create_modified_stage(velocity_scale=velocity_scale)
        new_dist, new_ttc, new_collision = self.validate_intervention(intervention_path)

        return InterventionResult(
            intervention_type="uniform_speed_reduction",
            intervention_value=reduction_percent,
            intervention_description=f"Reduce speed by {reduction_percent:.1f}%",
            original_min_distance=self.baseline_min_dist,
            original_min_ttc=self.baseline_min_ttc,
            original_collision=self.baseline_collision,
            new_min_distance=new_dist,
            new_min_ttc=new_ttc,
            new_collision=new_collision,
            collision_avoided=self.baseline_collision and not new_collision,
            threshold_achieved=new_dist >= self.safety_threshold,
            safety_margin_gained=new_dist - self.baseline_min_dist
        )

    def test_targeted_braking(
        self,
        braking_profile: BrakingProfile,
        frames_before_critical: int = 10
    ) -> InterventionResult:
        """
        Test targeted braking intervention.

        Args:
            braking_profile: Intensity of braking
            frames_before_critical: When to start braking (frames before critical event)

        Returns:
            InterventionResult with metrics
        """
        deceleration = BRAKING_DECELERATION[braking_profile]
        braking_start = max(self.start_frame, self.critical_frame - frames_before_critical)

        intervention_path = self.create_modified_stage(
            braking_start_frame=braking_start,
            deceleration=deceleration
        )
        new_dist, new_ttc, new_collision = self.validate_intervention(intervention_path)

        time_before = frames_before_critical * 0.1  # seconds

        return InterventionResult(
            intervention_type="targeted_braking",
            intervention_value=deceleration,
            intervention_description=f"Apply {braking_profile.value} braking ({deceleration} m/s²) "
                                    f"{time_before:.1f}s before critical point",
            original_min_distance=self.baseline_min_dist,
            original_min_ttc=self.baseline_min_ttc,
            original_collision=self.baseline_collision,
            new_min_distance=new_dist,
            new_min_ttc=new_ttc,
            new_collision=new_collision,
            collision_avoided=self.baseline_collision and not new_collision,
            threshold_achieved=new_dist >= self.safety_threshold,
            safety_margin_gained=new_dist - self.baseline_min_dist
        )

    def find_minimal_speed_reduction(
        self,
        step_size: float = 5.0,
        max_reduction: float = 50.0
    ) -> MinimalInterventionExplanation:
        """
        Find minimum speed reduction needed to achieve safety threshold.

        Uses graduated search: 5%, 10%, 15%... until threshold achieved.

        Args:
            step_size: Increment for speed reduction search (default 5%)
            max_reduction: Maximum reduction to test (default 50%)

        Returns:
            MinimalInterventionExplanation with results
        """
        all_interventions = []
        minimal_intervention = None

        reduction = step_size
        while reduction <= max_reduction:
            result = self.test_uniform_speed_reduction(reduction)
            all_interventions.append(result)

            if result.threshold_achieved and minimal_intervention is None:
                minimal_intervention = result
                break

            reduction += step_size

        # Generate explanation
        if minimal_intervention:
            explanation = (
                f"To achieve safe distance (>{self.safety_threshold}m), "
                f"ego vehicle should reduce speed by {minimal_intervention.intervention_value:.1f}%. "
                f"This increases minimum distance from {self.baseline_min_dist:.2f}m to "
                f"{minimal_intervention.new_min_distance:.2f}m."
            )
            feasible = minimal_intervention.intervention_value <= 30  # Reasonable reduction
        else:
            explanation = (
                f"Speed reduction alone (up to {max_reduction}%) is insufficient to achieve "
                f"safe distance threshold of {self.safety_threshold}m. "
                f"Consider braking intervention or agent avoidance."
            )
            feasible = False

        return MinimalInterventionExplanation(
            scenario_id=self.scenario_id,
            scenario_path=self.base_usd_path,
            safety_threshold_m=self.safety_threshold,
            original_min_distance=self.baseline_min_dist,
            original_min_ttc=self.baseline_min_ttc,
            original_collision=self.baseline_collision,
            critical_time_s=self.critical_time,
            critical_frame=self.critical_frame,
            minimal_intervention=minimal_intervention,
            intervention_mode="uniform_speed_reduction",
            all_interventions=all_interventions,
            explanation=explanation,
            physically_feasible=feasible,
            feasibility_notes="Speed reduction >30% may not be practical in real-time driving"
        )

    def find_minimal_braking(
        self,
        max_lead_time_s: float = 3.0,
        time_step_s: float = 0.5
    ) -> MinimalInterventionExplanation:
        """
        Find minimum braking intervention needed.

        Tests different braking profiles at various lead times.

        Args:
            max_lead_time_s: Maximum time before critical event to start braking
            time_step_s: Time increment for search

        Returns:
            MinimalInterventionExplanation with results
        """
        all_interventions = []
        minimal_intervention = None

        # Test from latest to earliest (minimum intervention)
        lead_times = np.arange(time_step_s, max_lead_time_s + time_step_s, time_step_s)

        # Start with comfortable braking, escalate if needed
        for profile in [BrakingProfile.COMFORTABLE, BrakingProfile.FIRM,
                        BrakingProfile.HARD, BrakingProfile.EMERGENCY]:
            for lead_time in lead_times:
                frames_before = int(lead_time / 0.1)  # 10 fps
                result = self.test_targeted_braking(profile, frames_before)
                all_interventions.append(result)

                if result.threshold_achieved and minimal_intervention is None:
                    minimal_intervention = result
                    break

            if minimal_intervention:
                break

        # Generate explanation
        if minimal_intervention:
            decel = minimal_intervention.intervention_value
            profile_name = next(
                p.value for p, d in BRAKING_DECELERATION.items() if d == decel
            )
            # Extract lead time from description (format: "...X.Xs before critical point")
            import re
            match = re.search(r'(\d+\.?\d*)s before', minimal_intervention.intervention_description)
            lead_time = float(match.group(1)) if match else 0.0

            explanation = (
                f"To achieve safe distance (>{self.safety_threshold}m), "
                f"ego vehicle should apply {profile_name} braking ({decel} m/s²) "
                f"{lead_time:.1f}s before the critical point. "
                f"This increases minimum distance from {self.baseline_min_dist:.2f}m to "
                f"{minimal_intervention.new_min_distance:.2f}m."
            )

            # Assess feasibility
            feasible = decel <= BRAKING_DECELERATION[BrakingProfile.FIRM]
            feasibility_notes = (
                "Comfortable to firm braking is achievable in normal driving. "
                if feasible else
                "Hard/emergency braking may require advanced driver assistance systems."
            )
        else:
            explanation = (
                f"Even emergency braking is insufficient to achieve "
                f"safe distance threshold of {self.safety_threshold}m. "
                f"Collision may be unavoidable without steering intervention."
            )
            feasible = False
            feasibility_notes = "Braking alone cannot prevent this scenario."

        return MinimalInterventionExplanation(
            scenario_id=self.scenario_id,
            scenario_path=self.base_usd_path,
            safety_threshold_m=self.safety_threshold,
            original_min_distance=self.baseline_min_dist,
            original_min_ttc=self.baseline_min_ttc,
            original_collision=self.baseline_collision,
            critical_time_s=self.critical_time,
            critical_frame=self.critical_frame,
            minimal_intervention=minimal_intervention,
            intervention_mode="targeted_braking",
            all_interventions=all_interventions,
            explanation=explanation,
            physically_feasible=feasible,
            feasibility_notes=feasibility_notes
        )

    def comprehensive_analysis(self) -> Dict[str, MinimalInterventionExplanation]:
        """
        Run comprehensive intervention analysis with all modes.

        Returns:
            Dict mapping intervention mode to explanation
        """
        results = {}

        # Uniform speed reduction
        print(f"  Testing uniform speed reduction...")
        results['uniform'] = self.find_minimal_speed_reduction()

        # Targeted braking
        print(f"  Testing targeted braking...")
        results['braking'] = self.find_minimal_braking()

        return results


def analyze_scenario_interventions(
    scenario_path: str,
    safety_threshold: float = 2.5,
    verbose: bool = True
) -> Dict[str, MinimalInterventionExplanation]:
    """
    Analyze a scenario for minimal interventions.

    Args:
        scenario_path: Path to base USD scenario
        safety_threshold: Target minimum safe distance
        verbose: Print progress

    Returns:
        Dict of intervention explanations
    """
    scenario_id = os.path.basename(scenario_path).replace("_base.usd", "")

    if verbose:
        print(f"\n{'='*70}")
        print(f"MINIMAL INTERVENTION ANALYSIS: {scenario_id}")
        print(f"Safety Threshold: {safety_threshold}m")
        print(f"{'='*70}")

    engine = EgoInterventionEngine(scenario_path, safety_threshold)

    if verbose:
        print(f"\nBaseline Metrics:")
        print(f"  Min Distance: {engine.baseline_min_dist:.3f}m")
        print(f"  Min TTC: {engine.baseline_min_ttc:.3f}s")
        print(f"  Collision: {engine.baseline_collision}")
        print(f"  Critical Frame: {engine.critical_frame}")
        print(f"  Avg Ego Speed: {engine.get_average_ego_speed():.2f} m/s")

    results = engine.comprehensive_analysis()

    if verbose:
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")

        for mode, explanation in results.items():
            print(f"\n--- {mode.upper()} ---")
            print(f"Explanation: {explanation.explanation}")
            print(f"Physically Feasible: {explanation.physically_feasible}")

            if explanation.minimal_intervention:
                mi = explanation.minimal_intervention
                print(f"\nMinimal Intervention:")
                print(f"  Type: {mi.intervention_type}")
                print(f"  Value: {mi.intervention_value}")
                print(f"  Description: {mi.intervention_description}")
                print(f"  New Min Distance: {mi.new_min_distance:.3f}m")
                print(f"  Safety Margin Gained: {mi.safety_margin_gained:.3f}m")

            print(f"\nAll Interventions Tested: {len(explanation.all_interventions)}")
            for i, interv in enumerate(explanation.all_interventions[:5]):
                status = "PASS" if interv.threshold_achieved else "FAIL"
                print(f"  {i+1}. {interv.intervention_description}: "
                      f"{interv.new_min_distance:.3f}m [{status}]")

    return results


def batch_analyze_scenarios(
    scenarios_dir: str = "scenarios",
    scenarios_csv: str = "scenarios.csv",
    safety_threshold: float = 2.5,
    output_file: str = "evaluation_output/minimal_interventions.json",
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Analyze multiple scenarios for minimal interventions.

    Returns dict with classification:
    {
        'scenario_id': {
            'uniform': {...},
            'braking': {...},
            'classification': 'ego_resolvable' | 'non_ego_resolvable',
            'min_intervention_type': 'speed_reduction' | 'braking' | None,
            'min_intervention_value': float | None,  # % or m/s²
            'achievable_margin_m': float,  # Best distance achieved
            'braking_profile': str | None,  # comfortable/firm/hard/emergency
        }
    }
    """
    import csv
    import json
    from dataclasses import asdict

    # Load scenario IDs
    with open(scenarios_csv, newline='') as f:
        reader = csv.DictReader(f)
        scenario_ids = [row['scenario_id'].strip() for row in reader]

    all_results = {}

    for sid in scenario_ids:
        scenario_path = os.path.join(scenarios_dir, f"{sid}_base.usd")
        if not os.path.exists(scenario_path):
            continue

        if verbose:
            print(f"\nProcessing: {sid}")

        try:
            results = analyze_scenario_interventions(
                scenario_path, safety_threshold, verbose=False
            )

            # Convert to serializable format (handle numpy types)
            def to_native(val):
                """Convert numpy types to Python native types."""
                if hasattr(val, 'item'):  # numpy scalar
                    return val.item()
                elif isinstance(val, dict):
                    return {k: to_native(v) for k, v in val.items()}
                elif isinstance(val, (list, tuple)):
                    return [to_native(v) for v in val]
                return val

            # Build base result with intervention modes
            scenario_result = {
                mode: {
                    'scenario_id': exp.scenario_id,
                    'safety_threshold_m': float(exp.safety_threshold_m),
                    'original_min_distance': float(exp.original_min_distance),
                    'original_min_ttc': float(exp.original_min_ttc),
                    'explanation': exp.explanation,
                    'physically_feasible': bool(exp.physically_feasible),
                    'minimal_intervention': to_native(asdict(exp.minimal_intervention)) if exp.minimal_intervention else None,
                    'num_interventions_tested': len(exp.all_interventions)
                }
                for mode, exp in results.items()
            }

            # Determine classification
            uniform_exp = results.get('uniform')
            braking_exp = results.get('braking')

            speed_resolved = (uniform_exp and uniform_exp.minimal_intervention and
                            uniform_exp.minimal_intervention.threshold_achieved)
            braking_resolved = (braking_exp and braking_exp.minimal_intervention and
                              braking_exp.minimal_intervention.threshold_achieved)

            is_ego_resolvable = speed_resolved or braking_resolved

            # Determine minimum intervention (prefer speed reduction as less aggressive)
            min_intervention_type = None
            min_intervention_value = None
            braking_profile = None
            achievable_margin = 0.0

            if speed_resolved and braking_resolved:
                # Both work - pick the "gentler" one
                speed_val = uniform_exp.minimal_intervention.intervention_value
                braking_val = braking_exp.minimal_intervention.intervention_value

                # Compare: speed reduction % vs braking profile severity
                # Speed reduction under 20% is generally gentler than any braking
                if speed_val <= 20:
                    min_intervention_type = 'speed_reduction'
                    min_intervention_value = speed_val
                    achievable_margin = uniform_exp.minimal_intervention.new_min_distance
                else:
                    # Check braking severity
                    if braking_val <= BRAKING_DECELERATION[BrakingProfile.COMFORTABLE]:
                        min_intervention_type = 'braking'
                        min_intervention_value = braking_val
                        achievable_margin = braking_exp.minimal_intervention.new_min_distance
                    else:
                        min_intervention_type = 'speed_reduction'
                        min_intervention_value = speed_val
                        achievable_margin = uniform_exp.minimal_intervention.new_min_distance

                # Determine braking profile if braking is an option
                for profile, decel in BRAKING_DECELERATION.items():
                    if abs(braking_val - decel) < 0.1:
                        braking_profile = profile.value
                        break

            elif speed_resolved:
                min_intervention_type = 'speed_reduction'
                min_intervention_value = uniform_exp.minimal_intervention.intervention_value
                achievable_margin = uniform_exp.minimal_intervention.new_min_distance

            elif braking_resolved:
                min_intervention_type = 'braking'
                min_intervention_value = braking_exp.minimal_intervention.intervention_value
                achievable_margin = braking_exp.minimal_intervention.new_min_distance

                # Determine braking profile
                for profile, decel in BRAKING_DECELERATION.items():
                    if abs(min_intervention_value - decel) < 0.1:
                        braking_profile = profile.value
                        break

            else:
                # Not resolvable - find best achievable margin
                best_margin = 0.0
                if uniform_exp and uniform_exp.all_interventions:
                    for interv in uniform_exp.all_interventions:
                        if interv.new_min_distance > best_margin:
                            best_margin = interv.new_min_distance
                if braking_exp and braking_exp.all_interventions:
                    for interv in braking_exp.all_interventions:
                        if interv.new_min_distance > best_margin:
                            best_margin = interv.new_min_distance
                achievable_margin = best_margin

            # Add classification fields
            scenario_result['classification'] = 'ego_resolvable' if is_ego_resolvable else 'non_ego_resolvable'
            scenario_result['min_intervention_type'] = min_intervention_type
            scenario_result['min_intervention_value'] = float(min_intervention_value) if min_intervention_value else None
            scenario_result['achievable_margin_m'] = float(achievable_margin)
            scenario_result['braking_profile'] = braking_profile

            all_results[sid] = scenario_result

            if verbose:
                status = "EGO-RESOLVABLE" if is_ego_resolvable else "NON-EGO-RESOLVABLE"
                print(f"  Classification: {status}")
                if min_intervention_type == 'speed_reduction':
                    print(f"  Min intervention: {min_intervention_value:.1f}% speed reduction")
                elif min_intervention_type == 'braking':
                    print(f"  Min intervention: {braking_profile} braking ({min_intervention_value:.1f} m/s²)")
                print(f"  Achievable margin: {achievable_margin:.2f}m")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to: {output_file}")

    return all_results


def compute_ego_intervention_summary(results: Dict[str, Dict]) -> dict:
    """
    Compute aggregate statistics for ego intervention analysis.

    Args:
        results: Output from batch_analyze_scenarios()

    Returns:
        {
            'total_scenarios': int,
            'ego_resolvable_count': int,
            'ego_resolvable_rate': float,
            'non_ego_resolvable_count': int,
            'avg_speed_reduction_needed': float | None,  # For speed-resolvable scenarios
            'avg_braking_decel_needed': float | None,    # For braking-resolvable scenarios
            'braking_profile_distribution': {
                'comfortable': int,
                'firm': int,
                'hard': int,
                'emergency': int
            },
            'intervention_type_distribution': {
                'speed_reduction': int,
                'braking': int,
                'none': int
            },
            'avg_achievable_margin_m': float,
            'non_resolvable_scenarios': [str],  # List of scenario IDs
            'ego_resolvable_scenarios': [str],  # List of scenario IDs
            'by_intervention_type': {
                'speed_reduction': {
                    'count': int,
                    'avg_value': float,
                    'min_value': float,
                    'max_value': float,
                    'scenarios': [str]
                },
                'braking': {
                    'count': int,
                    'avg_value': float,
                    'min_value': float,
                    'max_value': float,
                    'scenarios': [str]
                }
            }
        }
    """
    total = len(results)
    ego_resolvable = []
    non_resolvable = []

    speed_reduction_values = []
    speed_reduction_scenarios = []
    braking_decel_values = []
    braking_scenarios = []
    achievable_margins = []

    braking_profile_dist = {
        'comfortable': 0,
        'firm': 0,
        'hard': 0,
        'emergency': 0
    }

    intervention_type_dist = {
        'speed_reduction': 0,
        'braking': 0,
        'none': 0
    }

    for sid, data in results.items():
        classification = data.get('classification', 'non_ego_resolvable')
        min_type = data.get('min_intervention_type')
        min_value = data.get('min_intervention_value')
        braking_profile = data.get('braking_profile')
        achievable_margin = data.get('achievable_margin_m', 0)

        achievable_margins.append(achievable_margin)

        if classification == 'ego_resolvable':
            ego_resolvable.append(sid)

            if min_type == 'speed_reduction':
                intervention_type_dist['speed_reduction'] += 1
                if min_value is not None:
                    speed_reduction_values.append(min_value)
                    speed_reduction_scenarios.append(sid)

            elif min_type == 'braking':
                intervention_type_dist['braking'] += 1
                if min_value is not None:
                    braking_decel_values.append(min_value)
                    braking_scenarios.append(sid)

                if braking_profile and braking_profile in braking_profile_dist:
                    braking_profile_dist[braking_profile] += 1

        else:
            non_resolvable.append(sid)
            intervention_type_dist['none'] += 1

    # Compute averages
    avg_speed_reduction = None
    if speed_reduction_values:
        avg_speed_reduction = sum(speed_reduction_values) / len(speed_reduction_values)

    avg_braking_decel = None
    if braking_decel_values:
        avg_braking_decel = sum(braking_decel_values) / len(braking_decel_values)

    avg_achievable_margin = sum(achievable_margins) / len(achievable_margins) if achievable_margins else 0

    # Build detailed by-type breakdown
    by_intervention_type = {}

    if speed_reduction_values:
        by_intervention_type['speed_reduction'] = {
            'count': len(speed_reduction_values),
            'avg_value': round(sum(speed_reduction_values) / len(speed_reduction_values), 2),
            'min_value': round(min(speed_reduction_values), 2),
            'max_value': round(max(speed_reduction_values), 2),
            'scenarios': speed_reduction_scenarios
        }

    if braking_decel_values:
        by_intervention_type['braking'] = {
            'count': len(braking_decel_values),
            'avg_value': round(sum(braking_decel_values) / len(braking_decel_values), 2),
            'min_value': round(min(braking_decel_values), 2),
            'max_value': round(max(braking_decel_values), 2),
            'scenarios': braking_scenarios
        }

    return {
        'total_scenarios': total,
        'ego_resolvable_count': len(ego_resolvable),
        'ego_resolvable_rate': round(len(ego_resolvable) / total, 3) if total > 0 else 0,
        'non_ego_resolvable_count': len(non_resolvable),
        'avg_speed_reduction_needed': round(avg_speed_reduction, 2) if avg_speed_reduction else None,
        'avg_braking_decel_needed': round(avg_braking_decel, 2) if avg_braking_decel else None,
        'braking_profile_distribution': braking_profile_dist,
        'intervention_type_distribution': intervention_type_dist,
        'avg_achievable_margin_m': round(avg_achievable_margin, 3),
        'non_resolvable_scenarios': non_resolvable,
        'ego_resolvable_scenarios': ego_resolvable,
        'by_intervention_type': by_intervention_type
    }


def print_ego_intervention_summary(summary: dict):
    """Print formatted ego intervention summary."""
    print("\n" + "=" * 80)
    print("EGO INTERVENTION SUMMARY")
    print("=" * 80)

    total = summary['total_scenarios']
    resolvable = summary['ego_resolvable_count']
    non_resolvable = summary['non_ego_resolvable_count']

    print(f"\nOverall Classification:")
    print(f"  Total scenarios: {total}")
    print(f"  Ego-resolvable: {resolvable} ({summary['ego_resolvable_rate']*100:.1f}%)")
    print(f"  Non-ego-resolvable: {non_resolvable} ({non_resolvable/total*100:.1f}%)" if total > 0 else "")

    print(f"\nIntervention Type Distribution:")
    for itype, count in summary['intervention_type_distribution'].items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {itype}: {count} ({pct:.1f}%)")

    if summary['avg_speed_reduction_needed']:
        print(f"\nSpeed Reduction Analysis:")
        print(f"  Average needed: {summary['avg_speed_reduction_needed']:.1f}%")
        if 'speed_reduction' in summary['by_intervention_type']:
            sr = summary['by_intervention_type']['speed_reduction']
            print(f"  Range: {sr['min_value']:.1f}% - {sr['max_value']:.1f}%")
            print(f"  Scenarios: {sr['count']}")

    if summary['avg_braking_decel_needed']:
        print(f"\nBraking Analysis:")
        print(f"  Average deceleration: {summary['avg_braking_decel_needed']:.1f} m/s²")
        if 'braking' in summary['by_intervention_type']:
            br = summary['by_intervention_type']['braking']
            print(f"  Range: {br['min_value']:.1f} - {br['max_value']:.1f} m/s²")
            print(f"  Scenarios: {br['count']}")

        print(f"\n  Braking Profile Distribution:")
        for profile, count in summary['braking_profile_distribution'].items():
            if count > 0:
                print(f"    {profile}: {count}")

    print(f"\nAverage Achievable Margin: {summary['avg_achievable_margin_m']:.3f}m")

    if summary['non_resolvable_scenarios']:
        print(f"\nNon-Resolvable Scenarios ({len(summary['non_resolvable_scenarios'])}):")
        # Show first 10
        for sid in summary['non_resolvable_scenarios'][:10]:
            print(f"  - {sid}")
        if len(summary['non_resolvable_scenarios']) > 10:
            print(f"  ... and {len(summary['non_resolvable_scenarios']) - 10} more")


def print_intervention_summary(results: Dict[str, Dict]):
    """Print summary of intervention analysis across scenarios."""
    print("\n" + "=" * 80)
    print("MINIMAL INTERVENTION SUMMARY")
    print("=" * 80)

    uniform_results = []
    braking_results = []

    for sid, modes in results.items():
        if 'uniform' in modes and modes['uniform']['minimal_intervention']:
            uniform_results.append({
                'scenario_id': sid,
                'reduction': modes['uniform']['minimal_intervention']['intervention_value'],
                'new_dist': modes['uniform']['minimal_intervention']['new_min_distance'],
                'feasible': modes['uniform']['physically_feasible']
            })

        if 'braking' in modes and modes['braking']['minimal_intervention']:
            braking_results.append({
                'scenario_id': sid,
                'deceleration': modes['braking']['minimal_intervention']['intervention_value'],
                'new_dist': modes['braking']['minimal_intervention']['new_min_distance'],
                'feasible': modes['braking']['physically_feasible']
            })

    print(f"\n--- UNIFORM SPEED REDUCTION ---")
    print(f"Scenarios resolved: {len(uniform_results)}/{len(results)}")
    if uniform_results:
        reductions = [r['reduction'] for r in uniform_results]
        print(f"Reduction range: {min(reductions):.1f}% - {max(reductions):.1f}%")
        print(f"Average reduction needed: {sum(reductions)/len(reductions):.1f}%")
        feasible = sum(1 for r in uniform_results if r['feasible'])
        print(f"Physically feasible: {feasible}/{len(uniform_results)}")

    print(f"\n--- TARGETED BRAKING ---")
    print(f"Scenarios resolved: {len(braking_results)}/{len(results)}")
    if braking_results:
        decels = [r['deceleration'] for r in braking_results]
        print(f"Deceleration range: {min(decels):.1f} - {max(decels):.1f} m/s²")
        print(f"Average deceleration needed: {sum(decels)/len(decels):.1f} m/s²")
        feasible = sum(1 for r in braking_results if r['feasible'])
        print(f"Physically feasible: {feasible}/{len(braking_results)}")

    # Breakdown by braking profile
    print(f"\nBraking Profile Distribution:")
    profile_counts = {}
    for r in braking_results:
        decel = r['deceleration']
        for profile, d in BRAKING_DECELERATION.items():
            if abs(decel - d) < 0.1:
                profile_counts[profile.value] = profile_counts.get(profile.value, 0) + 1
                break
    for profile, count in profile_counts.items():
        print(f"  {profile}: {count} scenarios")


# --- CLI ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Minimal-Intervention Counterfactual Explanations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single scenario
  python minimal_intervention.py --scenario scenarios/abc123_base.usd

  # Analyze all scenarios
  python minimal_intervention.py --batch

  # Custom safety threshold
  python minimal_intervention.py --scenario scenarios/abc123_base.usd --threshold 3.0
        """
    )

    parser.add_argument("--scenario", help="Path to single scenario USD")
    parser.add_argument("--batch", action="store_true", help="Analyze all scenarios")
    parser.add_argument("--threshold", type=float, default=2.5,
                        help="Safety distance threshold (meters)")
    parser.add_argument("--scenarios-dir", default="scenarios",
                        help="Directory with USD scenarios")
    parser.add_argument("--scenarios-csv", default="scenarios.csv",
                        help="CSV with scenario metadata")
    parser.add_argument("--output", default="evaluation_output/minimal_interventions.json",
                        help="Output file for batch results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    if args.scenario:
        analyze_scenario_interventions(args.scenario, args.threshold, verbose=True)

    elif args.batch:
        results = batch_analyze_scenarios(
            scenarios_dir=args.scenarios_dir,
            scenarios_csv=args.scenarios_csv,
            safety_threshold=args.threshold,
            output_file=args.output,
            verbose=args.verbose
        )
        print_intervention_summary(results)

        # Compute and print ego intervention summary
        summary = compute_ego_intervention_summary(results)
        print_ego_intervention_summary(summary)

        # Save summary to separate file
        import json
        summary_file = args.output.replace('.json', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to: {summary_file}")

    else:
        parser.print_help()
