from pxr import Usd, UsdGeom, Gf, Sdf, Vt
import math
from waymo_open_dataset.protos import scenario_pb2
import argparse
import glob
import csv
import os
from typing import Optional, List, Dict, Any

# Waymo object type enum mapping (from scenario_pb2.Track.ObjectType)
# Used for semantic ranking in diagnosis engine
WAYMO_OBJECT_TYPES = {
    0: "TYPE_UNSET",
    1: "TYPE_VEHICLE",
    2: "TYPE_PEDESTRIAN",
    3: "TYPE_CYCLIST",
    4: "TYPE_OTHER",
}


def create_staged_agent(stage, agent_id, agent_type, width, length, height, waymo_object_type=0):
    """
    Defines a clean Agent prim structure:
    /World/Agents/Agent_{id}  <-- Xform (Holds Animation + Waymo metadata)
       /Geometry              <-- Cube (Holds Dimensions)

    Custom attributes added for counterfactual analysis:
    - waymo:objectType (int): Waymo enum value (0-4)
    - waymo:objectTypeString (string): Human-readable type name
    - waymo:velocityX (time-sampled float): Per-frame X velocity (m/s)
    - waymo:velocityY (time-sampled float): Per-frame Y velocity (m/s)
    - waymo:extentLength (time-sampled or static float): Bounding box length
    - waymo:extentWidth (time-sampled or static float): Bounding box width
    - waymo:extentHeight (time-sampled or static float): Bounding box height
    - waymo:firstValidFrame (int): First frame where track is valid
    - waymo:lastValidFrame (int): Last frame where track is valid

    Standard attribute set by the animation loop:
    - visibility (time-sampled token): "inherited" while the track is valid,
      "invisible" while it is not, so gaps are not rendered.
    """
    # Sanitize ID for USD path
    safe_id = str(agent_id).replace("-", "_")
    prim_path = f"/World/Agents/Agent_{safe_id}"

    # 1. Create the Transform (Animation holder)
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()

    # 2. Add Waymo object type attributes (for semantic ranking in diagnosis engine)
    # Integer enum value from scenario_pb2.Track.ObjectType
    prim.CreateAttribute("waymo:objectType", Sdf.ValueTypeNames.Int).Set(waymo_object_type)
    # Human-readable string for debugging/visualization
    type_string = WAYMO_OBJECT_TYPES.get(waymo_object_type, "TYPE_UNKNOWN")
    prim.CreateAttribute("waymo:objectTypeString", Sdf.ValueTypeNames.String).Set(type_string)

    # 3. Create velocity attributes (will be populated with time samples in animation loop)
    # These are critical for TTC computation: closing speed v_c(t) = |v_ego - v_other|
    prim.CreateAttribute("waymo:velocityX", Sdf.ValueTypeNames.Float)
    prim.CreateAttribute("waymo:velocityY", Sdf.ValueTypeNames.Float)

    # 4. Create extent attributes (will be time-sampled if dimensions vary per-state)
    prim.CreateAttribute("waymo:extentLength", Sdf.ValueTypeNames.Float)
    prim.CreateAttribute("waymo:extentWidth", Sdf.ValueTypeNames.Float)
    prim.CreateAttribute("waymo:extentHeight", Sdf.ValueTypeNames.Float)

    # 5. Track validity span attributes (will be set after processing all states)
    prim.CreateAttribute("waymo:firstValidFrame", Sdf.ValueTypeNames.Int)
    prim.CreateAttribute("waymo:lastValidFrame", Sdf.ValueTypeNames.Int)

    # 6. Create the Geometry (Visual representation)
    mesh = UsdGeom.Cube.Define(stage, f"{prim_path}/Geometry")

    # Scale the cube to match agent dimensions (USD Cube is 2x2x2 by default, so scale by half)
    mesh.AddScaleOp().Set(Gf.Vec3f(length/2, width/2, height/2))

    # Set color based on type (Ego=Blue, Others=Red)
    color = Gf.Vec3f(0, 0, 1) if agent_type == "EGO" else Gf.Vec3f(1, 0, 0)
    mesh.GetDisplayColorAttr().Set([color])

    return xform

def waymo_to_usd(scenario, output_path):
    print(f"🔄 Converting Scenario {scenario.scenario_id} to USD...")

    # 1. Setup Stage
    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)  # Waymo is Z-up
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)        # 1 unit = 1 meter

    # Create Root
    root = UsdGeom.Xform.Define(stage, "/World")
    root_prim = root.GetPrim()

    stage.SetStartTimeCode(0)

    num_steps = len(scenario.timestamps_seconds)  # Waymo defines per-step timestamps
    if num_steps == 0:
        # fallback if timestamps_seconds missing for some reason
        num_steps = max(len(t.states) for t in scenario.tracks)

    stage.SetEndTimeCode(num_steps - 1)
    stage.SetFramesPerSecond(10)

    # --- SCENARIO METADATA (stored on /World prim) ---
    # timestamps_seconds: Array of absolute timestamps for each frame (useful for TTC timing)
    if scenario.timestamps_seconds:
        timestamps_attr = root_prim.CreateAttribute(
            "waymo:timestampsSeconds", Sdf.ValueTypeNames.FloatArray
        )
        timestamps_attr.Set(list(scenario.timestamps_seconds))

    # current_time_index: The "current" frame in the scenario (where prediction starts)
    root_prim.CreateAttribute("waymo:currentTimeIndex", Sdf.ValueTypeNames.Int).Set(
        scenario.current_time_index
    )

    # tracks_to_predict: List of track IDs that Waymo flags as "interesting" for prediction
    # These are typically the agents involved in the scenario's key interaction
    if scenario.tracks_to_predict:
        predict_ids = [ttp.track_index for ttp in scenario.tracks_to_predict]
        root_prim.CreateAttribute(
            "waymo:tracksToPredict", Sdf.ValueTypeNames.IntArray
        ).Set(predict_ids)

    
    # 2. Get SDC ID
    sdc_track = scenario.tracks[scenario.sdc_track_index]

    # 3. Process All Agents
    for track in scenario.tracks:
        is_ego = (track.id == sdc_track.id)

        # FORCE NAME TO "EGO" SO VALIDATOR FINDS IT
        agent_name = "EGO" if is_ego else track.id
        agent_type = "EGO" if is_ego else "AGENT"

        # --- ROBUST DIMENSION EXTRACTION (track-level defaults) ---
        l, w, h = 4.5, 2.0, 1.6  # Default values

        if is_ego:
            print(f"  → Processing EGO (track.id={track.id}, {len(track.states)} states)")

        try:
            # Attempt 1: Standard Waymo 1.x location
            if track.length > 0: l = track.length
            if track.width > 0: w = track.width
            if track.height > 0: h = track.height
        except AttributeError:
            try:
                # Attempt 2: Nested box_dimensions
                if track.box_dimensions.length > 0: l = track.box_dimensions.length
                if track.box_dimensions.width > 0: w = track.box_dimensions.width
                if track.box_dimensions.height > 0: h = track.box_dimensions.height
            except AttributeError:
                # Keep defaults if all else fails
                pass

        # Get Waymo object type enum (TYPE_VEHICLE=1, TYPE_PEDESTRIAN=2, etc.)
        waymo_obj_type = track.object_type if hasattr(track, 'object_type') else 0

        # Create Prim with object type metadata
        xform = create_staged_agent(stage, agent_name, agent_type, w, l, h, waymo_obj_type)
        prim = xform.GetPrim()

        # Get attribute handles for time-sampled data
        vel_x_attr = prim.GetAttribute("waymo:velocityX")
        vel_y_attr = prim.GetAttribute("waymo:velocityY")
        extent_l_attr = prim.GetAttribute("waymo:extentLength")
        extent_w_attr = prim.GetAttribute("waymo:extentWidth")
        extent_h_attr = prim.GetAttribute("waymo:extentHeight")

        # Add Operation Stacks (Order matters! T * R * S)
        translate_op = xform.AddTranslateOp()
        rotate_op = xform.AddRotateZOp()

        # Track validity span
        first_valid_frame = None
        last_valid_frame = None

        # Visibility follows track validity. USD holds the nearest transform
        # sample outside the valid span and interpolates linearly across
        # internal gaps, so an agent would otherwise appear frozen or
        # gliding where Waymo has no observation. Token attributes use held
        # interpolation, so one sample per transition is equivalent to one
        # sample per frame.
        vis_attr = UsdGeom.Imageable(xform).CreateVisibilityAttr()
        prev_visibility = None
        for i, state in enumerate(track.states):
            visibility = UsdGeom.Tokens.inherited if state.valid else UsdGeom.Tokens.invisible
            if visibility != prev_visibility:
                vis_attr.Set(visibility, i)
                prev_visibility = visibility

        # Check if per-state dimensions exist and vary (for time-sampling decision)
        # Waymo states may have length/width/height fields that can change per frame
        state_has_dimensions = False
        dimensions_vary = False
        if track.states:
            first_state = track.states[0]
            state_has_dimensions = (
                hasattr(first_state, 'length') and
                hasattr(first_state, 'width') and
                hasattr(first_state, 'height')
            )
            if state_has_dimensions and len(track.states) > 1:
                # Check if dimensions actually vary across states
                for s in track.states[1:]:
                    if s.valid and first_state.valid:
                        if (s.length != first_state.length or
                            s.width != first_state.width or
                            s.height != first_state.height):
                            dimensions_vary = True
                            break

        # 4. ANIMATION LOOP
        for i, state in enumerate(track.states):
            if not state.valid:
                continue

            time_code = i  # Frame number

            # Track validity span
            if first_valid_frame is None:
                first_valid_frame = i
            last_valid_frame = i

            # Position
            translate_op.Set(Gf.Vec3d(state.center_x, state.center_y, state.center_z), time_code)

            # Rotation (Waymo Heading is Radians -> USD Degrees)
            degrees = math.degrees(state.heading)
            rotate_op.Set(degrees, time_code)

            # Velocity (critical for TTC computation: v_c(t) = closing speed)
            # Store per-frame velocity from state.velocity_x/y
            if hasattr(state, 'velocity_x') and hasattr(state, 'velocity_y'):
                vel_x_attr.Set(float(state.velocity_x), time_code)
                vel_y_attr.Set(float(state.velocity_y), time_code)

            # Per-frame bounding box dimensions (if available and varying)
            if state_has_dimensions:
                if dimensions_vary:
                    # Time-sample dimensions since they change
                    extent_l_attr.Set(float(state.length), time_code)
                    extent_w_attr.Set(float(state.width), time_code)
                    extent_h_attr.Set(float(state.height), time_code)

        # Set static extent values if dimensions don't vary (or use track-level defaults)
        if state_has_dimensions and not dimensions_vary:
            # Use first valid state's dimensions as static value
            for s in track.states:
                if s.valid:
                    extent_l_attr.Set(float(s.length))
                    extent_w_attr.Set(float(s.width))
                    extent_h_attr.Set(float(s.height))
                    break
        elif not state_has_dimensions:
            # Fall back to track-level dimensions
            extent_l_attr.Set(float(l))
            extent_w_attr.Set(float(w))
            extent_h_attr.Set(float(h))

        # Set track validity span metadata
        if first_valid_frame is not None:
            prim.GetAttribute("waymo:firstValidFrame").Set(first_valid_frame)
            prim.GetAttribute("waymo:lastValidFrame").Set(last_valid_frame)

    # Save
    stage.GetRootLayer().Save()
    print(f"✅ Saved: {output_path}")
    return stage


def add_scenario_summary(
    stage: Usd.Stage,
    scenario,
    analysis_results: Optional[List[Dict[str, Any]]] = None,
    typology: Optional[str] = None,
    baseline_collision: Optional[bool] = None,
    oracle_singleton_solutions: Optional[List[str]] = None,
):
    """
    Add scenario summary to /World customData["scenario"].

    Following the RLxUSD pattern of storing episode configuration in customData,
    this function stores AV scenario metadata for self-describing USD files.

    Required fields (always populated from scenario):
    - avxusd_version: str = "0.1"
    - scenario_id: str
    - sdc_track_id: int (ego vehicle ID from scenario.sdc_track_index)
    - num_agents: int (total number of tracks)
    - duration_frames: int
    - duration_seconds: float
    - frames_per_second: int = 10

    Optional fields (populated from analysis_results or explicit args):
    - typology: str (Intersection, Highway_Merge, etc.)
    - min_distance_m: float
    - min_ttc_s: float
    - closest_obj_id: str
    - closest_obj_type: str
    - baseline_collision: bool
    - oracle_singleton_solutions: list[str]

    Args:
        stage: The USD stage to annotate
        scenario: Waymo scenario protobuf object
        analysis_results: Optional list of dicts from CollisionValidator.analyze_scenario()
                         Used to extract min_distance, min_ttc, closest object info
        typology: Optional scenario typology classification
        baseline_collision: Optional bool indicating if baseline has collision
        oracle_singleton_solutions: Optional list of agent IDs that solve the scenario
    """
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        print("Error: /World prim not found")
        return

    # Calculate duration
    num_frames = int(stage.GetEndTimeCode() - stage.GetStartTimeCode() + 1)
    fps = int(stage.GetFramesPerSecond()) if stage.GetFramesPerSecond() else 10

    # Use timestamps if available for accurate duration
    if scenario.timestamps_seconds:
        duration_seconds = float(scenario.timestamps_seconds[-1] - scenario.timestamps_seconds[0])
    else:
        duration_seconds = num_frames / fps

    # Get SDC (ego) track ID
    sdc_track = scenario.tracks[scenario.sdc_track_index]
    sdc_track_id = int(sdc_track.id)

    # Count agents (excluding ego)
    num_agents = len(scenario.tracks)

    # Build the scenario summary dictionary
    # Note: USD customData values must be basic types (str, int, float, bool)
    # or VtArray for lists
    summary = {
        "avxusd_version": "0.1",
        "scenario_id": str(scenario.scenario_id),
        "sdc_track_id": sdc_track_id,
        "num_agents": num_agents,
        "duration_frames": num_frames,
        "duration_seconds": duration_seconds,
        "frames_per_second": fps,
    }

    # Add optional typology
    if typology is not None:
        summary["typology"] = str(typology)

    # Add optional baseline collision flag
    if baseline_collision is not None:
        summary["baseline_collision"] = bool(baseline_collision)

    # Add optional oracle singleton solutions
    if oracle_singleton_solutions is not None:
        # Store as comma-separated string since customData doesn't support nested arrays well
        summary["oracle_singleton_solutions"] = ",".join(str(s) for s in oracle_singleton_solutions)

    # Extract metrics from analysis results if provided
    if analysis_results and len(analysis_results) > 0:
        # Results are sorted by criticality, so first entry is most critical
        closest = analysis_results[0]

        if closest.get('min_distance') is not None:
            summary["min_distance_m"] = float(closest['min_distance'])

        if closest.get('min_ttc') is not None:
            summary["min_ttc_s"] = float(closest['min_ttc'])

        if closest.get('agent_id') is not None:
            summary["closest_obj_id"] = str(closest['agent_id'])

        if closest.get('object_type') is not None:
            summary["closest_obj_type"] = str(closest['object_type'])

    # Get existing customData and update it
    existing_custom_data = world_prim.GetCustomData()

    # Convert to a regular dict if needed, then update
    custom_data_dict = dict(existing_custom_data) if existing_custom_data else {}
    custom_data_dict["scenario"] = summary

    # Set the updated customData
    world_prim.SetCustomData(custom_data_dict)

    print(f"Added scenario summary to /World customData['scenario']")


def load_scenario_summary(stage: Usd.Stage) -> Optional[Dict[str, Any]]:
    """
    Load scenario summary from /World customData["scenario"].

    Returns:
        Dictionary with scenario metadata, or None if not found.
    """
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        return None

    custom_data = world_prim.GetCustomData()
    if not custom_data:
        return None

    scenario_data = custom_data.get("scenario")
    if scenario_data:
        # Convert VtDictionary to regular dict
        return dict(scenario_data)
    return None


# --- RUNNER ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert Waymo scenarios to USD format",
        epilog="Example: python waymo_to_usd.py --extracted data/extracted_scenarios"
    )
    parser.add_argument("--extracted", required=True,
                        help="Directory with extracted .pb files")
    parser.add_argument("--csv", default="scenarios.csv",
                        help="Path to scenarios.csv (must contain scenario_id column)")
    parser.add_argument("--out_dir", default="scenarios",
                        help="Output directory for USD files")
    parser.add_argument("--suffix", default="_base.usd",
                        help="Output filename suffix")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite USDs if they already exist")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load target scenario_ids from CSV
    with open(args.csv, newline="") as f:
        r = csv.DictReader(f)
        target_ids = [row["scenario_id"].strip() for row in r if row.get("scenario_id")]
    remaining = set(target_ids)

    print(f"Target scenarios: {len(target_ids)}")

    converted = 0
    skipped_existing = 0

    print(f"📂 Using extracted scenarios from: {args.extracted}")
    pb_files = sorted(glob.glob(os.path.join(args.extracted, "*.pb")))
    print(f"Found {len(pb_files)} .pb files")

    for pb_path in pb_files:
        sid = os.path.basename(pb_path).replace(".pb", "")

        if sid not in remaining:
            continue

        out_path = os.path.join(args.out_dir, f"{sid}{args.suffix}")

        if (not args.overwrite) and os.path.exists(out_path):
            print(f"Exists, skipping: {out_path}")
            skipped_existing += 1
            remaining.remove(sid)
            continue

        # Load scenario from .pb file
        with open(pb_path, 'rb') as f:
            s = scenario_pb2.Scenario()
            s.ParseFromString(f.read())

        waymo_to_usd(s, out_path)
        converted += 1
        remaining.remove(sid)

    print("\n" + "=" * 60)
    print(f"Converted: {converted}")
    print(f"Skipped (already existed): {skipped_existing}")
    print(f"Missing: {len(remaining)}")
    if remaining:
        print(f"Missing IDs: {list(remaining)[:5]}{'...' if len(remaining) > 5 else ''}")
    print("=" * 60)



