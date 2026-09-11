"""
convert_synthetic_to_avxusd.py - Convert baked synthetic scenarios to AVXUSD format

Converts Omniverse-baked USD files to match the AVXUSD schema:
1. Restructure hierarchy to /World/Agents/Agent_*
2. Add waymo: namespace attributes
3. Compute velocities from position deltas
4. Add Geometry child prims with bounding boxes
5. Add customData["scenario"] metadata

Usage:
    python convert_synthetic_to_avxusd.py --input synthetic_01_baked.usd --output synthetic_01_base.usd
    python convert_synthetic_to_avxusd.py --batch --input-dir scenarios/synthetic_scenarios
"""

import os
import math
import argparse
import glob
from datetime import datetime
from pxr import Usd, UsdGeom, Sdf, Gf, Vt


# Mapping from prim names to agent info
# Customize based on your synthetic scenario naming
AGENT_MAPPING = {
    # Scenario 01: Pedestrian crossing
    'Ego': {'id': 'EGO', 'type': 1, 'type_str': 'TYPE_VEHICLE', 'extent': (4.5, 2.0, 1.5)},
    'Ego_Vehicle': {'id': 'EGO', 'type': 1, 'type_str': 'TYPE_VEHICLE', 'extent': (4.5, 2.0, 1.5)},
    'Pedestrian': {'id': '001', 'type': 2, 'type_str': 'TYPE_PEDESTRIAN', 'extent': (0.5, 0.5, 1.8)},

    # Scenario 02: Rear-end
    'Lead_Vehicle': {'id': '002', 'type': 1, 'type_str': 'TYPE_VEHICLE', 'extent': (4.5, 2.0, 1.5)},

    # Scenario 03: Cut-in
    'Other_Car': {'id': '003', 'type': 1, 'type_str': 'TYPE_VEHICLE', 'extent': (4.5, 2.0, 1.5)},

    # Scenario 04: Intersection
    'Car_2': {'id': '004', 'type': 1, 'type_str': 'TYPE_VEHICLE', 'extent': (4.5, 2.0, 1.5)},

    # Scenario 05: Occluded pedestrian
    'Parked_car': {'id': '005', 'type': 1, 'type_str': 'TYPE_VEHICLE', 'extent': (4.5, 2.0, 1.5)},
    'Parked_Car': {'id': '005', 'type': 1, 'type_str': 'TYPE_VEHICLE', 'extent': (4.5, 2.0, 1.5)},

    # Generic fallbacks
    'Vehicle': {'id': '010', 'type': 1, 'type_str': 'TYPE_VEHICLE', 'extent': (4.5, 2.0, 1.5)},
    'Cyclist': {'id': '011', 'type': 3, 'type_str': 'TYPE_CYCLIST', 'extent': (1.8, 0.6, 1.5)},
}

# Typology mapping based on filename keywords
TYPOLOGY_KEYWORD_MAP = {
    'pedestrian': 'Pedestrian',
    'rearend': 'Highway_Merge',
    'cutin': 'Urban_Cut_in',
    'intersection': 'Intersection',
    'occluded': 'Occluded_Turn',
}

# Explicit per-scenario-number mapping (baked filenames lack descriptive keywords)
TYPOLOGY_BY_NUMBER = {
    '01': 'Pedestrian',
    '02': 'Highway_Merge',
    '03': 'Urban_Cut_in',
    '04': 'Intersection',
    '05': 'Occluded_Turn',
}


def infer_typology(filename: str) -> str:
    """Infer typology from filename keywords or scenario number."""
    filename_lower = filename.lower()
    # Try keyword match first (works for original descriptive filenames)
    for key, typology in TYPOLOGY_KEYWORD_MAP.items():
        if key in filename_lower:
            return typology
    # Fallback: extract scenario number from synthetic_XX pattern
    import re
    m = re.search(r'synthetic_(\d+)', filename_lower)
    if m:
        return TYPOLOGY_BY_NUMBER.get(m.group(1), 'Synthetic')
    return 'Synthetic'


def get_agent_info(prim_name: str, used_ids: set) -> dict:
    """Get agent info from prim name, ensuring unique IDs."""
    if prim_name in AGENT_MAPPING:
        info = AGENT_MAPPING[prim_name].copy()
    else:
        # Generate unique ID for unknown prims
        base_id = 100
        while str(base_id).zfill(3) in used_ids:
            base_id += 1
        info = {
            'id': str(base_id).zfill(3),
            'type': 1,
            'type_str': 'TYPE_VEHICLE',
            'extent': (4.5, 2.0, 1.5)
        }

    # Ensure unique ID (skip uniqueness check for EGO)
    if info['id'] != 'EGO' and info['id'] in used_ids:
        base = int(info['id']) if info['id'].isdigit() else 100
        while str(base).zfill(3) in used_ids:
            base += 1
        info['id'] = str(base).zfill(3)

    used_ids.add(info['id'])
    return info


def compute_velocities(positions: list, dt: float = 0.1) -> tuple:
    """
    Compute velocities from position time series using central differences.

    Args:
        positions: List of (x, y, z) tuples in meters.
        dt: Time step in seconds.

    Returns:
        (vel_x, vel_y) as lists of float.
    """
    n = len(positions)
    if n < 2:
        return [0.0] * n, [0.0] * n

    vel_x = []
    vel_y = []

    for i in range(n):
        if i == 0:
            # Forward difference for first frame
            vx = (positions[1][0] - positions[0][0]) / dt
            vy = (positions[1][1] - positions[0][1]) / dt
        elif i == n - 1:
            # Backward difference for last frame
            vx = (positions[-1][0] - positions[-2][0]) / dt
            vy = (positions[-1][1] - positions[-2][1]) / dt
        else:
            # Central difference
            vx = (positions[i + 1][0] - positions[i - 1][0]) / (2 * dt)
            vy = (positions[i + 1][1] - positions[i - 1][1]) / (2 * dt)

        vel_x.append(vx)
        vel_y.append(vy)

    return vel_x, vel_y


def compute_heading_degrees(vx: float, vy: float) -> float:
    """Compute heading in degrees (CCW from X-axis) from velocity components."""
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(vy, vx))


def detect_axis_remap(stage, animated_prims, start_frame, end_frame, meters_per_unit):
    """
    Auto-detect if the baked file has mislabeled upAxis.

    Omniverse baking sometimes declares Z-up but keeps Y-up coordinates
    (forward motion along Z instead of XY). This function detects that case.

    Returns 'yup_in_zup' if coordinates need remapping, 'none' otherwise.
    """
    up_axis = UsdGeom.GetStageUpAxis(stage)
    if up_axis != UsdGeom.Tokens.z:
        return 'none'  # Explicit Y-up handled by caller

    # Sample the first animated prim's displacement in each axis
    for prim in animated_prims:
        xformable = UsdGeom.Xformable(prim)
        translate_ops = [op for op in xformable.GetOrderedXformOps()
                         if 'translate' in op.GetOpName().lower()]
        if not translate_ops:
            continue

        p0 = translate_ops[0].Get(Usd.TimeCode(start_frame))
        pe = translate_ops[0].Get(Usd.TimeCode(end_frame))
        if p0 is None or pe is None:
            continue

        dx = abs(pe[0] - p0[0]) * meters_per_unit
        dy = abs(pe[1] - p0[1]) * meters_per_unit
        dz = abs(pe[2] - p0[2]) * meters_per_unit
        xy_disp = math.sqrt(dx ** 2 + dy ** 2)

        # If Z displacement dominates XY by >2x, coordinates are Y-up layout
        if dz > 1.0 and dz > xy_disp * 2:
            return 'yup_in_zup'

    return 'none'


def extract_positions_from_prim(prim, start_frame: int, end_frame: int,
                                meters_per_unit: float, remap_mode: str) -> list:
    """
    Extract world-space positions from a prim across all frames.

    Handles coordinate conversion:
    - Scales from scene units to meters using metersPerUnit
    - remap_mode:
        'none'        - no axis remap needed
        'yup'         - explicit Y-up: (X,Y,Z)_yup -> (X, Z, Y)_zup
        'yup_in_zup'  - declares Z-up but has Y-up layout:
                         (X,Y,Z) -> (X, -Z, Y)  (forward=-Z -> +Y north)

    Returns list of (x, y, z) in meters, Z-up coordinate system.
    """
    xformable = UsdGeom.Xformable(prim)
    positions = []

    for frame in range(start_frame, end_frame + 1):
        time = Usd.TimeCode(frame)

        # Try ordered xform ops first (translate)
        translate_ops = [op for op in xformable.GetOrderedXformOps()
                         if 'translate' in op.GetOpName().lower()]

        if translate_ops:
            pos = translate_ops[0].Get(time)
            if pos is not None:
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            else:
                world_xform = xformable.ComputeLocalToWorldTransform(time)
                trans = world_xform.ExtractTranslation()
                x, y, z = float(trans[0]), float(trans[1]), float(trans[2])
        else:
            world_xform = xformable.ComputeLocalToWorldTransform(time)
            trans = world_xform.ExtractTranslation()
            x, y, z = float(trans[0]), float(trans[1]), float(trans[2])

        # Convert to meters
        x *= meters_per_unit
        y *= meters_per_unit
        z *= meters_per_unit

        # Apply axis remap
        if remap_mode == 'yup':
            # Explicit Y-up: (X, Y_up, Z_fwd) -> (X, Z_fwd, Y_up)
            x, y, z = x, z, y
        elif remap_mode == 'yup_in_zup':
            # Mislabeled Z-up with Y-up layout:
            # Omniverse: X=right, Y=up, -Z=forward
            # Waymo Z-up: X=east, Y=north, Z=up
            # Remap: (X, Y_up, Z_fwd) -> (X, -Z_fwd=north, Y_up=up)
            x, y, z = x, -z, y

        positions.append((x, y, z))

    return positions


def find_animated_prims(world_prim):
    """
    Find all agent prims under /World that have animation data.
    Skips environment, lighting, ground, and OmniGraph prims.
    """
    skip_names = {'Ground', 'Environment', 'Light', 'Camera', 'PushGraph',
                  'Looks', 'ground', 'groundCollider', 'Sky', 'DistantLight'}
    animated = []

    for child in world_prim.GetChildren():
        name = child.GetName()
        if name in skip_names:
            continue
        # Accept Xform and Mesh types (OmniGraph baked may produce either)
        if child.IsA(UsdGeom.Xformable):
            animated.append(child)

    return animated


def convert_synthetic_to_avxusd(
    input_path: str,
    output_path: str,
    fps: float = 10.0,
    verbose: bool = True
) -> bool:
    """
    Convert a baked synthetic USD to AVXUSD format.

    Follows the patterns established in waymo_to_usd.py:
    - Agent Xform with xformOp:translate + xformOp:rotateZ
    - Geometry as UsdGeom.Cube with scale and displayColor
    - waymo: namespace attributes for identity, kinematics, extent, validity

    Returns True if successful.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Converting: {os.path.basename(input_path)}")
        print(f"{'=' * 60}")

    # Open source stage
    src_stage = Usd.Stage.Open(input_path)
    if not src_stage:
        print(f"ERROR: Could not open {input_path}")
        return False

    start_frame = int(src_stage.GetStartTimeCode())
    end_frame = int(src_stage.GetEndTimeCode())
    num_frames = end_frame - start_frame + 1
    dt = 1.0 / fps

    # Detect scene units and up-axis
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(src_stage)
    up_axis = UsdGeom.GetStageUpAxis(src_stage)

    if verbose:
        print(f"  Frames: {start_frame} to {end_frame} ({num_frames} total)")
        print(f"  Source FPS: {src_stage.GetFramesPerSecond()}")
        print(f"  metersPerUnit: {meters_per_unit}")
        print(f"  upAxis: {up_axis}")

    # Find all animated prims under /World
    world_prim = src_stage.GetPrimAtPath("/World")
    if not world_prim:
        print("ERROR: /World prim not found")
        return False

    animated_prims = find_animated_prims(world_prim)

    if not animated_prims:
        print("WARNING: No agent prims found under /World.")
        print("  Baked files may be empty. Re-bake in Omniverse with:")
        print("  Window > Animation > Bake Animation (select all agent prims)")
        return False

    if verbose:
        print(f"  Found {len(animated_prims)} agent prims:")
        for p in animated_prims:
            print(f"    - {p.GetName()} [{p.GetTypeName()}]")

    # Determine axis remap mode
    if up_axis == UsdGeom.Tokens.y:
        remap_mode = 'yup'
    else:
        # Auto-detect mislabeled Z-up (Omniverse bake artifact)
        remap_mode = detect_axis_remap(
            src_stage, animated_prims, start_frame, end_frame, meters_per_unit
        )

    if verbose:
        print(f"  Axis remap: {remap_mode}")

    # Extract positions for each prim
    prim_data = {}
    for prim in animated_prims:
        positions = extract_positions_from_prim(
            prim, start_frame, end_frame, meters_per_unit, remap_mode
        )
        prim_data[prim.GetName()] = positions

    # Create output stage (matching waymo_to_usd.py conventions)
    out_stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(out_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(out_stage, 1.0)
    out_stage.SetStartTimeCode(0)
    out_stage.SetEndTimeCode(num_frames - 1)
    out_stage.SetFramesPerSecond(fps)

    # Create /World
    world_xform = UsdGeom.Xform.Define(out_stage, "/World")
    world = world_xform.GetPrim()

    # Generate timestamps array
    timestamps = [i * dt for i in range(num_frames)]
    ts_attr = world.CreateAttribute("waymo:timestampsSeconds", Sdf.ValueTypeNames.FloatArray)
    ts_attr.Set(Vt.FloatArray(timestamps))

    # Current time index (start of prediction window - 0 for synthetic)
    world.CreateAttribute("waymo:currentTimeIndex", Sdf.ValueTypeNames.Int).Set(0)

    # Create /World/Agents container
    UsdGeom.Xform.Define(out_stage, "/World/Agents")

    # Process each agent
    used_ids = set()
    agent_infos = []

    for prim_name, positions in prim_data.items():
        agent_info = get_agent_info(prim_name, used_ids)
        agent_id = agent_info['id']
        is_ego = (agent_id == 'EGO')

        if verbose:
            print(f"  Processing {prim_name} -> Agent_{agent_id} ({agent_info['type_str']})")

        # --- Create agent prim (matching waymo_to_usd.py:create_staged_agent) ---
        agent_path = f"/World/Agents/Agent_{agent_id}"
        xform = UsdGeom.Xform.Define(out_stage, agent_path)
        agent_prim = xform.GetPrim()

        # Core identity attributes
        agent_prim.CreateAttribute("waymo:objectType", Sdf.ValueTypeNames.Int).Set(
            agent_info['type']
        )
        agent_prim.CreateAttribute("waymo:objectTypeString", Sdf.ValueTypeNames.String).Set(
            agent_info['type_str']
        )

        # Velocity attributes (populated below with time samples)
        vel_x_attr = agent_prim.CreateAttribute("waymo:velocityX", Sdf.ValueTypeNames.Float)
        vel_y_attr = agent_prim.CreateAttribute("waymo:velocityY", Sdf.ValueTypeNames.Float)

        # Extent attributes (static for synthetic)
        extent = agent_info['extent']  # (length, width, height)
        agent_prim.CreateAttribute("waymo:extentLength", Sdf.ValueTypeNames.Float).Set(
            float(extent[0])
        )
        agent_prim.CreateAttribute("waymo:extentWidth", Sdf.ValueTypeNames.Float).Set(
            float(extent[1])
        )
        agent_prim.CreateAttribute("waymo:extentHeight", Sdf.ValueTypeNames.Float).Set(
            float(extent[2])
        )

        # Track validity span
        agent_prim.CreateAttribute("waymo:firstValidFrame", Sdf.ValueTypeNames.Int).Set(0)
        agent_prim.CreateAttribute("waymo:lastValidFrame", Sdf.ValueTypeNames.Int).Set(
            num_frames - 1
        )

        # --- Transform operations on Agent Xform (T * R order, matching waymo_to_usd.py) ---
        translate_op = xform.AddTranslateOp()
        rotate_op = xform.AddRotateZOp()

        # Compute velocities from position deltas
        vel_x, vel_y = compute_velocities(positions, dt)

        # Write time-sampled transforms and velocities
        for i in range(num_frames):
            time_code = Usd.TimeCode(i)
            pos = positions[i]

            # Position
            translate_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]), time_code)

            # Heading from velocity (degrees, CCW from X-axis)
            heading = compute_heading_degrees(vel_x[i], vel_y[i])
            rotate_op.Set(heading, time_code)

            # Velocity
            vel_x_attr.Set(float(vel_x[i]), time_code)
            vel_y_attr.Set(float(vel_y[i]), time_code)

        # --- Create Geometry child (matching waymo_to_usd.py:create_staged_agent) ---
        geom = UsdGeom.Cube.Define(out_stage, f"{agent_path}/Geometry")

        # Scale cube to match agent dimensions (USD Cube is 2x2x2, scale by half)
        geom.AddScaleOp().Set(Gf.Vec3f(extent[0] / 2, extent[1] / 2, extent[2] / 2))

        # Display color: blue for EGO, red for others
        color = Gf.Vec3f(0, 0, 1) if is_ego else Gf.Vec3f(1, 0, 0)
        geom.GetDisplayColorAttr().Set([color])

        agent_infos.append({
            'id': agent_id,
            'type': agent_info['type_str'],
            'original_name': prim_name,
        })

    # --- Add customData["scenario"] (matching waymo_to_usd.py:add_scenario_summary) ---
    scenario_id = os.path.basename(output_path).replace('_base.usd', '').replace('.usd', '')
    typology = infer_typology(os.path.basename(input_path))

    scenario_metadata = {
        'avxusd_version': '0.1',
        'scenario_id': scenario_id,
        'sdc_track_id': 'EGO',
        'num_agents': len(agent_infos),
        'duration_frames': num_frames,
        'duration_seconds': float(num_frames * dt),
        'frames_per_second': int(fps),
        'typology': typology,
        'source': 'synthetic',
        'created_at': datetime.now().isoformat(),
    }

    world.SetCustomDataByKey('scenario', scenario_metadata)

    # Save
    out_stage.GetRootLayer().Save()

    if verbose:
        print(f"\n  Saved: {output_path}")
        print(f"     Agents: {len(agent_infos)}")
        print(f"     Frames: {num_frames}")
        print(f"     Typology: {typology}")

    return True


def batch_convert(input_dir: str, output_dir: str = None, verbose: bool = True):
    """Convert all *_baked.usd files in directory."""
    if output_dir is None:
        output_dir = input_dir

    baked_files = sorted(glob.glob(os.path.join(input_dir, "*_baked.usd")))

    if not baked_files:
        print(f"No *_baked.usd files found in {input_dir}")
        return

    print(f"Found {len(baked_files)} baked files to convert")

    success = 0
    for baked_path in baked_files:
        # synthetic_01_baked.usd -> synthetic_01_base.usd
        basename = os.path.basename(baked_path)
        output_name = basename.replace('_baked.usd', '_base.usd')
        output_path = os.path.join(output_dir, output_name)

        if convert_synthetic_to_avxusd(baked_path, output_path, verbose=verbose):
            success += 1

    print(f"\n{'=' * 60}")
    print(f"BATCH CONVERSION COMPLETE: {success}/{len(baked_files)} succeeded")
    print(f"{'=' * 60}")


def add_to_scenarios_csv(
    converted_dir: str,
    csv_path: str = "scenarios.csv",
    verbose: bool = True
):
    """
    Analyze converted synthetic scenarios and append to scenarios.csv.
    """
    import csv
    import sys
    import numpy as np

    # Ensure src/ is on the path for collision_check import
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from collision_check import CollisionValidator

    base_files = sorted(glob.glob(os.path.join(converted_dir, "*_base.usd")))
    base_files = [f for f in base_files if 'synthetic' in os.path.basename(f).lower()]

    if not base_files:
        print(f"No synthetic *_base.usd files found in {converted_dir}")
        return

    print(f"\nAnalyzing {len(base_files)} synthetic scenarios...")

    new_rows = []
    for usd_path in base_files:
        scenario_id = os.path.basename(usd_path).replace('_base.usd', '')

        try:
            validator = CollisionValidator(usd_path)
            end_frame = int(validator.stage.GetEndTimeCode())
            results = validator.analyze_scenario(0, end_frame)

            # Filter out EGO
            agent_results = [r for r in results if r['agent_id'] != 'EGO']

            if not agent_results:
                print(f"  {scenario_id}: No non-EGO agents found")
                continue

            # Find closest agent
            closest = min(
                agent_results,
                key=lambda r: r['min_distance'] if r['min_distance'] is not None else float('inf')
            )

            # Global minimums
            min_distance = min(
                r['min_distance'] for r in agent_results if r['min_distance'] is not None
            )
            min_ttc_values = [r['min_ttc'] for r in agent_results if r['min_ttc'] is not None]
            min_ttc = min(min_ttc_values) if min_ttc_values else None

            # Get typology from customData
            world = validator.stage.GetPrimAtPath("/World")
            scenario_data = world.GetCustomDataByKey('scenario') or {}
            typology = scenario_data.get('typology', 'Synthetic')
            duration = scenario_data.get('duration_seconds', end_frame * 0.1)

            # Compute safety score (matching existing CSV formula)
            safety_score = 0.0
            if min_ttc is not None and min_ttc < 5.0:
                safety_score += (5.0 - min_ttc) * 10
            if min_distance is not None and min_distance < 5.0:
                safety_score += (5.0 - min_distance) * 10

            # Compute average ego speed
            ego_prim = validator.stage.GetPrimAtPath("/World/Agents/Agent_EGO")
            avg_speed = 0.0
            if ego_prim:
                vx_attr = ego_prim.GetAttribute("waymo:velocityX")
                vy_attr = ego_prim.GetAttribute("waymo:velocityY")
                if vx_attr and vy_attr:
                    speeds = []
                    for t in range(0, end_frame + 1):
                        vx = vx_attr.Get(Usd.TimeCode(t))
                        vy = vy_attr.Get(Usd.TimeCode(t))
                        if vx is not None and vy is not None:
                            speeds.append(np.sqrt(float(vx) ** 2 + float(vy) ** 2))
                    avg_speed = float(np.mean(speeds)) if speeds else 0.0

            row = {
                'scenario_id': scenario_id,
                'typology': typology,
                'min_distance_m': round(min_distance, 2) if min_distance is not None else None,
                'min_ttc_s': round(min_ttc, 2) if min_ttc is not None else None,
                'safety_score': round(safety_score, 1),
                'near_miss_rate': 0.0,
                'avg_speed_ms': round(avg_speed, 2),
                'num_objects': len(agent_results),
                'duration_s': round(duration, 2),
                'critical_time_s': round(
                    closest['min_distance_frame'] * 0.1, 2
                ) if closest.get('min_distance_frame') else 0,
                'closest_obj_id': closest['agent_id'],
                'closest_obj_type': closest['object_type'],
            }

            new_rows.append(row)
            ttc_str = f", ttc={min_ttc:.2f}s" if min_ttc is not None else ""
            print(f"  {scenario_id}: dist={min_distance:.2f}m{ttc_str}")

        except Exception as e:
            print(f"  {scenario_id}: ERROR - {e}")

    if not new_rows:
        print("No scenarios to add")
        return

    # Read existing CSV
    existing_rows = []
    existing_ids = set()
    fieldnames = None

    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames)
            for row in reader:
                existing_rows.append(row)
                existing_ids.add(row.get('scenario_id', ''))

    if not fieldnames:
        fieldnames = list(new_rows[0].keys())

    # Add new rows (skip duplicates)
    added = 0
    for row in new_rows:
        if row['scenario_id'] not in existing_ids:
            existing_rows.append(row)
            added += 1
        else:
            print(f"  Skipped duplicate: {row['scenario_id']}")

    # Write updated CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)

    print(f"\nAdded {added} synthetic scenarios to {csv_path}")
    print(f"   Total scenarios: {len(existing_rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert baked synthetic scenarios to AVXUSD format"
    )
    parser.add_argument("--input", help="Single baked USD file to convert")
    parser.add_argument("--output", help="Output path for converted file")
    parser.add_argument("--batch", action="store_true",
                        help="Convert all *_baked.usd in input-dir")
    parser.add_argument("--input-dir", default="scenarios/synthetic_scenarios",
                        help="Directory containing baked USD files")
    parser.add_argument("--output-dir",
                        help="Output directory (defaults to input-dir)")
    parser.add_argument("--add-csv", action="store_true",
                        help="Add converted scenarios to scenarios.csv")
    parser.add_argument("--csv", default="scenarios.csv",
                        help="Path to scenarios.csv")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)

    args = parser.parse_args()

    if args.input:
        output = args.output or args.input.replace('_baked.usd', '_base.usd')
        convert_synthetic_to_avxusd(args.input, output, verbose=args.verbose)
    elif args.batch:
        batch_convert(args.input_dir, args.output_dir, verbose=args.verbose)

    if args.add_csv:
        converted_dir = args.output_dir or args.input_dir
        add_to_scenarios_csv(converted_dir, args.csv, verbose=args.verbose)

    if not args.input and not args.batch and not args.add_csv:
        parser.print_help()
