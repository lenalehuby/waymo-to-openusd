"""
select_safety_critical_scenarios.py - ROBUST QUOTA VERSION
Scans all available TFRecords until quotas are met.
Skips corrupt files automatically.
"""

import tensorflow as tf
import os
import numpy as np
from waymo_open_dataset.protos import scenario_pb2
import csv
from collections import defaultdict
import glob
import sys

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CONFIGURATION
TARGET_PER_TYPOLOGY = 5
TYPOLOGIES = ["Intersection", "Highway_Merge", "Urban_Cut_in", "Pedestrian", "Occluded_Turn"]

# Distance threshold for "near-miss" detection (meters)
# Note: This is NOT collision detection - actual collisions would be < 0.5m
NEAR_MISS_THRESHOLD_M = 2.5

def compute_ttc(ego_state, obj_state, ego_next=None, obj_next=None):
    dx = obj_state.center_x - ego_state.center_x
    dy = obj_state.center_y - ego_state.center_y
    distance = np.sqrt(dx**2 + dy**2)
    
    if ego_next and obj_next and distance > 0:
        dt = 0.1
        ego_vx = (ego_next.center_x - ego_state.center_x) / dt
        ego_vy = (ego_next.center_y - ego_state.center_y) / dt
        obj_vx = (obj_next.center_x - obj_state.center_x) / dt
        obj_vy = (obj_next.center_y - obj_state.center_y) / dt
        
        rel_vx = obj_vx - ego_vx
        rel_vy = obj_vy - ego_vy
        
        closing_speed = -(dx * rel_vx + dy * rel_vy) / distance
        
        if closing_speed > 0.1:
            ttc = distance / closing_speed
            return ttc if ttc > 0 else float('inf')
    return float('inf')

def classify_typology(scenario, ego_track, avg_speed, closest_obj_type, min_distance):
    # 1. PEDESTRIAN - Check if closest object is a vulnerable road user
    if closest_obj_type in ["TYPE_PEDESTRIAN", "TYPE_CYCLIST"] and min_distance < 6.0:
        return "Pedestrian"
    
    # Check map features for context clues
    has_stop_sign = False
    has_crosswalk = False
    has_lane_merge = False

    for map_feature in scenario.map_features:
        if map_feature.HasField('stop_sign'): has_stop_sign = True
        elif map_feature.HasField('crosswalk'): has_crosswalk = True
        elif map_feature.HasField('lane'):
            # Multiple entry lanes suggests a merge point
            if len(map_feature.lane.entry_lanes) > 1: has_lane_merge = True

    # 2. INTERSECTION - Stop signs and crosswalks are strong intersection indicators
    if has_stop_sign or has_crosswalk:
        return "Intersection"

    # 3. HIGH SPEED scenarios (>12 m/s ≈ 43 km/h)
    if avg_speed > 12.0:
        if has_lane_merge:
            return "Highway_Merge"
        else:
            # High speed without merge - assume limited visibility scenario
            return "Occluded_Turn"

    # 4. LOW SPEED scenarios
    else:
        if has_lane_merge:
            return "Urban_Cut_in"
        else:
            # Default: low speed urban driving near intersections
            return "Intersection"

def analyze_scenario(scenario):
    # Ego Detection
    ego_track = None
    ego_id = None
    
    if hasattr(scenario, 'sdc_track_index') and scenario.sdc_track_index >= 0:
        try:
            if scenario.sdc_track_index < len(scenario.tracks):
                ego_track = scenario.tracks[scenario.sdc_track_index]
                ego_id = ego_track.id
        except (IndexError, AttributeError):
            pass
    
    if not ego_track:
        for track in scenario.tracks:
            if track.object_type == 1:
                ego_track = track; ego_id = track.id; break
                
    if not ego_track: return None
    
    # Metrics
    min_ttc = float('inf')
    min_distance = float('inf')
    near_miss_frames = 0  # Frames where any object is within NEAR_MISS_THRESHOLD_M
    closest_obj_id = None
    closest_obj_type = None
    critical_time = 0
    # Consistent with waymo_to_usd.py WAYMO_OBJECT_TYPES
    TYPE_MAP = {0: "TYPE_UNSET", 1: "TYPE_VEHICLE", 2: "TYPE_PEDESTRIAN", 3: "TYPE_CYCLIST", 4: "TYPE_OTHER"}
    
    # Speed
    speeds = []
    for i in range(len(scenario.timestamps_seconds) - 1):
        if i < len(ego_track.states) and i+1 < len(ego_track.states):
            if ego_track.states[i].valid and ego_track.states[i+1].valid:
                dx = ego_track.states[i+1].center_x - ego_track.states[i].center_x
                dy = ego_track.states[i+1].center_y - ego_track.states[i].center_y
                dt = scenario.timestamps_seconds[i+1] - scenario.timestamps_seconds[i]
                if dt > 0: speeds.append(np.sqrt(dx**2 + dy**2) / dt)
    avg_speed = np.mean(speeds) if speeds else 0.0
    
    # Frame Loop
    valid_timesteps = 0
    for t_idx in range(len(scenario.timestamps_seconds)):
        if t_idx >= len(ego_track.states) or not ego_track.states[t_idx].valid: continue
        valid_timesteps += 1
        
        ego_state = ego_track.states[t_idx]
        ego_next = ego_track.states[t_idx+1] if t_idx+1 < len(ego_track.states) and ego_track.states[t_idx+1].valid else None
        
        any_close = False
        for track in scenario.tracks:
            if track.id == ego_id or t_idx >= len(track.states) or not track.states[t_idx].valid: continue
            
            obj_state = track.states[t_idx]
            obj_next = track.states[t_idx+1] if t_idx+1 < len(track.states) and track.states[t_idx+1].valid else None
            
            dist = np.sqrt((ego_state.center_x - obj_state.center_x)**2 + (ego_state.center_y - obj_state.center_y)**2)
            ttc = compute_ttc(ego_state, obj_state, ego_next, obj_next)
            
            if dist < NEAR_MISS_THRESHOLD_M: any_close = True
            if dist < min_distance:
                min_distance = dist
                critical_time = scenario.timestamps_seconds[t_idx]
                closest_obj_id = track.id
                closest_obj_type = TYPE_MAP.get(track.object_type, "UNKNOWN")
            if ttc < min_ttc: min_ttc = ttc
            
        if any_close: near_miss_frames += 1

    near_miss_rate = near_miss_frames / valid_timesteps if valid_timesteps > 0 else 0.0
    typology = classify_typology(scenario, ego_track, avg_speed, closest_obj_type, min_distance)
    
    safety_score = 0
    if min_ttc < 5.0: safety_score += (5.0 - min_ttc) * 10
    if min_distance < 5.0: safety_score += (5.0 - min_distance) * 10
    safety_score += near_miss_rate * 50

    # Safe duration calculation
    duration = 0.0
    if scenario.timestamps_seconds:
        duration = scenario.timestamps_seconds[-1] - scenario.timestamps_seconds[0]

    return {
        'scenario_id': scenario.scenario_id,
        'min_ttc_s': min_ttc if min_ttc != float('inf') else None,
        'min_distance_m': min_distance if min_distance != float('inf') else None,
        'near_miss_rate': near_miss_rate,
        'critical_time_s': critical_time,
        'closest_obj_id': closest_obj_id,
        'closest_obj_type': closest_obj_type,
        'typology': typology,
        'num_objects': len(scenario.tracks) - 1,
        'duration_s': duration,
        'avg_speed_ms': avg_speed,
        'safety_score': safety_score
    }

def save_csv(scenarios, output='scenarios.csv'):
    with open(output, 'w', newline='') as f:
        fieldnames = ['scenario_id', 'typology', 'min_distance_m', 'min_ttc_s',
                     'safety_score', 'near_miss_rate', 'avg_speed_ms',
                     'num_objects', 'duration_s', 'critical_time_s',
                     'closest_obj_id', 'closest_obj_type']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scenarios)
    print(f"✅ Saved CSV to: {output}")


def save_extracted_scenarios(scenario_cache, selected_ids, output_dir='data/extracted_scenarios'):
    """
    Save selected scenario protobufs to individual .pb files for fast loading.
    This avoids re-scanning the full TFRecord dataset when converting to USD.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for sid in selected_ids:
        if sid in scenario_cache:
            out_path = os.path.join(output_dir, f"{sid}.pb")
            with open(out_path, 'wb') as f:
                f.write(scenario_cache[sid])
            saved += 1
    print(f"✅ Extracted {saved} scenario protobufs to: {output_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Scan Waymo Open Motion Dataset TFRecords and select safety-critical scenarios",
        epilog="Example: python select_safety_critical_scenarios.py --tfrecord-dir data/raw_tfrecords")
    parser.add_argument("--tfrecord-dir", default="data/raw_tfrecords",
                        help="Directory containing training_20s.tfrecord-*-of-* shards")
    parser.add_argument("--csv", default="scenarios.csv",
                        help="Output CSV listing the selected scenarios")
    parser.add_argument("--extracted-dir", default="data/extracted_scenarios",
                        help="Output directory for the selected scenario .pb files")
    args = parser.parse_args()

    TFRECORD_DIR = args.tfrecord_dir
    tfrecord_pattern = os.path.join(TFRECORD_DIR, "training_20s.tfrecord-*-of-*")
    all_files = sorted(glob.glob(tfrecord_pattern))
    
    if not all_files:
        print(f"❌ No TFRecord files found at {tfrecord_pattern}")
        exit(1)
    
    print(f"Found {len(all_files)} TFRecord files.")
    print(f"Goal: {TARGET_PER_TYPOLOGY} scenarios per typology.")
    print("Strategy: Full scan, skipping corrupt files.")
    print("="*100)
    
    all_candidates = defaultdict(list)
    seen_scenario_ids = set()  # Track already-added scenarios to avoid duplicates
    scenario_cache = {}  # Cache raw protobuf bytes for extraction
    quotas_filled = {t: False for t in TYPOLOGIES}

    for i, tfrecord_path in enumerate(all_files):
        if all(quotas_filled.values()):
            print("\n✅ All quotas filled! Stopping scan early.")
            break
            
        print(f"\n📂 [{i+1}/{len(all_files)}] Scanning: {os.path.basename(tfrecord_path)}")
        
        try:
            raw_dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
            
            # Use iterator to handle individual bad records
            for j, data in enumerate(raw_dataset):
                try:
                    raw_bytes = data.numpy()
                    scenario = scenario_pb2.Scenario()
                    scenario.ParseFromString(raw_bytes)
                    res = analyze_scenario(scenario)

                    if res and res['min_distance_m'] is not None and res['min_distance_m'] < 5.0:
                        sid = res['scenario_id']
                        typ = res['typology']
                        # Skip if we've already added this scenario (avoid duplicates)
                        if sid in seen_scenario_ids:
                            continue
                        if len(all_candidates[typ]) < TARGET_PER_TYPOLOGY + 10:
                            all_candidates[typ].append(res)
                            seen_scenario_ids.add(sid)
                            # Cache raw bytes for later extraction
                            scenario_cache[sid] = raw_bytes
                            # print(f"  + Found {typ}")
                except Exception:
                    # Skip individual corrupt/malformed records without stopping the scan
                    # This is intentional - Waymo TFRecords occasionally have bad entries
                    continue
                
                # Check status periodically
                if j % 50 == 0:
                    for t in TYPOLOGIES:
                        if len(all_candidates[t]) >= TARGET_PER_TYPOLOGY:
                            quotas_filled[t] = True
                    if all(quotas_filled.values()): break
                    
        except Exception as e:
            print(f"⚠️  Skipping corrupt file {os.path.basename(tfrecord_path)}: {str(e)}")
            continue

        status_str = " | ".join([f"{t}: {len(all_candidates[t])}/{TARGET_PER_TYPOLOGY}" for t in TYPOLOGIES])
        print(f"   Status: {status_str}")

    print("\n" + "="*100)
    print("FINAL SELECTION")
    print("="*100)
    
    selected_final = []
    for typ in TYPOLOGIES:
        candidates = all_candidates[typ]
        if candidates:
            candidates.sort(key=lambda x: x['safety_score'], reverse=True)
            selected = candidates[:TARGET_PER_TYPOLOGY]
            selected_final.extend(selected)
            print(f"{typ:<15}: {len(selected)}/{TARGET_PER_TYPOLOGY} selected")
        else:
            print(f"{typ:<15}: 0/{TARGET_PER_TYPOLOGY} selected (None found)")

    if len(selected_final) > 0:
        save_csv(selected_final, args.csv)
        # Extract scenario protobufs for fast USD conversion
        selected_ids = [s['scenario_id'] for s in selected_final]
        save_extracted_scenarios(scenario_cache, selected_ids, args.extracted_dir)


