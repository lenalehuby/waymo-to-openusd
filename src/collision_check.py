from pxr import Usd, UsdGeom, Gf, Sdf
import numpy as np
from typing import List, Optional

# Semantic priority for object types (higher = more critical for safety)
# Used for ranking agents in root-cause analysis
OBJECT_TYPE_PRIORITY = {
    "TYPE_PEDESTRIAN": 100,  # Highest priority - vulnerable road users
    "TYPE_CYCLIST": 90,
    "TYPE_VEHICLE": 50,
    "TYPE_OTHER": 30,
    "TYPE_UNSET": 10,
}


class CollisionValidator:
    def __init__(self, stage_path):
        self.stage = Usd.Stage.Open(stage_path)
        # Create a BBoxCache. Using 'default' time initially, but we will set it per frame.
        self.cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        # Load scenario timestamps for accurate time conversion
        world_prim = self.stage.GetPrimAtPath("/World")
        ts_attr = world_prim.GetAttribute("waymo:timestampsSeconds")
        self.timestamps = list(ts_attr.Get()) if ts_attr and ts_attr.Get() else None

    def frame_to_time(self, frame):
        """Convert frame index to time in seconds using actual timestamps."""
        if self.timestamps and 0 <= frame < len(self.timestamps):
            return self.timestamps[frame]
        return frame * 0.1  # Fallback to 10 fps assumption

    def get_agent_prim(self, geometry_prim):
        """Get the parent Agent prim from a Geometry prim."""
        # /World/Agents/Agent_XXX/Geometry -> /World/Agents/Agent_XXX
        return geometry_prim.GetParent()

    def get_velocity(self, agent_prim, time):
        """
        Get velocity vector for an agent at a specific time.
        Returns (vx, vy) in m/s, or (0, 0) if not available.
        """
        # agent_prim is the Xform (e.g., /World/Agents/Agent_EGO)
        vx_attr = agent_prim.GetAttribute("waymo:velocityX")
        vy_attr = agent_prim.GetAttribute("waymo:velocityY")

        vx = vx_attr.Get(time) if vx_attr else None
        vy = vy_attr.Get(time) if vy_attr else None

        return (float(vx) if vx is not None else 0.0,
                float(vy) if vy is not None else 0.0)

    def get_object_type(self, agent_prim):
        """
        Get the Waymo object type for an agent.
        Returns (type_int, type_string, priority).
        """
        type_attr = agent_prim.GetAttribute("waymo:objectType")
        type_str_attr = agent_prim.GetAttribute("waymo:objectTypeString")

        type_int = int(type_attr.Get()) if type_attr and type_attr.Get() is not None else 0
        type_str = str(type_str_attr.Get()) if type_str_attr and type_str_attr.Get() else "TYPE_UNSET"
        priority = OBJECT_TYPE_PRIORITY.get(type_str, 10)

        return type_int, type_str, priority

    def get_validity_span(self, agent_prim):
        """
        Get the valid frame range for an agent.
        Returns (first_valid, last_valid) or (0, end_frame) if not available.
        """
        first_attr = agent_prim.GetAttribute("waymo:firstValidFrame")
        last_attr = agent_prim.GetAttribute("waymo:lastValidFrame")

        first = int(first_attr.Get()) if first_attr and first_attr.Get() is not None else 0
        last = int(last_attr.Get()) if last_attr and last_attr.Get() is not None else int(self.stage.GetEndTimeCode())

        return first, last

    def compute_ttc(self, ego_prim, agent_prim, time):
        """
        Compute Time-To-Collision between ego and agent at a specific time.

        TTC = distance / closing_speed
        closing_speed = -(dx * rel_vx + dy * rel_vy) / distance

        Returns TTC in seconds, or float('inf') if not approaching.
        """
        # Get positions (center of OBB)
        ego_center = np.mean(self.get_obb(ego_prim, time), axis=0)[:2]  # XY only
        agent_center = np.mean(self.get_obb(agent_prim, time), axis=0)[:2]

        dx = agent_center[0] - ego_center[0]
        dy = agent_center[1] - ego_center[1]
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 0.01:  # Already overlapping
            return 0.0

        # Get velocities from parent Xform prims
        ego_xform = self.get_agent_prim(ego_prim)
        agent_xform = self.get_agent_prim(agent_prim)

        ego_vx, ego_vy = self.get_velocity(ego_xform, time)
        agent_vx, agent_vy = self.get_velocity(agent_xform, time)

        # Relative velocity (agent relative to ego)
        rel_vx = agent_vx - ego_vx
        rel_vy = agent_vy - ego_vy

        # Closing speed along the line connecting centers
        # Positive = approaching, negative = separating
        closing_speed = -(dx * rel_vx + dy * rel_vy) / distance

        if closing_speed > 0.1:  # Approaching with meaningful speed
            ttc = distance / closing_speed
            return ttc if ttc > 0 else float('inf')

        return float('inf')  # Not approaching
    
    def get_obb(self, prim, time):
        """
        Extracts the Oriented Bounding Box (OBB) for a prim at a specific time.
        Returns: 8 corner points in World Space.
        """
        # Set time for the cache
        self.cache.SetTime(time)
        
        # Get World Transform
        xform = UsdGeom.Xformable(prim)
        world_transform = xform.ComputeLocalToWorldTransform(time)
        
        # Get Local Bounds (The unrotated cube dimensions)
        bound = self.cache.ComputeLocalBound(prim)
        range_min = bound.GetRange().GetMin()
        range_max = bound.GetRange().GetMax()
        
        # Calculate 8 corners of the local box
        corners = [
            Gf.Vec3d(range_min[0], range_min[1], range_min[2]),
            Gf.Vec3d(range_max[0], range_min[1], range_min[2]),
            Gf.Vec3d(range_min[0], range_max[1], range_min[2]),
            Gf.Vec3d(range_max[0], range_max[1], range_min[2]),
            Gf.Vec3d(range_min[0], range_min[1], range_max[2]),
            Gf.Vec3d(range_max[0], range_min[1], range_max[2]),
            Gf.Vec3d(range_min[0], range_max[1], range_max[2]),
            Gf.Vec3d(range_max[0], range_max[1], range_max[2]),
        ]
        
        # Transform corners to World Space
        world_corners = [world_transform.Transform(p) for p in corners]
        return world_corners

    def check_overlap(self, poly1, poly2):
        """
        Simple Separating Axis Theorem (SAT) check for 2D overlap (XY plane).
        For vehicles, 2D check is usually sufficient and faster.
        """
        # Project bottom-face points onto XY plane in sequential winding order.
        # Corners 0-3 are: (min,min), (max,min), (min,max), (max,max).
        # Reorder to CW winding: 0,1,3,2 → (min,min),(max,min),(max,max),(min,max).
        p1 = [np.array([p[0], p[1]]) for p in [poly1[0], poly1[1], poly1[3], poly1[2]]]
        p2 = [np.array([p[0], p[1]]) for p in [poly2[0], poly2[1], poly2[3], poly2[2]]]
        
        polys = [p1, p2]
        
        for poly in polys:
            for i1 in range(len(poly)):
                i2 = (i1 + 1) % len(poly)
                p1_edge = poly[i1]
                p2_edge = poly[i2]
                
                normal = np.array([-(p2_edge[1] - p1_edge[1]), p2_edge[0] - p1_edge[0]])
                # Handle zero-length edges safely
                norm_len = np.linalg.norm(normal)
                if norm_len == 0:
                    continue
                normal = normal / norm_len
                
                min1, max1 = float('inf'), float('-inf')
                for p in p1:
                    proj = np.dot(p, normal)
                    min1 = min(min1, proj)
                    max1 = max(max1, proj)
                    
                min2, max2 = float('inf'), float('-inf')
                for p in p2:
                    proj = np.dot(p, normal)
                    min2 = min(min2, proj)
                    max2 = max(max2, proj)
                    
                if max1 < min2 or max2 < min1:
                    return False # Separating axis found!
                    
        return True # No separating axis found -> Collision

    def _point_to_segment_dist(self, p, a, b):
        """Minimum distance from point p to line segment ab (2D)."""
        ab = b - a
        ab_sq = np.dot(ab, ab)
        if ab_sq < 1e-12:
            return np.linalg.norm(p - a)
        t = np.clip(np.dot(p - a, ab) / ab_sq, 0.0, 1.0)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def compute_obb_surface_distance(self, obb1, obb2):
        """
        Compute minimum surface-to-surface distance between two OBBs in XY plane.

        Returns 0.0 if the OBBs overlap (SAT test), otherwise the minimum
        Euclidean distance between the two convex quadrilateral boundaries.
        """
        if self.check_overlap(obb1, obb2):
            return 0.0

        # Project bottom-face corners to XY in CW winding: 0,1,3,2
        p1 = [np.array([p[0], p[1]]) for p in [obb1[0], obb1[1], obb1[3], obb1[2]]]
        p2 = [np.array([p[0], p[1]]) for p in [obb2[0], obb2[1], obb2[3], obb2[2]]]

        min_dist = float('inf')

        # Check all vertex-to-edge distances in both directions
        for poly_a, poly_b in [(p1, p2), (p2, p1)]:
            for vertex in poly_a:
                for j in range(len(poly_b)):
                    d = self._point_to_segment_dist(
                        vertex, poly_b[j], poly_b[(j + 1) % len(poly_b)]
                    )
                    min_dist = min(min_dist, d)

        return min_dist

    def analyze_scenario(self, start_time=0, end_time=None):
        """
        Comprehensive scenario analysis with TTC computation and semantic ranking.

        Returns a list of agent analysis results sorted by criticality:
        - TTC-based risk score
        - Object type priority (pedestrians > cyclists > vehicles)
        - Minimum distance achieved

        Each result contains:
        {
            'agent_id': str,
            'agent_path': str,
            'object_type': str,
            'type_priority': int,
            'min_distance': float,
            'min_distance_frame': int,
            'min_distance_time': float,
            'min_ttc': float,
            'min_ttc_frame': int,
            'min_ttc_time': float,
            'valid_frames': (int, int),
            'criticality_score': float,  # Combined risk metric
        }
        """
        if end_time is None:
            end_time = int(self.stage.GetEndTimeCode())

        ego_geom = self.stage.GetPrimAtPath("/World/Agents/Agent_EGO/Geometry")
        if not ego_geom:
            print("❌ EGO not found")
            return []

        # Collect all non-EGO agents
        all_agents = [
            p for p in self.stage.Traverse()
            if p.GetName() == "Geometry"
            and "Agent_" in p.GetPath().pathString
            and "EGO" not in p.GetPath().pathString
        ]

        results = []

        for agent_geom in all_agents:
            agent_xform = self.get_agent_prim(agent_geom)
            agent_path = agent_geom.GetPath().pathString
            agent_id = agent_path.split("Agent_")[1].split("/")[0]

            # Get object type and priority
            type_int, type_str, type_priority = self.get_object_type(agent_xform)

            # Get validity span
            first_valid, last_valid = self.get_validity_span(agent_xform)

            # Track metrics
            min_dist = float('inf')
            min_dist_frame = -1
            min_ttc = float('inf')
            min_ttc_frame = -1

            # Scan frames within agent's valid range
            scan_start = max(start_time, first_valid)
            scan_end = min(end_time, last_valid)

            for t in range(scan_start, scan_end + 1):
                time = Usd.TimeCode(t)

                # OBB surface-to-surface distance (0 when overlapping)
                ego_obb = self.get_obb(ego_geom, time)
                agent_obb = self.get_obb(agent_geom, time)
                dist = self.compute_obb_surface_distance(ego_obb, agent_obb)

                if dist < min_dist:
                    min_dist = dist
                    min_dist_frame = t

                # TTC
                ttc = self.compute_ttc(ego_geom, agent_geom, time)
                if ttc < min_ttc:
                    min_ttc = ttc
                    min_ttc_frame = t

            # Compute criticality score (higher = more critical)
            # Combines: inverse TTC, inverse distance, type priority
            ttc_score = (5.0 - min(min_ttc, 5.0)) * 20 if min_ttc != float('inf') else 0
            dist_score = (10.0 - min(min_dist, 10.0)) * 10 if min_dist != float('inf') else 0
            criticality = ttc_score + dist_score + type_priority

            results.append({
                'agent_id': agent_id,
                'agent_path': agent_path,
                'object_type': type_str,
                'type_priority': type_priority,
                'min_distance': min_dist if min_dist != float('inf') else None,
                'min_distance_frame': min_dist_frame,
                'min_distance_time': self.frame_to_time(min_dist_frame) if min_dist_frame >= 0 else None,
                'min_ttc': min_ttc if min_ttc != float('inf') else None,
                'min_ttc_frame': min_ttc_frame,
                'min_ttc_time': self.frame_to_time(min_ttc_frame) if min_ttc_frame >= 0 else None,
                'valid_frames': (first_valid, last_valid),
                'criticality_score': criticality,
            })

        # Sort by criticality (highest first)
        results.sort(key=lambda x: x['criticality_score'], reverse=True)
        return results

    def print_analysis_report(self, results, top_n=10):
        """Print a formatted analysis report."""
        print("\n" + "=" * 80)
        print("SCENARIO ANALYSIS REPORT - Top Critical Agents")
        print("=" * 80)
        print(f"{'Rank':<5} {'Agent':<12} {'Type':<18} {'MinDist':<10} {'MinTTC':<10} {'Score':<8}")
        print("-" * 80)

        for i, r in enumerate(results[:top_n]):
            dist_str = f"{r['min_distance']:.2f}m" if r['min_distance'] else "N/A"
            ttc_str = f"{r['min_ttc']:.2f}s" if r['min_ttc'] else "N/A"
            print(f"{i+1:<5} {r['agent_id']:<12} {r['object_type']:<18} {dist_str:<10} {ttc_str:<10} {r['criticality_score']:<8.1f}")

        print("=" * 80)

    def write_metrics_to_usd(self, output_path: str, results: Optional[List[dict]] = None):
        """
        Write analysis metrics to USD stage using metrics: namespace.

        For each agent, adds time-sampled attributes:
        - metrics:distanceToEgo (float) - Distance at each frame
        - metrics:ttc (float) - Time-to-collision at each frame (inf if diverging)
        - metrics:closingSpeed (float) - Relative approach speed

        Also adds static metrics:
        - metrics:minDistance (float)
        - metrics:minTTC (float)
        - metrics:criticalityScore (float)

        Args:
            output_path: Path to write the annotated USD file
            results: Optional pre-computed results from analyze_scenario().
                     If not provided, will be computed.
        """
        # Create a copy of the stage for output
        self.stage.Export(output_path)
        output_stage = Usd.Stage.Open(output_path)

        # Create a BBoxCache for the output stage
        output_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        # Get ego geometry prim
        ego_geom = output_stage.GetPrimAtPath("/World/Agents/Agent_EGO/Geometry")
        if not ego_geom:
            print("EGO not found, cannot write metrics")
            return

        ego_xform = ego_geom.GetParent()

        # Get time range
        start_time = int(output_stage.GetStartTimeCode())
        end_time = int(output_stage.GetEndTimeCode())

        # Compute results if not provided
        if results is None:
            results = self.analyze_scenario(start_time, end_time)

        # Build a lookup for summary metrics by agent path
        results_by_path = {r['agent_path']: r for r in results}

        # Get all non-EGO agents
        all_agents = [
            p for p in output_stage.Traverse()
            if p.GetName() == "Geometry"
            and "Agent_" in p.GetPath().pathString
            and "EGO" not in p.GetPath().pathString
        ]

        print(f"Writing metrics for {len(all_agents)} agents to {output_path}...")

        for agent_geom in all_agents:
            agent_path = agent_geom.GetPath().pathString
            agent_xform = agent_geom.GetParent()

            # Get agent validity span
            first_attr = agent_xform.GetAttribute("waymo:firstValidFrame")
            last_attr = agent_xform.GetAttribute("waymo:lastValidFrame")
            first_valid = int(first_attr.Get()) if first_attr and first_attr.Get() is not None else start_time
            last_valid = int(last_attr.Get()) if last_attr and last_attr.Get() is not None else end_time

            # Create time-sampled attributes on the agent xform
            dist_attr = agent_xform.CreateAttribute("metrics:distanceToEgo", Sdf.ValueTypeNames.Float)
            ttc_attr = agent_xform.CreateAttribute("metrics:ttc", Sdf.ValueTypeNames.Float)
            speed_attr = agent_xform.CreateAttribute("metrics:closingSpeed", Sdf.ValueTypeNames.Float)

            # Compute and write time-sampled metrics for each valid frame
            for t in range(max(start_time, first_valid), min(end_time, last_valid) + 1):
                time = Usd.TimeCode(t)
                output_cache.SetTime(time)

                # Get OBB corners and compute centers
                ego_obb = self._get_obb_with_cache(output_cache, ego_geom, time)
                agent_obb = self._get_obb_with_cache(output_cache, agent_geom, time)

                ego_center = np.mean(ego_obb, axis=0)[:2]  # XY only
                agent_center = np.mean(agent_obb, axis=0)[:2]

                dx = agent_center[0] - ego_center[0]
                dy = agent_center[1] - ego_center[1]
                distance = float(np.sqrt(dx**2 + dy**2))

                # Get velocities
                ego_vx_attr = ego_xform.GetAttribute("waymo:velocityX")
                ego_vy_attr = ego_xform.GetAttribute("waymo:velocityY")
                agent_vx_attr = agent_xform.GetAttribute("waymo:velocityX")
                agent_vy_attr = agent_xform.GetAttribute("waymo:velocityY")

                ego_vx = float(ego_vx_attr.Get(time)) if ego_vx_attr and ego_vx_attr.Get(time) is not None else 0.0
                ego_vy = float(ego_vy_attr.Get(time)) if ego_vy_attr and ego_vy_attr.Get(time) is not None else 0.0
                agent_vx = float(agent_vx_attr.Get(time)) if agent_vx_attr and agent_vx_attr.Get(time) is not None else 0.0
                agent_vy = float(agent_vy_attr.Get(time)) if agent_vy_attr and agent_vy_attr.Get(time) is not None else 0.0

                # Relative velocity (agent relative to ego)
                rel_vx = agent_vx - ego_vx
                rel_vy = agent_vy - ego_vy

                # Closing speed along line connecting centers
                # Positive = approaching, negative = separating
                if distance > 0.01:
                    closing_speed = -(dx * rel_vx + dy * rel_vy) / distance
                else:
                    closing_speed = 0.0

                # TTC calculation
                if closing_speed > 0.1:  # Approaching with meaningful speed
                    ttc = distance / closing_speed
                else:
                    ttc = float('inf')

                # Handle infinity for USD (use large sentinel value)
                ttc_value = min(ttc, 9999.0)

                dist_attr.Set(distance, time)
                ttc_attr.Set(float(ttc_value), time)
                speed_attr.Set(float(closing_speed), time)

            # Write static summary metrics from results
            summary = results_by_path.get(agent_path, {})

            min_dist = summary.get('min_distance')
            if min_dist is not None:
                min_dist_attr = agent_xform.CreateAttribute("metrics:minDistance", Sdf.ValueTypeNames.Float)
                min_dist_attr.Set(float(min_dist))

            min_ttc = summary.get('min_ttc')
            if min_ttc is not None:
                min_ttc_attr = agent_xform.CreateAttribute("metrics:minTTC", Sdf.ValueTypeNames.Float)
                min_ttc_attr.Set(float(min(min_ttc, 9999.0)))

            crit = summary.get('criticality_score')
            if crit is not None:
                crit_attr = agent_xform.CreateAttribute("metrics:criticalityScore", Sdf.ValueTypeNames.Float)
                crit_attr.Set(float(crit))

        # Save the stage
        output_stage.GetRootLayer().Save()
        print(f"Metrics written to {output_path}")

    def _get_obb_with_cache(self, cache, prim, time):
        """
        Helper to get OBB corners using a specific cache.
        Returns: 8 corner points in World Space as numpy arrays.
        """
        xform = UsdGeom.Xformable(prim)
        world_transform = xform.ComputeLocalToWorldTransform(time)

        bound = cache.ComputeLocalBound(prim)
        range_min = bound.GetRange().GetMin()
        range_max = bound.GetRange().GetMax()

        corners = [
            Gf.Vec3d(range_min[0], range_min[1], range_min[2]),
            Gf.Vec3d(range_max[0], range_min[1], range_min[2]),
            Gf.Vec3d(range_min[0], range_max[1], range_min[2]),
            Gf.Vec3d(range_max[0], range_max[1], range_min[2]),
            Gf.Vec3d(range_min[0], range_min[1], range_max[2]),
            Gf.Vec3d(range_max[0], range_min[1], range_max[2]),
            Gf.Vec3d(range_min[0], range_max[1], range_max[2]),
            Gf.Vec3d(range_max[0], range_max[1], range_max[2]),
        ]

        world_corners = [np.array(world_transform.Transform(p)) for p in corners]
        return world_corners

    def run_validation(self, start_time=0, end_time=None):
        if end_time is None:
            end_time = int(self.stage.GetEndTimeCode())
        print("💥 Running Collision Validation...")

        # Get Ego and Agents
        ego = self.stage.GetPrimAtPath("/World/Agents/Agent_EGO/Geometry")
        agents = [p for p in self.stage.Traverse() 
                  if "Agent_" in p.GetPath().pathString 
                  and "EGO" not in p.GetPath().pathString 
                  and p.GetName() == "Geometry"]
        
        collisions = []

        # Only compare a pair on frames where both tracks are valid. Outside
        # its waymo:firstValidFrame-lastValidFrame span an agent has no
        # transform samples and USD would hold its nearest one.
        ego_first, ego_last = self.get_validity_span(self.get_agent_prim(ego))
        spans = {}
        for agent in agents:
            agent_first, agent_last = self.get_validity_span(self.get_agent_prim(agent))
            spans[agent.GetPath().pathString] = (max(ego_first, agent_first),
                                                 min(ego_last, agent_last))

        for t in range(start_time, end_time + 1):
            time = Usd.TimeCode(t)
            ego_obb = None

            for agent in agents:
                pair_first, pair_last = spans[agent.GetPath().pathString]
                if t < pair_first or t > pair_last:
                    continue
                if ego_obb is None:
                    ego_obb = self.get_obb(ego, time)
                agent_obb = self.get_obb(agent, time)

                if self.check_overlap(ego_obb, agent_obb):
                    collisions.append((t, agent.GetPath().pathString))
                    print(f"   [Frame {t}] COLLISION DETECTED with {agent.GetPath().pathString}")
                    
        if not collisions:
            print("✅ No Collisions Detected.")
        else:
            print(f"❌ Found {len(collisions)} collision frames.")
        return collisions

    def run_validation_with_removal(self, removal_set, start_time=0, end_time=None):
        """
        Run collision validation with specific agents removed.
        
        Args:
            removal_set: set of agent IDs to exclude (e.g., {2784})
            start_time: starting frame
            end_time: ending frame
            
        Returns:
            bool: True if PASS (no collisions), False if FAIL (collisions found)
        """
        if end_time is None:
            end_time = int(self.stage.GetEndTimeCode())
        print(f"🧪 Testing with removal set: {removal_set}")

        # Get ego
        ego = self.stage.GetPrimAtPath("/World/Agents/Agent_EGO/Geometry")
        
        # Get all agents
        all_agents = [p for p in self.stage.Traverse() 
                      if "Agent_" in p.GetPath().pathString 
                      and "EGO" not in p.GetPath().pathString 
                      and p.GetName() == "Geometry"]
        
        # Filter out removed agents
        active_agents = []
        for agent in all_agents:
            try:
                # Assuming path structure like ".../Agent_1234/..."
                agent_id_str = agent.GetPath().pathString.split("Agent_")[1].split("/")[0]
                if int(agent_id_str) not in removal_set:
                    active_agents.append(agent)
            except (IndexError, ValueError):
                # If naming doesn't match expected pattern, keep the agent to be safe
                active_agents.append(agent)
        
        print(f"   Active agents: {len(active_agents)} (removed {len(removal_set)})")
        
        collision_count = 0
        
        for t in range(start_time, end_time + 1):
            time = Usd.TimeCode(t)
            ego_obb = self.get_obb(ego, time)

            for agent in active_agents:
                agent_obb = self.get_obb(agent, time)

                if self.check_overlap(ego_obb, agent_obb):
                    collision_count += 1
                    break  # Only count one collision per frame to save time
        
        if collision_count == 0:
            print(f"   ✅ PASS: No collisions detected")
            return True
        else:
            print(f"   ❌ FAIL: {collision_count} collision frames detected")
            return False

# --- RUNNER ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run frame-by-frame OBB collision validation between Agent_EGO and all other agents",
        epilog="Example: python collision_check.py scenarios/abc123_base.usd")
    parser.add_argument("usd_file", help="Path to a base scenario USD file")
    args = parser.parse_args()

    validator = CollisionValidator(args.usd_file)
    validator.run_validation()

