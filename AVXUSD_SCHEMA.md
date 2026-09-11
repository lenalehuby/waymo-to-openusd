# AVxUSD v0.1 Schema

**Version:** 0.1
**Status:** Draft
**Based on:** RLxUSD Section 4.4 patterns

This document defines the schema for AVxUSD (Autonomous Vehicle USD), a standardized format for representing AV scenarios in Universal Scene Description (USD). AVxUSD enables counterfactual analysis, collision detection, and root-cause diagnosis of safety-critical scenarios.

---

## Table of Contents

1. [Scene Primitives](#scene-primitives)
2. [Required Attributes on Agent Prims](#required-attributes-on-agent-prims)
3. [World Prim Attributes](#world-prim-attributes)
4. [Metrics Namespace](#metrics-namespace-optional-post-analysis)
5. [customData Structures](#customdata-structures)
6. [Intervention Layer Schema](#intervention-layer-schema)
7. [Validation Rules](#validation-rules)
8. [Time and Coordinate Conventions](#time-and-coordinate-conventions)

---

## Scene Primitives

| Path | USD Type | Required | Description |
|------|----------|----------|-------------|
| `/World` | Xform | MUST | Scenario root; holds `customData["scenario"]` and global metadata |
| `/World/Agents` | Xform | MUST | Container for all tracked agents |
| `/World/Agents/Agent_{id}` | Xform | MUST | Agent transform with animation + Waymo metadata |
| `/World/Agents/Agent_{id}/Geometry` | Cube | MUST | Visual representation with oriented bounding box (OBB) |
| `/World/Agents/Agent_EGO` | Xform | MUST | Ego vehicle (Self-Driving Car); special-cased for analysis |

### Prim Hierarchy Example

```
/World                              # Xform - scenario root
├── customData["scenario"]          # Scenario metadata
├── waymo:timestampsSeconds         # Frame timestamps
├── waymo:currentTimeIndex          # Prediction start frame
└── /Agents                         # Xform - agent container
    ├── /Agent_EGO                  # Xform - ego vehicle
    │   ├── waymo:objectType        # int: 1 (VEHICLE)
    │   ├── waymo:velocityX         # float (time-sampled)
    │   ├── waymo:velocityY         # float (time-sampled)
    │   └── /Geometry               # Cube - visual OBB
    │       └── displayColor        # [0, 0, 1] (blue for EGO)
    ├── /Agent_717                  # Xform - other agent
    │   ├── waymo:objectType        # int: 2 (PEDESTRIAN)
    │   ├── waymo:velocityX         # float (time-sampled)
    │   ├── waymo:velocityY         # float (time-sampled)
    │   ├── metrics:distanceToEgo   # float (time-sampled, post-analysis)
    │   ├── metrics:ttc             # float (time-sampled, post-analysis)
    │   └── /Geometry               # Cube - visual OBB
    │       └── displayColor        # [1, 0, 0] (red for others)
    └── /Agent_823                  # ... additional agents
```

---

## Required Attributes on Agent Prims

These attributes are defined on `/World/Agents/Agent_{id}` Xform prims.

### Core Identity Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `waymo:objectType` | `int` | MUST | Waymo object type enum value |
| `waymo:objectTypeString` | `string` | MUST | Human-readable type name |

**Object Type Enum Values:**

| Value | String | Description |
|-------|--------|-------------|
| 0 | `TYPE_UNSET` | Unknown or unclassified |
| 1 | `TYPE_VEHICLE` | Cars, trucks, motorcycles |
| 2 | `TYPE_PEDESTRIAN` | Pedestrians (highest priority) |
| 3 | `TYPE_CYCLIST` | Cyclists, scooters |
| 4 | `TYPE_OTHER` | Other objects |

### Kinematic Attributes (Time-Sampled)

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `waymo:velocityX` | `float` | MUST | X velocity in m/s |
| `waymo:velocityY` | `float` | MUST | Y velocity in m/s |

### Extent Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `waymo:extentLength` | `float` | SHOULD | Bounding box length (X) in meters |
| `waymo:extentWidth` | `float` | SHOULD | Bounding box width (Y) in meters |
| `waymo:extentHeight` | `float` | SHOULD | Bounding box height (Z) in meters |

**Note:** Extent attributes MAY be time-sampled if dimensions vary per frame.

### Track Validity Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `waymo:firstValidFrame` | `int` | SHOULD | First frame where track is valid |
| `waymo:lastValidFrame` | `int` | SHOULD | Last frame where track is valid |

### Transform Operations

Agent Xforms MUST define the following operations in order:

1. `xformOp:translate` (time-sampled) - Position in world coordinates
2. `xformOp:rotateZ` (time-sampled) - Heading angle in degrees

### Visibility

Transform samples are written only for frames where the Waymo track is valid. Outside that set USD holds the nearest sample and interpolates across internal gaps, so the agent Xform's standard `visibility` attribute is time-sampled to `inherited` on valid frames and `invisible` on invalid ones (one sample per transition; token attributes use held interpolation). Analysis code that reads `xformOp:*` directly is not affected by visibility; renderers hide the agent.

---

## World Prim Attributes

These attributes are defined on the `/World` prim.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `waymo:timestampsSeconds` | `float[]` | SHOULD | Absolute timestamps for each frame |
| `waymo:currentTimeIndex` | `int` | SHOULD | Frame where prediction starts |
| `waymo:tracksToPredict` | `int[]` | MAY | Track indices flagged for prediction |

---

## Metrics Namespace (Optional, Post-Analysis)

Metrics are computed by analysis tools and written to agent prims using the `metrics:` namespace prefix. This follows the RLxUSD pattern for derived data.

### Time-Sampled Metrics (per frame)

| Attribute | Type | Description |
|-----------|------|-------------|
| `metrics:distanceToEgo` | `float` | Distance to ego vehicle center (meters) |
| `metrics:ttc` | `float` | Time-to-collision in seconds (9999.0 = not approaching) |
| `metrics:closingSpeed` | `float` | Relative approach speed (m/s, positive = approaching) |

### Static Summary Metrics

| Attribute | Type | Description |
|-----------|------|-------------|
| `metrics:minDistance` | `float` | Minimum distance observed over scenario |
| `metrics:minTTC` | `float` | Minimum TTC observed (9999.0 = never approaching) |
| `metrics:criticalityScore` | `float` | Combined risk score (higher = more critical) |

### Criticality Score Formula

```
criticality = ttc_score + dist_score + type_priority

where:
  ttc_score = (5.0 - min(min_ttc, 5.0)) * 20
  dist_score = (10.0 - min(min_dist, 10.0)) * 10
  type_priority = {PEDESTRIAN: 100, CYCLIST: 90, VEHICLE: 50, OTHER: 30, UNSET: 10}
```

---

## customData Structures

USD `customData` is used to store structured metadata following the RLxUSD pattern.

### customData["scenario"] on /World

Stored on base scenario USD files. Required for self-describing scenarios.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `avxusd_version` | `string` | MUST | Schema version (e.g., "0.1") |
| `scenario_id` | `string` | MUST | Unique scenario identifier |
| `sdc_track_id` | `int` | MUST | Ego vehicle track ID |
| `num_agents` | `int` | MUST | Total number of tracked agents |
| `duration_frames` | `int` | MUST | Number of frames in scenario |
| `duration_seconds` | `float` | MUST | Duration in seconds |
| `frames_per_second` | `int` | MUST | Frame rate (typically 10) |
| `typology` | `string` | MAY | Scenario classification (e.g., "Intersection") |
| `min_distance_m` | `float` | MAY | Minimum ego-agent distance |
| `min_ttc_s` | `float` | MAY | Minimum TTC observed |
| `closest_obj_id` | `string` | MAY | ID of most critical agent |
| `closest_obj_type` | `string` | MAY | Type of most critical agent |
| `baseline_collision` | `bool` | MAY | Whether baseline has collision |
| `oracle_singleton_solutions` | `string` | MAY | Comma-separated agent IDs that solve scenario |

**Example:**
```python
{
    "avxusd_version": "0.1",
    "scenario_id": "abc123def456",
    "sdc_track_id": 1234,
    "num_agents": 15,
    "duration_frames": 91,
    "duration_seconds": 9.0,
    "frames_per_second": 10,
    "typology": "Intersection",
    "min_distance_m": 0.35,
    "min_ttc_s": 1.2,
    "closest_obj_id": "717",
    "closest_obj_type": "TYPE_PEDESTRIAN",
    "baseline_collision": True,
    "oracle_singleton_solutions": "717,823"
}
```

### customData["intervention"] on /World

Stored on intervention layer USD files.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `avxusd_version` | `string` | MUST | Schema version |
| `base_scenario` | `string` | MUST | Filename of base USD |
| `removal_set` | `string` | MUST | Comma-separated list of removed agent IDs |
| `removal_set_types` | `string` | MUST | Comma-separated object types of removed agents |
| `created_at` | `string` | MUST | ISO 8601 timestamp |
| `num_removed` | `int` | MUST | Count of deactivated agents |
| `collision_threshold_m` | `float` | MAY | Threshold used for analysis |
| `intervention_reason` | `string` | MAY | Description of intervention |

**Note:** Lists are stored as comma-separated strings for USD crate format compatibility.

**Example:**
```python
{
    "avxusd_version": "0.1",
    "base_scenario": "abc123def456_base.usd",
    "removal_set": "717,823",
    "removal_set_types": "TYPE_PEDESTRIAN,TYPE_VEHICLE",
    "created_at": "2024-01-15T10:30:00",
    "num_removed": 2,
    "collision_threshold_m": 0.5,
    "intervention_reason": "Oracle singleton removal"
}
```

### customData["counterfactual"] on /World

Stored on intervention layers after testing.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `test_passed` | `bool` | MUST | Whether collision was avoided |
| `tested_at` | `string` | MUST | ISO 8601 timestamp |
| `new_min_distance_m` | `float` | SHOULD | Distance after intervention |
| `new_min_ttc_s` | `float` | SHOULD | TTC after intervention |
| `is_oracle_minimal` | `bool` | MAY | Whether this is a minimal solution |
| `agents_tested` | `int` | MAY | Number of agents tested |
| `strategy_used` | `string` | MAY | Ranking strategy used |
| `baseline_collision` | `bool` | MAY | Original collision status |
| `baseline_min_distance_m` | `float` | MAY | Original minimum distance |
| `baseline_min_ttc_s` | `float` | MAY | Original minimum TTC |
| `distance_improvement_m` | `float` | MAY | Improvement in distance |

**Example:**
```python
{
    "test_passed": True,
    "tested_at": "2024-01-15T10:35:00",
    "new_min_distance_m": 3.5,
    "new_min_ttc_s": 2.1,
    "is_oracle_minimal": True,
    "agents_tested": 3,
    "strategy_used": "semantic",
    "baseline_collision": True,
    "baseline_min_distance_m": 0.35,
    "baseline_min_ttc_s": 1.2,
    "distance_improvement_m": 3.15
}
```

---

## Intervention Layer Schema

Intervention layers use USD's composition system to override base scenarios.

### Layer Structure

```
intervention_717.usd
├── subLayerPaths: ["./base.usd"]   # Reference to base
└── /World                           # Over prim
    ├── waymo:removalSet             # IntArray of removed IDs
    ├── waymo:removalSetStr          # StringArray of removed IDs
    ├── customData["intervention"]   # Intervention metadata
    ├── customData["counterfactual"] # Test results (post-analysis)
    └── /Agents
        └── /Agent_717               # Over prim
            ├── active: false        # Deactivate agent
            └── waymo:deactivatedBy  # "intervention_layer"
```

### Intervention Attributes

| Attribute | Type | Location | Description |
|-----------|------|----------|-------------|
| `waymo:removalSet` | `int[]` | `/World` | Numeric agent IDs removed |
| `waymo:removalSetStr` | `string[]` | `/World` | String agent IDs removed |
| `waymo:deactivatedBy` | `string` | Agent prim | Source of deactivation |
| `active` | `bool` | Agent prim | Set to `false` to deactivate |

---

## Validation Rules

### Stage Configuration (MUST)

1. Stage MUST set `upAxis` to `Z` (Waymo convention)
2. Stage MUST set `metersPerUnit` to `1.0`
3. Stage MUST define `startTimeCode` (typically 0)
4. Stage MUST define `endTimeCode` (last frame)
5. Stage SHOULD set `framesPerSecond` to `10`

### Prim Requirements (MUST)

6. `/World` prim MUST exist
7. `/World/Agents` prim MUST exist
8. `/World/Agents/Agent_EGO` prim MUST exist for collision analysis
9. All agent prims MUST have `/Geometry` child with visual representation

### Attribute Requirements (MUST/SHOULD)

10. All agent prims MUST have `waymo:objectType` attribute
11. All agent prims MUST have `waymo:objectTypeString` attribute
12. All agent prims MUST have time-sampled `waymo:velocityX` and `waymo:velocityY`
13. Agent prims SHOULD have `waymo:firstValidFrame` and `waymo:lastValidFrame`

### Time Sampling Rules (MUST)

14. All time-sampled attributes MUST share the same time samples
15. Transform operations MUST be sampled at scenario frame rate (10 Hz)
16. Time codes MUST be integer frame indices (0, 1, 2, ...)

### Intervention Layer Rules (MUST)

17. Intervention layers MUST sublayer their base USD
18. Intervention layers MUST use relative paths for sublayer references
19. Deactivated agents MUST set `active = false`
20. Intervention layers MUST store `customData["intervention"]`

### customData Rules (MUST/SHOULD)

21. `customData["scenario"]` MUST include `avxusd_version`
22. `customData["scenario"]` MUST include `scenario_id`
23. `customData["intervention"]` MUST include `base_scenario`
24. `customData["intervention"]` MUST include `removal_set`

### Sentinel Values

25. TTC values MUST use `9999.0` as sentinel for "not approaching" or infinity
26. Distance values of `-1` indicate "not computed"
27. Frame indices of `-1` indicate "not applicable"

---

## Time and Coordinate Conventions

### Coordinate System

- **Origin:** Arbitrary (typically near scenario center)
- **X-axis:** East (positive)
- **Y-axis:** North (positive)
- **Z-axis:** Up (positive)
- **Units:** Meters
- **Heading:** Counter-clockwise from X-axis, in degrees for USD, radians in source

### Time System

- **Frame Rate:** 10 Hz (100ms per frame)
- **Time Codes:** Integer frame indices (0, 1, 2, ...)
- **Timestamps:** Absolute timestamps stored in `waymo:timestampsSeconds`
- **Duration:** Typically 9.1 seconds (91 frames) for Waymo scenarios

### Conversion Formulas

```python
# Frame to time (seconds)
time_seconds = timestamps_seconds[frame] if timestamps_seconds else frame * 0.1

# Heading conversion (Waymo radians to USD degrees)
degrees = math.degrees(waymo_heading)

# TTC calculation
closing_speed = -(dx * rel_vx + dy * rel_vy) / distance
ttc = distance / closing_speed if closing_speed > 0.1 else infinity
```

---

## File Naming Conventions

| Pattern | Description | Example |
|---------|-------------|---------|
| `{scenario_id}_base.usd` | Base scenario file | `abc123_base.usd` |
| `{scenario_id}_intervention_{ids}.usd` | Intervention layer | `abc123_intervention_717.usd` |
| `{scenario_id}_intervention_{id1}_{id2}.usd` | Multi-agent intervention | `abc123_intervention_717_823.usd` |
| `{scenario_id}_metrics.usd` | Scenario with metrics | `abc123_metrics.usd` |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2024-01 | Initial draft based on RLxUSD patterns |

---

## References

- [Universal Scene Description (USD)](https://openusd.org/release/index.html)
- [Waymo Open Dataset](https://waymo.com/open/)
- RLxUSD: Reinforcement Learning in USD (Section 4.4)
