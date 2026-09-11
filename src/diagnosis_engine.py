"""
diagnosis_engine.py - Counterfactual Diagnosis Engine for Root-Cause Analysis

Implements the diagnosis algorithm from Section 4.4:
1. Rank all actors using chosen strategy
2. Test removing each sequentially via intervention layers
3. Return first actor whose removal eliminates collision/near-miss

Ranking Strategies (Section 4.5):
- Random: Baseline for comparison
- Distance-Only: Sort by minimum distance to EGO
- Semantic Priority: Pedestrians > Cyclists > Vehicles
- TTC-First: Sort by time-to-collision (most urgent first)

Usage:
    python diagnosis_engine.py --base scenarios/abc123_base.usd --strategy semantic
    python diagnosis_engine.py --base scenarios/abc123_base.usd --strategy ttc --verbose
"""

import os
import random
import tempfile
import shutil
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
from pxr import Usd

from collision_check import CollisionValidator
from intervention_layer import create_intervention_layer


class RankingStrategy(Enum):
    RANDOM = "random"
    DISTANCE = "distance"
    SEMANTIC = "semantic"
    TTC = "ttc"


# Semantic priority weights (higher = more critical)
SEMANTIC_PRIORITY = {
    "TYPE_PEDESTRIAN": 100,
    "TYPE_CYCLIST": 90,
    "TYPE_VEHICLE": 50,
    "TYPE_OTHER": 30,
    "TYPE_UNSET": 10,
}


@dataclass
class DiagnosisResult:
    """Structured output for diagnosis (Section 4.3)."""
    collision: bool
    timestamp: Optional[float]  # Time of collision/near-miss (None if no collision)
    margin_ttc: float           # Minimum TTC observed
    margin_dist: float          # Minimum distance observed
    root_cause_agent: Optional[str] = None  # Agent ID if identified
    removal_set: Optional[Set[str]] = None  # Set of agents removed to eliminate collision
    strategy_used: Optional[str] = None
    agents_tested: int = 0
    explanation: str = ""  # Natural language summary


@dataclass
class AgentRanking:
    """Agent with computed ranking metrics."""
    agent_id: str
    object_type: str
    min_distance: float
    min_ttc: float
    min_distance_frame: int
    priority_score: float  # Combined score for ranking


def generate_diagnosis_explanation(
    scenario_id: str,
    baseline: DiagnosisResult,
    result: DiagnosisResult,
    agent_ranking: Optional[AgentRanking],
    strategy: RankingStrategy,
    rankings: Optional[List[AgentRanking]] = None
) -> str:
    """
    Generate natural language explanation for root-cause diagnosis.

    Args:
        scenario_id: Identifier for the scenario being analyzed
        baseline: DiagnosisResult from baseline (no interventions)
        result: DiagnosisResult after diagnosis (may include root cause)
        agent_ranking: AgentRanking for the identified root cause agent (if found)
        strategy: RankingStrategy used for the diagnosis
        rankings: Full list of agent rankings (for "no single agent" case)

    Examples:
    - "The near-miss was caused by Agent_717 (TYPE_VEHICLE). Removing this
       agent increases minimum distance from 2.34m to 4.12m, resolving the
       safety violation. The agent was ranked #1 using distance-based strategy."

    - "No single agent removal resolves the near-miss. The closest approach
       (2.34m) involves Agent_717 (TYPE_VEHICLE) and Agent_823 (TYPE_PEDESTRIAN).
       Consider Tier 2 pairwise analysis."

    Returns:
        Natural language explanation string
    """
    # Case 1: No collision in baseline - nothing to diagnose
    if not baseline.collision:
        return (
            f"Scenario {scenario_id}: No collision or near-miss detected in baseline. "
            f"Minimum distance to other agents was {baseline.margin_dist:.2f}m "
            f"with TTC of {baseline.margin_ttc:.2f}s. No intervention required."
        )

    # Case 2: Root cause identified - single agent removal resolves issue
    if result.root_cause_agent and agent_ranking:
        # Determine rank of the identified agent
        rank_position = 1
        if rankings:
            for i, r in enumerate(rankings):
                if r.agent_id == result.root_cause_agent:
                    rank_position = i + 1
                    break

        # Format object type for readability
        object_type = agent_ranking.object_type
        type_display = object_type.replace("TYPE_", "").lower()

        # Build the explanation
        explanation = (
            f"The near-miss was caused by Agent_{result.root_cause_agent} ({object_type}). "
            f"Removing this {type_display} increases minimum distance from "
            f"{baseline.margin_dist:.2f}m to {result.margin_dist:.2f}m, "
            f"resolving the safety violation. "
        )

        # Add ranking context
        strategy_name = strategy.value.replace("_", "-") + "-based"
        explanation += (
            f"The agent was ranked #{rank_position} using {strategy_name} strategy"
        )

        # Add additional context based on strategy
        if strategy == RankingStrategy.DISTANCE:
            explanation += f" (closest approach: {agent_ranking.min_distance:.2f}m)"
        elif strategy == RankingStrategy.TTC:
            if agent_ranking.min_ttc < float('inf'):
                explanation += f" (min TTC: {agent_ranking.min_ttc:.2f}s)"
        elif strategy == RankingStrategy.SEMANTIC:
            priority = SEMANTIC_PRIORITY.get(object_type, 10)
            explanation += f" (semantic priority: {priority})"

        explanation += "."

        # Add testing efficiency note
        if result.agents_tested == 1:
            explanation += " Identified on first test."
        else:
            explanation += f" Tested {result.agents_tested} agents before identification."

        return explanation

    # Case 3: No single agent removal resolves the issue
    if rankings and len(rankings) >= 2:
        # Get top 2 agents for context
        top_agents = rankings[:2]
        agent1 = top_agents[0]
        agent2 = top_agents[1]

        explanation = (
            f"No single agent removal resolves the near-miss in scenario {scenario_id}. "
            f"The closest approach ({baseline.margin_dist:.2f}m) involves multiple agents. "
            f"Top candidates: Agent_{agent1.agent_id} ({agent1.object_type}, "
            f"dist={agent1.min_distance:.2f}m) and Agent_{agent2.agent_id} "
            f"({agent2.object_type}, dist={agent2.min_distance:.2f}m). "
        )

        explanation += f"Tested {result.agents_tested} agents using {strategy.value} strategy. "
        explanation += "Consider Tier 2 pairwise analysis."

        return explanation

    # Case 4: Fallback - collision but no rankings available
    return (
        f"Scenario {scenario_id}: Collision/near-miss detected (min distance: "
        f"{baseline.margin_dist:.2f}m) but root cause could not be identified. "
        f"Tested {result.agents_tested} agents using {strategy.value} strategy. "
        f"Consider Tier 2 pairwise analysis or manual inspection."
    )


def rank_agents(validator: CollisionValidator, strategy: RankingStrategy,
                start_frame: int = 0, end_frame: int = None) -> List[AgentRanking]:
    """
    Rank all non-EGO agents according to the specified strategy.

    Returns list of AgentRanking sorted by priority (highest priority first).
    """
    if end_frame is None:
        end_frame = int(validator.stage.GetEndTimeCode())

    # Get analysis results from validator
    results = validator.analyze_scenario(start_frame, end_frame)

    rankings = []
    for r in results:
        if r['agent_id'] == 'EGO':
            continue

        # Compute priority score based on strategy
        if strategy == RankingStrategy.RANDOM:
            priority = random.random()
        elif strategy == RankingStrategy.DISTANCE:
            # Lower distance = higher priority
            priority = 1.0 / (r['min_distance'] + 0.001)
        elif strategy == RankingStrategy.SEMANTIC:
            # Type priority + distance tiebreaker
            type_priority = SEMANTIC_PRIORITY.get(r['object_type'], 10)
            dist_factor = 1.0 / (r['min_distance'] + 0.001)
            priority = type_priority * 1000 + dist_factor
        elif strategy == RankingStrategy.TTC:
            # Lower TTC = higher priority (more urgent)
            ttc = r['min_ttc'] if r['min_ttc'] is not None else float('inf')
            priority = 1.0 / (ttc + 0.001)
        else:
            priority = 0

        rankings.append(AgentRanking(
            agent_id=r['agent_id'],
            object_type=r['object_type'],
            min_distance=r['min_distance'] if r['min_distance'] is not None else float('inf'),
            min_ttc=r['min_ttc'] if r['min_ttc'] is not None else float('inf'),
            min_distance_frame=r['min_distance_frame'] if r['min_distance_frame'] is not None else 0,
            priority_score=priority
        ))

    # Sort by priority (descending)
    rankings.sort(key=lambda x: x.priority_score, reverse=True)
    return rankings


def check_baseline(validator: CollisionValidator, start_frame: int = 0,
                   end_frame: int = None, collision_threshold: float = 0.5) -> DiagnosisResult:
    """
    Check T(emptyset) - baseline with no interventions.

    Returns DiagnosisResult with collision status and margins.
    """
    if end_frame is None:
        end_frame = int(validator.stage.GetEndTimeCode())

    results = validator.analyze_scenario(start_frame, end_frame)

    # Find global minimum distance and TTC
    min_dist = float('inf')
    min_ttc = float('inf')
    collision_time = None
    collision_detected = False

    ego_prim = validator.stage.GetPrimAtPath("/World/Agents/Agent_EGO/Geometry")

    for r in results:
        if r['agent_id'] == 'EGO':
            continue
        if r['min_distance'] is not None and r['min_distance'] < min_dist:
            min_dist = r['min_distance']
        if r['min_ttc'] is not None and r['min_ttc'] < min_ttc:
            min_ttc = r['min_ttc']

    # SAT collision check, limited to frames where both the ego track and the
    # agent track are valid (intersection of the two validity spans).
    ego_first, ego_last = validator.get_validity_span(validator.get_agent_prim(ego_prim))
    for agent_r in results:
        if agent_r['agent_id'] == 'EGO':
            continue
        agent_path = f"/World/Agents/Agent_{agent_r['agent_id']}/Geometry"
        agent_prim = validator.stage.GetPrimAtPath(agent_path)
        if not agent_prim:
            continue

        agent_first, agent_last = agent_r['valid_frames']
        pair_start = max(start_frame, ego_first, agent_first)
        pair_end = min(end_frame, ego_last, agent_last)

        for t in range(pair_start, pair_end + 1):
            time = Usd.TimeCode(t)
            ego_obb = validator.get_obb(ego_prim, time)
            agent_obb = validator.get_obb(agent_prim, time)

            if validator.check_overlap(ego_obb, agent_obb):
                collision_detected = True
                collision_time = validator.frame_to_time(t)
                break

        if collision_detected:
            break

    # Also check if min_distance is below threshold (near-miss = potential collision)
    if min_dist < collision_threshold:
        collision_detected = True
        # Find the frame with minimum distance for timestamp
        for r in results:
            if r['min_distance'] == min_dist and r['min_distance_frame'] is not None:
                collision_time = validator.frame_to_time(r['min_distance_frame'])
                break

    return DiagnosisResult(
        collision=collision_detected,
        timestamp=collision_time,
        margin_ttc=min_ttc if min_ttc != float('inf') else -1,
        margin_dist=min_dist if min_dist != float('inf') else -1
    )


def test_intervention(base_usd_path: str, removal_set: Set[str],
                      start_frame: int = 0, end_frame: int = None,
                      collision_threshold: float = 0.5) -> DiagnosisResult:
    """
    Test T({removal_set}) - scenario with specified agents removed.

    Creates temporary intervention layer and checks for collisions.
    """
    # Create temporary directory for intervention
    temp_dir = tempfile.mkdtemp(prefix="intervention_")

    try:
        # Copy base USD to temp dir for relative path resolution
        base_filename = os.path.basename(base_usd_path)
        temp_base = os.path.join(temp_dir, base_filename)
        shutil.copy(base_usd_path, temp_base)

        # Create intervention layer
        removal_suffix = "_".join(str(x) for x in sorted(removal_set))
        temp_intervention = os.path.join(temp_dir, f"intervention_{removal_suffix}.usd")

        create_intervention_layer(temp_base, removal_set, temp_intervention)

        # Open intervened stage and validate
        validator = CollisionValidator(temp_intervention)

        if end_frame is None:
            end_frame = int(validator.stage.GetEndTimeCode())

        result = check_baseline(validator, start_frame, end_frame, collision_threshold)
        result.removal_set = removal_set

        return result

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def diagnose(base_usd_path: str, strategy: RankingStrategy = RankingStrategy.SEMANTIC,
             collision_threshold: float = 0.5, max_agents: int = 10,
             verbose: bool = False) -> DiagnosisResult:
    """
    Main diagnosis algorithm (Section 4.4):

    1. Check baseline T(emptyset) for collision
    2. If no collision, return early
    3. Rank actors using specified strategy
    4. Test removing each actor sequentially
    5. Return first actor whose removal eliminates collision

    Args:
        base_usd_path: Path to base USD scenario
        strategy: Ranking strategy to use
        collision_threshold: Distance threshold for collision detection (meters)
        max_agents: Maximum number of agents to test
        verbose: Print progress information

    Returns:
        DiagnosisResult with root cause identification
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"DIAGNOSIS: {os.path.basename(base_usd_path)}")
        print(f"Strategy: {strategy.value}")
        print(f"{'='*60}")

    # Load base scenario
    validator = CollisionValidator(base_usd_path)
    end_frame = int(validator.stage.GetEndTimeCode())

    # Step 1: Check baseline
    if verbose:
        print("\n[1] Checking baseline T(emptyset)...")

    baseline = check_baseline(validator, 0, end_frame, collision_threshold)

    if verbose:
        print(f"    Collision: {baseline.collision}")
        print(f"    Min distance: {baseline.margin_dist:.2f}m")
        print(f"    Min TTC: {baseline.margin_ttc:.2f}s")

    # Extract scenario ID from path
    scenario_id = os.path.basename(base_usd_path).replace("_base.usd", "")

    # Step 2: If no collision, nothing to diagnose
    if not baseline.collision:
        if verbose:
            print("\n[2] No collision detected - nothing to diagnose")
        baseline.strategy_used = strategy.value
        baseline.explanation = generate_diagnosis_explanation(
            scenario_id=scenario_id,
            baseline=baseline,
            result=baseline,
            agent_ranking=None,
            strategy=strategy,
            rankings=None
        )
        return baseline

    # Step 3: Rank agents
    if verbose:
        print(f"\n[2] Ranking agents using {strategy.value} strategy...")

    rankings = rank_agents(validator, strategy, 0, end_frame)

    if verbose:
        print(f"    Found {len(rankings)} non-EGO agents")
        for i, r in enumerate(rankings[:5]):
            print(f"    {i+1}. Agent_{r.agent_id} ({r.object_type}) "
                  f"dist={r.min_distance:.2f}m ttc={r.min_ttc:.2f}s")

    # Step 4: Test removing each agent sequentially
    if verbose:
        print(f"\n[3] Testing interventions (max {max_agents} agents)...")

    agents_tested = 0
    for i, agent in enumerate(rankings[:max_agents]):
        agents_tested += 1
        removal_set = {agent.agent_id}

        if verbose:
            print(f"\n    Testing T({{{agent.agent_id}}})...")

        result = test_intervention(base_usd_path, removal_set, 0, end_frame, collision_threshold)

        if verbose:
            print(f"    Collision after removal: {result.collision}")
            print(f"    New min distance: {result.margin_dist:.2f}m")

        # Step 5: Return if collision eliminated
        if not result.collision:
            if verbose:
                print(f"\n[4] ROOT CAUSE IDENTIFIED: Agent_{agent.agent_id}")
                print(f"    Type: {agent.object_type}")
                print(f"    Original distance: {agent.min_distance:.2f}m")
                print(f"    Original TTC: {agent.min_ttc:.2f}s")

            result.root_cause_agent = agent.agent_id
            result.strategy_used = strategy.value
            result.agents_tested = agents_tested
            result.explanation = generate_diagnosis_explanation(
                scenario_id=scenario_id,
                baseline=baseline,
                result=result,
                agent_ranking=agent,
                strategy=strategy,
                rankings=rankings
            )
            return result

    # No single agent removal eliminated collision
    if verbose:
        print(f"\n[4] No single agent identified as root cause")
        print(f"    Tested {agents_tested} agents")
        print(f"    Consider Tier 2 (pairwise removal) analysis")

    baseline.strategy_used = strategy.value
    baseline.agents_tested = agents_tested
    baseline.explanation = generate_diagnosis_explanation(
        scenario_id=scenario_id,
        baseline=baseline,
        result=baseline,
        agent_ranking=None,
        strategy=strategy,
        rankings=rankings
    )
    return baseline


def diagnose_tier2(base_usd_path: str, strategy: RankingStrategy = RankingStrategy.SEMANTIC,
                   collision_threshold: float = 0.5, max_pairs: int = 10,
                   verbose: bool = False) -> DiagnosisResult:
    """
    Tier 2 diagnosis: Test pairwise actor removal.

    Used when single-actor removal doesn't eliminate collision.
    Tests combinations of top-ranked agents.
    """
    scenario_id = os.path.basename(base_usd_path).replace("_base.usd", "")

    if verbose:
        print(f"\n{'='*60}")
        print(f"TIER 2 DIAGNOSIS: {os.path.basename(base_usd_path)}")
        print(f"{'='*60}")

    validator = CollisionValidator(base_usd_path)
    end_frame = int(validator.stage.GetEndTimeCode())

    # Get baseline for comparison
    baseline = check_baseline(validator, 0, end_frame, collision_threshold)

    # Get ranked agents
    rankings = rank_agents(validator, strategy, 0, end_frame)

    # Test pairs of top agents
    pairs_tested = 0
    for i in range(min(len(rankings), 5)):
        for j in range(i + 1, min(len(rankings), 5)):
            if pairs_tested >= max_pairs:
                break

            agent1 = rankings[i]
            agent2 = rankings[j]
            removal_set = {agent1.agent_id, agent2.agent_id}

            if verbose:
                print(f"\n    Testing T({{{agent1.agent_id}, {agent2.agent_id}}})...")

            result = test_intervention(base_usd_path, removal_set, 0, end_frame, collision_threshold)
            pairs_tested += 1

            if not result.collision:
                if verbose:
                    print(f"\n    ROOT CAUSE PAIR: Agent_{agent1.agent_id} + Agent_{agent2.agent_id}")

                result.root_cause_agent = f"{agent1.agent_id}+{agent2.agent_id}"
                result.strategy_used = f"{strategy.value}_tier2"
                result.agents_tested = pairs_tested
                result.explanation = (
                    f"The near-miss in scenario {scenario_id} was caused by the combination of "
                    f"Agent_{agent1.agent_id} ({agent1.object_type}) and "
                    f"Agent_{agent2.agent_id} ({agent2.object_type}). "
                    f"Removing both agents increases minimum distance from "
                    f"{baseline.margin_dist:.2f}m to {result.margin_dist:.2f}m, "
                    f"resolving the safety violation. "
                    f"Identified after testing {pairs_tested} pair(s) using {strategy.value} strategy."
                )
                return result

    if verbose:
        print(f"\n    No pair identified as root cause (tested {pairs_tested} pairs)")

    baseline.strategy_used = f"{strategy.value}_tier2"
    baseline.agents_tested = pairs_tested
    baseline.explanation = (
        f"No pairwise agent removal resolves the near-miss in scenario {scenario_id}. "
        f"Tested {pairs_tested} pairs of top-ranked agents using {strategy.value} strategy. "
        f"The collision (min distance: {baseline.margin_dist:.2f}m) may involve more than "
        f"two agents or require ego intervention analysis."
    )
    return baseline


def compare_strategies(base_usd_path: str, collision_threshold: float = 0.5,
                       verbose: bool = True) -> dict:
    """
    Compare all ranking strategies on a single scenario.

    Returns dict with results per strategy for ablation study.
    """
    results = {}

    for strategy in RankingStrategy:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing strategy: {strategy.value}")

        result = diagnose(base_usd_path, strategy, collision_threshold,
                         max_agents=10, verbose=False)

        results[strategy.value] = {
            'root_cause': result.root_cause_agent,
            'agents_tested': result.agents_tested,
            'collision_eliminated': result.root_cause_agent is not None,
            'margin_dist': result.margin_dist,
            'margin_ttc': result.margin_ttc,
            'explanation': result.explanation
        }

        if verbose:
            status = "FOUND" if result.root_cause_agent else "NOT FOUND"
            print(f"  Result: {status}")
            if result.root_cause_agent:
                print(f"  Root cause: Agent_{result.root_cause_agent}")
                print(f"  Agents tested: {result.agents_tested}")
            print(f"  Explanation: {result.explanation}")

    return results


# --- CLI ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Counterfactual diagnosis for root-cause analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single scenario diagnosis
  python diagnosis_engine.py --base scenarios/abc123_base.usd --strategy semantic

  # Compare all strategies
  python diagnosis_engine.py --base scenarios/abc123_base.usd --compare

  # Tier 2 pairwise analysis
  python diagnosis_engine.py --base scenarios/abc123_base.usd --tier2
        """
    )

    parser.add_argument("--base", required=True, help="Path to base USD file")
    parser.add_argument("--strategy", choices=["random", "distance", "semantic", "ttc"],
                        default="semantic", help="Ranking strategy")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Collision distance threshold (meters)")
    parser.add_argument("--max_agents", type=int, default=10,
                        help="Maximum agents to test")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all strategies")
    parser.add_argument("--tier2", action="store_true",
                        help="Run Tier 2 pairwise analysis")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    if not os.path.exists(args.base):
        print(f"Error: Base USD not found: {args.base}")
        exit(1)

    if args.compare:
        compare_strategies(args.base, args.threshold, verbose=True)
    elif args.tier2:
        strategy = RankingStrategy(args.strategy)
        diagnose_tier2(args.base, strategy, args.threshold, verbose=True)
    else:
        strategy = RankingStrategy(args.strategy)
        result = diagnose(args.base, strategy, args.threshold,
                         args.max_agents, verbose=args.verbose or True)

        print(f"\n{'='*60}")
        print("DIAGNOSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Strategy: {result.strategy_used}")
        print(f"Baseline collision: {result.collision}")
        print(f"Root cause agent: {result.root_cause_agent or 'None identified'}")
        print(f"Agents tested: {result.agents_tested}")
        print(f"Final margin (dist): {result.margin_dist:.2f}m")
        print(f"Final margin (TTC): {result.margin_ttc:.2f}s")
        print(f"\nExplanation:")
        print(f"  {result.explanation}")
