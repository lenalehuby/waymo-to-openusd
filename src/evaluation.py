"""
evaluation.py - Thesis Experimental Evaluation Framework

Implements the evaluation methodology from Section 5.2:
1. Oracle computation (Tier 1 & 2) - exhaustive ground truth
2. Strategy evaluation against oracles
3. Metrics computation (validator calls, MRR, Top-K, exact match)

Usage:
    # Full evaluation (all steps)
    python evaluation.py --all

    # Individual steps
    python evaluation.py --oracle-tier1
    python evaluation.py --oracle-tier2
    python evaluation.py --evaluate
    python evaluation.py --metrics

    # Quick test on subset
    python evaluation.py --all --max-scenarios 3
"""

import os
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import itertools

from collision_check import CollisionValidator
from diagnosis_engine import (
    RankingStrategy, DiagnosisResult, AgentRanking,
    rank_agents, check_baseline, test_intervention
)
from graduated_risk_framework import (
    load_scenarios_with_risk_tiers, get_scenarios_by_tier,
    RiskTier, RISK_TIER_THRESHOLDS, ScenarioRiskProfile
)


# Configuration
SCENARIOS_DIR = "scenarios"
SCENARIOS_CSV = "scenarios.csv"
OUTPUT_DIR = "evaluation_output"
COLLISION_THRESHOLD = 2.0  # meters - matches near-miss definition

# Typologies for Tier 2 deep-dive (one scenario per typology)
TYPOLOGIES = ["Intersection", "Highway_Merge", "Urban_Cut_in", "Pedestrian", "Occluded_Turn"]


@dataclass
class OracleTier1Result:
    """Oracle result for a single scenario (Tier 1)."""
    scenario_id: str
    typology: str
    baseline_collision: bool
    baseline_min_dist: float
    baseline_min_ttc: float
    total_agents: int
    passing_singletons: List[str]  # Agent IDs whose removal eliminates collision
    all_results: Dict[str, bool]   # agent_id -> passes (collision-free)
    computation_time_s: float


@dataclass
class OracleTier2Result:
    """Oracle result for pairwise removal (Tier 2)."""
    scenario_id: str
    typology: str
    singleton_failures: List[str]  # Agents that don't pass alone
    passing_pairs: List[Tuple[str, str]]  # Pairs that pass together
    all_pair_results: Dict[str, bool]  # "id1+id2" -> passes
    computation_time_s: float


@dataclass
class StrategyEvalResult:
    """Evaluation result for a single strategy on a single scenario."""
    scenario_id: str
    strategy: str
    validator_calls: int  # Calls until first oracle-correct solution
    wall_time_s: float    # Time to first diagnosis
    found_oracle_match: bool  # Exact match with oracle minimal set
    first_oracle_rank: int    # Rank of first oracle-correct actor (1-indexed, 0 if not found)
    diagnosed_agent: Optional[str]  # Agent identified by strategy
    oracle_agents: List[str]  # Ground truth oracle agents


def load_scenarios_metadata() -> Dict[str, dict]:
    """Load scenario metadata from CSV."""
    import csv
    metadata = {}
    csv_path = os.path.join(os.path.dirname(SCENARIOS_DIR), SCENARIOS_CSV)

    if not os.path.exists(csv_path):
        csv_path = SCENARIOS_CSV

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row['scenario_id'].strip()
            raw_dist = row.get('min_distance_m', '').strip()
            raw_ttc = row.get('min_ttc_s', '').strip()
            metadata[sid] = {
                'typology': row.get('typology', 'Unknown'),
                'min_distance_m': float(raw_dist) if raw_dist else None,
                'min_ttc_s': float(raw_ttc) if raw_ttc else None,
                'closest_obj_id': row.get('closest_obj_id', ''),
            }
    return metadata


def get_scenario_files() -> List[str]:
    """Get list of all base USD scenario files, including synthetic_scenarios/."""
    import glob
    pattern = os.path.join(SCENARIOS_DIR, "*_base.usd")
    synthetic_pattern = os.path.join(SCENARIOS_DIR, "synthetic_scenarios", "*_base.usd")
    return sorted(set(glob.glob(pattern) + glob.glob(synthetic_pattern)))


def get_scenario_id(usd_path: str) -> str:
    """Extract scenario ID from USD file path."""
    return os.path.basename(usd_path).replace("_base.usd", "")


def compute_oracle_tier1(scenario_path: str, metadata: dict,
                          threshold: float = COLLISION_THRESHOLD,
                          verbose: bool = True) -> OracleTier1Result:
    """
    Compute Tier 1 oracle: exhaustively test ALL singleton removals.

    No early stopping - we need complete ground truth for evaluation.
    """
    scenario_id = get_scenario_id(scenario_path)
    start_time = time.time()

    if verbose:
        print(f"\n  Computing Tier 1 oracle for {scenario_id}...")

    # Load scenario
    validator = CollisionValidator(scenario_path)
    end_frame = int(validator.stage.GetEndTimeCode())

    # Check baseline
    baseline = check_baseline(validator, 0, end_frame, threshold)

    if not baseline.collision:
        # No collision in baseline - nothing to diagnose
        return OracleTier1Result(
            scenario_id=scenario_id,
            typology=metadata.get('typology', 'Unknown'),
            baseline_collision=False,
            baseline_min_dist=baseline.margin_dist,
            baseline_min_ttc=baseline.margin_ttc,
            total_agents=0,
            passing_singletons=[],
            all_results={},
            computation_time_s=time.time() - start_time
        )

    # Get all non-EGO agents
    results = validator.analyze_scenario(0, end_frame)
    agents = [r for r in results if r['agent_id'] != 'EGO']

    if verbose:
        print(f"    Baseline: collision=True, dist={baseline.margin_dist:.2f}m")
        print(f"    Testing {len(agents)} agents exhaustively...")

    # Test each singleton
    all_results = {}
    passing_singletons = []

    for i, agent in enumerate(agents):
        agent_id = agent['agent_id']

        try:
            result = test_intervention(scenario_path, {agent_id}, 0, end_frame, threshold)
            passes = not result.collision
            all_results[agent_id] = passes

            if passes:
                passing_singletons.append(agent_id)

            if verbose and (i + 1) % 10 == 0:
                print(f"    Progress: {i+1}/{len(agents)} tested, {len(passing_singletons)} passing")

        except Exception as e:
            if verbose:
                print(f"    Warning: Failed to test agent {agent_id}: {e}")
            all_results[agent_id] = False

    elapsed = time.time() - start_time

    if verbose:
        print(f"    Complete: {len(passing_singletons)}/{len(agents)} singletons pass")
        print(f"    Time: {elapsed:.1f}s")

    return OracleTier1Result(
        scenario_id=scenario_id,
        typology=metadata.get('typology', 'Unknown'),
        baseline_collision=True,
        baseline_min_dist=baseline.margin_dist,
        baseline_min_ttc=baseline.margin_ttc,
        total_agents=len(agents),
        passing_singletons=passing_singletons,
        all_results=all_results,
        computation_time_s=elapsed
    )


def compute_oracle_tier2(scenario_path: str, tier1_result: OracleTier1Result,
                          threshold: float = COLLISION_THRESHOLD,
                          max_pairs: int = 100,
                          verbose: bool = True) -> OracleTier2Result:
    """
    Compute Tier 2 oracle: test pairs where neither singleton passes.

    Only tests pairs of agents that both fail as singletons.
    """
    scenario_id = get_scenario_id(scenario_path)
    start_time = time.time()

    if verbose:
        print(f"\n  Computing Tier 2 oracle for {scenario_id}...")

    # Get agents that fail as singletons
    singleton_failures = [
        agent_id for agent_id, passes in tier1_result.all_results.items()
        if not passes
    ]

    if verbose:
        print(f"    {len(singleton_failures)} singleton failures to pair")

    if len(singleton_failures) < 2:
        return OracleTier2Result(
            scenario_id=scenario_id,
            typology=tier1_result.typology,
            singleton_failures=singleton_failures,
            passing_pairs=[],
            all_pair_results={},
            computation_time_s=time.time() - start_time
        )

    # Load scenario
    validator = CollisionValidator(scenario_path)
    end_frame = int(validator.stage.GetEndTimeCode())

    # Generate pairs
    pairs = list(itertools.combinations(singleton_failures, 2))
    if len(pairs) > max_pairs:
        if verbose:
            print(f"    Limiting to {max_pairs} pairs (of {len(pairs)} possible)")
        # Prioritize pairs by combined criticality (if available)
        pairs = pairs[:max_pairs]

    if verbose:
        print(f"    Testing {len(pairs)} pairs...")

    # Test each pair
    all_pair_results = {}
    passing_pairs = []

    for i, (id1, id2) in enumerate(pairs):
        pair_key = f"{id1}+{id2}"

        try:
            result = test_intervention(scenario_path, {id1, id2}, 0, end_frame, threshold)
            passes = not result.collision
            all_pair_results[pair_key] = passes

            if passes:
                passing_pairs.append((id1, id2))

            if verbose and (i + 1) % 20 == 0:
                print(f"    Progress: {i+1}/{len(pairs)} tested, {len(passing_pairs)} passing")

        except Exception as e:
            if verbose:
                print(f"    Warning: Failed to test pair {pair_key}: {e}")
            all_pair_results[pair_key] = False

    elapsed = time.time() - start_time

    if verbose:
        print(f"    Complete: {len(passing_pairs)}/{len(pairs)} pairs pass")
        print(f"    Time: {elapsed:.1f}s")

    return OracleTier2Result(
        scenario_id=scenario_id,
        typology=tier1_result.typology,
        singleton_failures=singleton_failures,
        passing_pairs=passing_pairs,
        all_pair_results=all_pair_results,
        computation_time_s=elapsed
    )


def evaluate_strategy(scenario_path: str, strategy: RankingStrategy,
                      oracle: OracleTier1Result,
                      threshold: float = COLLISION_THRESHOLD,
                      verbose: bool = False) -> StrategyEvalResult:
    """
    Evaluate a single strategy on a single scenario against the oracle.

    Records:
    - Validator calls until first oracle-correct solution
    - Wall-clock time to first diagnosis
    - Whether it found an oracle-minimal set
    - Rank of first oracle-correct actor
    """
    scenario_id = get_scenario_id(scenario_path)
    start_time = time.time()

    # No collision case
    if not oracle.baseline_collision:
        return StrategyEvalResult(
            scenario_id=scenario_id,
            strategy=strategy.value,
            validator_calls=1,  # Just the baseline check
            wall_time_s=time.time() - start_time,
            found_oracle_match=True,  # No collision = trivially correct
            first_oracle_rank=0,
            diagnosed_agent=None,
            oracle_agents=[]
        )

    # No oracle solutions exist
    if not oracle.passing_singletons:
        return StrategyEvalResult(
            scenario_id=scenario_id,
            strategy=strategy.value,
            validator_calls=oracle.total_agents + 1,  # Baseline + all agents
            wall_time_s=time.time() - start_time,
            found_oracle_match=True,  # Correctly found no solution
            first_oracle_rank=0,
            diagnosed_agent=None,
            oracle_agents=[]
        )

    # Load scenario and rank agents
    validator = CollisionValidator(scenario_path)
    end_frame = int(validator.stage.GetEndTimeCode())

    rankings = rank_agents(validator, strategy, 0, end_frame)

    # Find rank of first oracle-correct agent
    oracle_set = set(oracle.passing_singletons)
    first_oracle_rank = 0
    for i, agent in enumerate(rankings):
        if agent.agent_id in oracle_set:
            first_oracle_rank = i + 1  # 1-indexed
            break

    # Simulate strategy execution (count calls until first oracle match)
    validator_calls = 1  # Baseline check
    diagnosed_agent = None

    for agent in rankings:
        validator_calls += 1
        if agent.agent_id in oracle_set:
            diagnosed_agent = agent.agent_id
            break

    elapsed = time.time() - start_time

    # Check if diagnosed agent is in oracle minimal set
    found_match = diagnosed_agent in oracle_set if diagnosed_agent else False

    return StrategyEvalResult(
        scenario_id=scenario_id,
        strategy=strategy.value,
        validator_calls=validator_calls,
        wall_time_s=elapsed,
        found_oracle_match=found_match,
        first_oracle_rank=first_oracle_rank,
        diagnosed_agent=diagnosed_agent,
        oracle_agents=oracle.passing_singletons
    )


def compute_metrics(eval_results: List[StrategyEvalResult],
                    oracles: Dict[str, OracleTier1Result]) -> dict:
    """
    Compute evaluation metrics per Section 5.2:
    - Mean validator calls per strategy
    - Exact match rate
    - MRR (Mean Reciprocal Rank)
    - Top-K accuracy (K=1, 3, 5)
    """
    # Group results by strategy
    by_strategy = defaultdict(list)
    for r in eval_results:
        by_strategy[r.strategy].append(r)

    metrics = {
        'per_strategy': {},
        'overall': {},
        'per_typology': defaultdict(dict),
    }

    for strategy, results in by_strategy.items():
        # Filter to scenarios with collisions (meaningful evaluation)
        collision_results = [r for r in results if oracles[r.scenario_id].baseline_collision]

        if not collision_results:
            continue

        n = len(collision_results)

        # Mean validator calls
        mean_calls = sum(r.validator_calls for r in collision_results) / n

        # Exact match rate
        exact_matches = sum(1 for r in collision_results if r.found_oracle_match)
        exact_match_rate = exact_matches / n

        # MRR (Mean Reciprocal Rank)
        reciprocal_ranks = []
        for r in collision_results:
            if r.first_oracle_rank > 0:
                reciprocal_ranks.append(1.0 / r.first_oracle_rank)
            else:
                reciprocal_ranks.append(0.0)
        mrr = sum(reciprocal_ranks) / n if reciprocal_ranks else 0.0

        # Top-K accuracy
        top_k_acc = {}
        for k in [1, 3, 5]:
            hits = sum(1 for r in collision_results if 0 < r.first_oracle_rank <= k)
            top_k_acc[f'top_{k}'] = hits / n

        metrics['per_strategy'][strategy] = {
            'num_scenarios': n,
            'mean_validator_calls': round(mean_calls, 2),
            'exact_match_rate': round(exact_match_rate, 3),
            'mrr': round(mrr, 3),
            **{k: round(v, 3) for k, v in top_k_acc.items()},
        }

    # Per-typology breakdown
    for strategy, results in by_strategy.items():
        for r in results:
            oracle = oracles[r.scenario_id]
            if not oracle.baseline_collision:
                continue
            typology = oracle.typology

            if strategy not in metrics['per_typology'][typology]:
                metrics['per_typology'][typology][strategy] = {
                    'scenarios': [],
                    'validator_calls': [],
                    'exact_matches': 0,
                    'ranks': [],
                }

            entry = metrics['per_typology'][typology][strategy]
            entry['scenarios'].append(r.scenario_id)
            entry['validator_calls'].append(r.validator_calls)
            if r.found_oracle_match:
                entry['exact_matches'] += 1
            if r.first_oracle_rank > 0:
                entry['ranks'].append(r.first_oracle_rank)

    # Aggregate per-typology metrics
    for typology, strat_data in metrics['per_typology'].items():
        for strategy, data in strat_data.items():
            n = len(data['scenarios'])
            if n > 0:
                data['mean_validator_calls'] = round(sum(data['validator_calls']) / n, 2)
                data['exact_match_rate'] = round(data['exact_matches'] / n, 3)
                data['mrr'] = round(sum(1/r for r in data['ranks']) / n, 3) if data['ranks'] else 0.0
            # Clean up intermediate data
            del data['validator_calls']
            del data['ranks']

    # Overall summary
    all_collision_results = [r for r in eval_results
                            if oracles[r.scenario_id].baseline_collision]
    if all_collision_results:
        metrics['overall'] = {
            'total_scenarios': len(all_collision_results) // len(by_strategy),
            'strategies_evaluated': list(by_strategy.keys()),
            'collision_threshold_m': COLLISION_THRESHOLD,
        }

    return metrics


def compute_metrics_by_risk_tier(
    eval_results: List[StrategyEvalResult],
    oracles: Dict[str, OracleTier1Result],
    scenario_profiles: Dict[str, ScenarioRiskProfile]
) -> dict:
    """
    Compute evaluation metrics broken down by risk tier.

    Args:
        eval_results: List of strategy evaluation results
        oracles: Dict mapping scenario_id to OracleTier1Result
        scenario_profiles: Dict mapping scenario_id to ScenarioRiskProfile

    Returns:
    {
        'per_tier': {
            'critical': {
                'num_scenarios': int,
                'distance_range': (float, float),
                'per_strategy': {
                    'ttc': {'resolution_rate': float, 'mean_calls': float, 'mrr': float, 'top_1': float, ...},
                    'distance': {...},
                    ...
                }
            },
            'high': {...},
            ...
        },
        'tier_strategy_matrix': [
            # For easy table generation
            {'tier': 'critical', 'strategy': 'ttc', 'resolution_rate': 0.85, 'mean_calls': 2.3, 'mrr': 0.78},
            ...
        ],
        'summary': {
            'best_strategy_by_tier': {'critical': 'ttc', 'high': 'semantic', ...},
            'tier_resolution_rates': {'critical': 0.85, 'high': 0.72, ...}
        }
    }
    """
    # Map scenario_id to risk tier
    scenario_to_tier = {}
    for sid, profile in scenario_profiles.items():
        scenario_to_tier[sid] = profile.risk_tier

    # Group results by tier and strategy
    by_tier_strategy = defaultdict(lambda: defaultdict(list))

    for r in eval_results:
        tier = scenario_to_tier.get(r.scenario_id)
        if tier is None:
            continue
        # Only include scenarios with baseline collisions
        if r.scenario_id in oracles and oracles[r.scenario_id].baseline_collision:
            by_tier_strategy[tier][r.strategy].append(r)

    metrics = {
        'per_tier': {},
        'tier_strategy_matrix': [],
        'summary': {
            'best_strategy_by_tier': {},
            'tier_resolution_rates': {}
        }
    }

    # Compute metrics for each tier
    for tier in [RiskTier.CRITICAL, RiskTier.HIGH, RiskTier.MODERATE, RiskTier.LOW]:
        tier_name = tier.value
        tier_data = by_tier_strategy.get(tier_name, {})

        if not tier_data:
            continue

        # Count unique scenarios in this tier
        all_scenarios_in_tier = set()
        for strategy_results in tier_data.values():
            for r in strategy_results:
                all_scenarios_in_tier.add(r.scenario_id)

        tier_metrics = {
            'num_scenarios': len(all_scenarios_in_tier),
            'distance_range': RISK_TIER_THRESHOLDS[tier],
            'per_strategy': {}
        }

        best_resolution = 0
        best_strategy = None

        for strategy, results in tier_data.items():
            if not results:
                continue

            n = len(results)

            # Mean validator calls
            mean_calls = sum(r.validator_calls for r in results) / n

            # Exact match / resolution rate
            exact_matches = sum(1 for r in results if r.found_oracle_match)
            resolution_rate = exact_matches / n

            # MRR (Mean Reciprocal Rank)
            reciprocal_ranks = []
            for r in results:
                if r.first_oracle_rank > 0:
                    reciprocal_ranks.append(1.0 / r.first_oracle_rank)
                else:
                    reciprocal_ranks.append(0.0)
            mrr = sum(reciprocal_ranks) / n if reciprocal_ranks else 0.0

            # Top-K accuracy
            top_k_acc = {}
            for k in [1, 3, 5]:
                hits = sum(1 for r in results if 0 < r.first_oracle_rank <= k)
                top_k_acc[f'top_{k}'] = round(hits / n, 3)

            strategy_metrics = {
                'num_scenarios': n,
                'resolution_rate': round(resolution_rate, 3),
                'mean_calls': round(mean_calls, 2),
                'mrr': round(mrr, 3),
                **top_k_acc
            }

            tier_metrics['per_strategy'][strategy] = strategy_metrics

            # Add to matrix format
            metrics['tier_strategy_matrix'].append({
                'tier': tier_name,
                'strategy': strategy,
                'resolution_rate': strategy_metrics['resolution_rate'],
                'mean_calls': strategy_metrics['mean_calls'],
                'mrr': strategy_metrics['mrr'],
                'top_1': strategy_metrics.get('top_1', 0),
                'top_3': strategy_metrics.get('top_3', 0),
                'num_scenarios': n
            })

            # Track best strategy
            if resolution_rate > best_resolution:
                best_resolution = resolution_rate
                best_strategy = strategy

        metrics['per_tier'][tier_name] = tier_metrics

        # Update summary
        if best_strategy:
            metrics['summary']['best_strategy_by_tier'][tier_name] = best_strategy
            metrics['summary']['tier_resolution_rates'][tier_name] = round(best_resolution, 3)

    # Sort matrix by tier order then strategy
    tier_order = {'critical': 0, 'high': 1, 'moderate': 2, 'low': 3}
    metrics['tier_strategy_matrix'].sort(
        key=lambda x: (tier_order.get(x['tier'], 99), x['strategy'])
    )

    return metrics


def print_risk_tier_metrics(metrics: dict):
    """Print formatted risk tier metrics summary."""
    print("\n" + "=" * 70)
    print("METRICS BY RISK TIER")
    print("=" * 70)

    for tier_name in ['critical', 'high', 'moderate', 'low']:
        if tier_name not in metrics['per_tier']:
            continue

        tier_data = metrics['per_tier'][tier_name]
        dist_range = tier_data['distance_range']

        print(f"\n--- {tier_name.upper()} TIER ({dist_range[0]}-{dist_range[1]}m) ---")
        print(f"Scenarios: {tier_data['num_scenarios']}")
        print()

        # Header
        print(f"{'Strategy':<12} {'Resolved':<10} {'Calls':<8} {'MRR':<8} {'Top-1':<8} {'Top-3':<8}")
        print("-" * 60)

        for strategy in ['semantic', 'distance', 'ttc', 'random']:
            if strategy not in tier_data['per_strategy']:
                continue
            data = tier_data['per_strategy'][strategy]
            print(f"{strategy:<12} {data['resolution_rate']*100:<9.1f}% "
                  f"{data['mean_calls']:<8.1f} {data['mrr']:<8.3f} "
                  f"{data.get('top_1', 0)*100:<7.1f}% {data.get('top_3', 0)*100:<7.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best Strategy by Tier")
    print("=" * 70)

    for tier_name, strategy in metrics['summary']['best_strategy_by_tier'].items():
        rate = metrics['summary']['tier_resolution_rates'].get(tier_name, 0)
        print(f"  {tier_name:<10}: {strategy} ({rate*100:.1f}% resolution)")


def run_oracle_tier1(max_scenarios: int = None, verbose: bool = True) -> Dict[str, OracleTier1Result]:
    """Run Tier 1 oracle computation for all scenarios."""
    print("\n" + "=" * 70)
    print("ORACLE COMPUTATION - TIER 1 (Singleton Removal)")
    print("=" * 70)

    metadata = load_scenarios_metadata()
    scenario_files = get_scenario_files()

    if max_scenarios:
        scenario_files = scenario_files[:max_scenarios]

    print(f"Processing {len(scenario_files)} scenarios...")

    oracles = {}
    total_start = time.time()

    for i, path in enumerate(scenario_files):
        sid = get_scenario_id(path)
        meta = metadata.get(sid, {})

        print(f"\n[{i+1}/{len(scenario_files)}] {sid} ({meta.get('typology', 'Unknown')})")

        try:
            result = compute_oracle_tier1(path, meta, COLLISION_THRESHOLD, verbose)
            oracles[sid] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Tier 1 complete: {len(oracles)} scenarios in {total_time:.1f}s")

    # Summary statistics
    with_collision = [o for o in oracles.values() if o.baseline_collision]
    if with_collision:
        avg_agents = sum(o.total_agents for o in with_collision) / len(with_collision)
        avg_passing = sum(len(o.passing_singletons) for o in with_collision) / len(with_collision)
        print(f"Scenarios with collision: {len(with_collision)}")
        print(f"Avg agents per scenario: {avg_agents:.1f}")
        print(f"Avg passing singletons: {avg_passing:.1f}")

    return oracles


def run_oracle_tier2(tier1_oracles: Dict[str, OracleTier1Result],
                      verbose: bool = True) -> Dict[str, OracleTier2Result]:
    """Run Tier 2 oracle computation for one scenario per typology."""
    print("\n" + "=" * 70)
    print("ORACLE COMPUTATION - TIER 2 (Pairwise Removal)")
    print("=" * 70)

    # Select one scenario per typology (prefer those with most singleton failures)
    by_typology = defaultdict(list)
    for sid, oracle in tier1_oracles.items():
        if oracle.baseline_collision:
            by_typology[oracle.typology].append((sid, oracle))

    selected = {}
    for typology in TYPOLOGIES:
        if typology in by_typology:
            # Pick scenario with most singleton failures (more pairs to test)
            candidates = by_typology[typology]
            best = max(candidates, key=lambda x: len(x[1].all_results) - len(x[1].passing_singletons))
            selected[typology] = best

    print(f"Selected {len(selected)} scenarios for Tier 2 (one per typology)")

    tier2_oracles = {}
    total_start = time.time()

    for typology, (sid, tier1) in selected.items():
        path = os.path.join(SCENARIOS_DIR, f"{sid}_base.usd")
        print(f"\n[{typology}] {sid}")

        try:
            result = compute_oracle_tier2(path, tier1, COLLISION_THRESHOLD,
                                          max_pairs=100, verbose=verbose)
            tier2_oracles[sid] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Tier 2 complete: {len(tier2_oracles)} scenarios in {total_time:.1f}s")

    return tier2_oracles


def run_strategy_evaluation(tier1_oracles: Dict[str, OracleTier1Result],
                            verbose: bool = True) -> List[StrategyEvalResult]:
    """Evaluate all strategies on all scenarios."""
    print("\n" + "=" * 70)
    print("STRATEGY EVALUATION")
    print("=" * 70)

    strategies = list(RankingStrategy)
    scenario_files = [os.path.join(SCENARIOS_DIR, f"{sid}_base.usd")
                      for sid in tier1_oracles.keys()]

    print(f"Evaluating {len(strategies)} strategies on {len(scenario_files)} scenarios")

    all_results = []
    total_start = time.time()

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy.value} ---")
        strategy_start = time.time()

        for path in scenario_files:
            sid = get_scenario_id(path)
            oracle = tier1_oracles[sid]

            try:
                result = evaluate_strategy(path, strategy, oracle, COLLISION_THRESHOLD, verbose)
                all_results.append(result)

                if verbose:
                    status = "✓" if result.found_oracle_match else "✗"
                    print(f"  {sid}: calls={result.validator_calls}, rank={result.first_oracle_rank} {status}")

            except Exception as e:
                print(f"  ERROR on {sid}: {e}")
                continue

        strategy_time = time.time() - strategy_start
        print(f"  Completed in {strategy_time:.1f}s")

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Evaluation complete: {len(all_results)} results in {total_time:.1f}s")

    return all_results


def save_results(tier1_oracles: Dict[str, OracleTier1Result],
                 tier2_oracles: Dict[str, OracleTier2Result],
                 eval_results: List[StrategyEvalResult],
                 metrics: dict):
    """Save all results to JSON files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Oracle Tier 1
    tier1_data = {
        'generated_at': datetime.now().isoformat(),
        'collision_threshold_m': COLLISION_THRESHOLD,
        'scenarios': {sid: asdict(o) for sid, o in tier1_oracles.items()}
    }
    with open(os.path.join(OUTPUT_DIR, 'oracle_tier1.json'), 'w') as f:
        json.dump(tier1_data, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/oracle_tier1.json")

    # Oracle Tier 2
    tier2_data = {
        'generated_at': datetime.now().isoformat(),
        'collision_threshold_m': COLLISION_THRESHOLD,
        'scenarios': {}
    }
    for sid, o in tier2_oracles.items():
        tier2_data['scenarios'][sid] = {
            **asdict(o),
            'passing_pairs': [list(p) for p in o.passing_pairs]  # Convert tuples
        }
    with open(os.path.join(OUTPUT_DIR, 'oracle_tier2.json'), 'w') as f:
        json.dump(tier2_data, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/oracle_tier2.json")

    # Evaluation results
    eval_data = {
        'generated_at': datetime.now().isoformat(),
        'collision_threshold_m': COLLISION_THRESHOLD,
        'results': [asdict(r) for r in eval_results]
    }
    with open(os.path.join(OUTPUT_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_data, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/evaluation_results.json")

    # Metrics summary
    metrics['generated_at'] = datetime.now().isoformat()
    # Convert defaultdict to regular dict for JSON
    metrics['per_typology'] = dict(metrics['per_typology'])
    with open(os.path.join(OUTPUT_DIR, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/metrics_summary.json")

    # Risk tier metrics (separate file for easier access)
    if 'per_risk_tier' in metrics:
        tier_metrics_data = {
            'generated_at': datetime.now().isoformat(),
            'collision_threshold_m': COLLISION_THRESHOLD,
            **metrics['per_risk_tier']
        }
        with open(os.path.join(OUTPUT_DIR, 'risk_tier_metrics.json'), 'w') as f:
            json.dump(tier_metrics_data, f, indent=2)
        print(f"Saved: {OUTPUT_DIR}/risk_tier_metrics.json")


def print_metrics_summary(metrics: dict):
    """Print formatted metrics summary."""
    print("\n" + "=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)

    print("\nPer-Strategy Results:")
    print("-" * 70)
    print(f"{'Strategy':<12} {'Calls':<8} {'Match%':<8} {'MRR':<8} {'Top-1':<8} {'Top-3':<8}")
    print("-" * 70)

    for strategy, data in metrics['per_strategy'].items():
        print(f"{strategy:<12} {data['mean_validator_calls']:<8} "
              f"{data['exact_match_rate']*100:<7.1f}% {data['mrr']:<8.3f} "
              f"{data.get('top_1', 0)*100:<7.1f}% {data.get('top_3', 0)*100:<7.1f}%")

    print("\nPer-Typology Breakdown:")
    print("-" * 70)
    for typology, strat_data in metrics.get('per_typology', {}).items():
        print(f"\n{typology}:")
        for strategy, data in strat_data.items():
            n = len(data.get('scenarios', []))
            print(f"  {strategy}: {n} scenarios, "
                  f"calls={data.get('mean_validator_calls', 'N/A')}, "
                  f"match={data.get('exact_match_rate', 0)*100:.0f}%, "
                  f"MRR={data.get('mrr', 0):.3f}")


def load_existing_oracles() -> Tuple[Optional[Dict], Optional[Dict]]:
    """Load existing oracle files if available."""
    tier1 = None
    tier2 = None

    tier1_path = os.path.join(OUTPUT_DIR, 'oracle_tier1.json')
    if os.path.exists(tier1_path):
        with open(tier1_path) as f:
            data = json.load(f)
            tier1 = {
                sid: OracleTier1Result(**s)
                for sid, s in data['scenarios'].items()
            }
        print(f"Loaded existing Tier 1 oracles: {len(tier1)} scenarios")

    tier2_path = os.path.join(OUTPUT_DIR, 'oracle_tier2.json')
    if os.path.exists(tier2_path):
        with open(tier2_path) as f:
            data = json.load(f)
            tier2 = {}
            for sid, s in data['scenarios'].items():
                s['passing_pairs'] = [tuple(p) for p in s['passing_pairs']]
                tier2[sid] = OracleTier2Result(**s)
        print(f"Loaded existing Tier 2 oracles: {len(tier2)} scenarios")

    return tier1, tier2


# --- CLI ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Thesis experimental evaluation framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation pipeline
  python evaluation.py --all

  # Only compute oracles
  python evaluation.py --oracle-tier1
  python evaluation.py --oracle-tier2

  # Only run strategy evaluation (requires existing oracles)
  python evaluation.py --evaluate

  # Quick test on subset
  python evaluation.py --all --max-scenarios 3 --verbose
        """
    )

    parser.add_argument("--all", action="store_true",
                        help="Run full evaluation pipeline")
    parser.add_argument("--oracle-tier1", action="store_true",
                        help="Compute Tier 1 oracles only")
    parser.add_argument("--oracle-tier2", action="store_true",
                        help="Compute Tier 2 oracles only")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run strategy evaluation only")
    parser.add_argument("--metrics", action="store_true",
                        help="Compute metrics from existing results")
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="Limit number of scenarios (for testing)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--use-existing", action="store_true",
                        help="Use existing oracle files if available")

    args = parser.parse_args()

    # Default to --all if no specific action
    if not any([args.all, args.oracle_tier1, args.oracle_tier2,
                args.evaluate, args.metrics]):
        args.all = True

    tier1_oracles = None
    tier2_oracles = None
    eval_results = None

    # Try to load existing oracles
    if args.use_existing or args.evaluate or args.metrics:
        tier1_oracles, tier2_oracles = load_existing_oracles()

    # Run Tier 1 oracle computation
    if args.all or args.oracle_tier1:
        tier1_oracles = run_oracle_tier1(args.max_scenarios, args.verbose)

    # Run Tier 2 oracle computation
    if (args.all or args.oracle_tier2) and tier1_oracles:
        tier2_oracles = run_oracle_tier2(tier1_oracles, args.verbose)

    # Run strategy evaluation
    if (args.all or args.evaluate) and tier1_oracles:
        eval_results = run_strategy_evaluation(tier1_oracles, args.verbose)

    # Compute and display metrics
    if tier1_oracles and eval_results:
        metrics = compute_metrics(eval_results, tier1_oracles)
        print_metrics_summary(metrics)

        # Compute and display risk tier metrics
        try:
            scenario_profiles = load_scenarios_with_risk_tiers(SCENARIOS_CSV)
            if scenario_profiles:
                tier_metrics = compute_metrics_by_risk_tier(
                    eval_results, tier1_oracles, scenario_profiles
                )
                print_risk_tier_metrics(tier_metrics)

                # Add tier metrics to the main metrics dict for saving
                metrics['per_risk_tier'] = tier_metrics
        except Exception as e:
            print(f"\nWarning: Could not compute risk tier metrics: {e}")

        # Save all results
        if tier2_oracles is None:
            tier2_oracles = {}
        save_results(tier1_oracles, tier2_oracles, eval_results, metrics)

    elif args.metrics:
        # Load existing evaluation results
        eval_path = os.path.join(OUTPUT_DIR, 'evaluation_results.json')
        if os.path.exists(eval_path) and tier1_oracles:
            with open(eval_path) as f:
                data = json.load(f)
                eval_results = [StrategyEvalResult(**r) for r in data['results']]
            metrics = compute_metrics(eval_results, tier1_oracles)
            print_metrics_summary(metrics)

            # Also compute risk tier metrics
            try:
                scenario_profiles = load_scenarios_with_risk_tiers(SCENARIOS_CSV)
                if scenario_profiles:
                    tier_metrics = compute_metrics_by_risk_tier(
                        eval_results, tier1_oracles, scenario_profiles
                    )
                    print_risk_tier_metrics(tier_metrics)
                    metrics['per_risk_tier'] = tier_metrics
            except Exception as e:
                print(f"\nWarning: Could not compute risk tier metrics: {e}")

            # Save updated metrics
            save_results(tier1_oracles, tier2_oracles or {}, eval_results, metrics)
        else:
            print("Error: Missing oracle or evaluation files")

    print("\nDone.")
