"""
graduated_risk_framework.py - Graduated Risk Framework for Counterfactual Analysis

Implements RSS-inspired tiered risk classification based on proximity thresholds:
- Critical: d < 0.5m (OBB overlap / near-contact)
- High: 0.5m <= d < 2.0m (inside collision threshold)
- Moderate: 2.0m <= d < 5.0m (safety corridor)
- Low: d >= 5.0m (background traffic)

Note: This is a static approximation of RSS's velocity-dependent d_min.
The collision threshold (2.0m) serves as the boundary between High and Moderate.

Analyzes how counterfactual outcomes vary across risk tiers to understand
the relationship between proximity and root-cause diagnosis effectiveness.

Usage:
    python graduated_risk_framework.py --analyze
    python graduated_risk_framework.py --report
    python graduated_risk_framework.py --compare-strategies
"""

import os
import csv
import json
import time
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from collision_check import CollisionValidator, OBJECT_TYPE_PRIORITY
from diagnosis_engine import (
    RankingStrategy, DiagnosisResult, AgentRanking,
    rank_agents, check_baseline, test_intervention, diagnose
)


class RiskTier(Enum):
    """Risk tiers based on minimum proximity to EGO vehicle (RSS-inspired)."""
    CRITICAL = "critical"      # d < 0.5m: OBB overlap / near-contact
    HIGH = "high"              # 0.5m <= d < 2.0m: Inside collision threshold
    MODERATE = "moderate"      # 2.0m <= d < 5.0m: Safety corridor
    LOW = "low"                # d >= 5.0m: Background traffic


# Risk tier threshold boundaries (in meters)
# Static approximation of RSS velocity-dependent d_min.
# The collision threshold (2.0m) is the boundary between High and Moderate.
RISK_TIER_THRESHOLDS = {
    RiskTier.CRITICAL: (0.0, 0.5),    # OBB overlap / near-contact
    RiskTier.HIGH: (0.5, 2.0),        # Inside collision threshold
    RiskTier.MODERATE: (2.0, 5.0),    # Safety corridor
    RiskTier.LOW: (5.0, float('inf')),  # Background traffic
}

# Risk tier characteristics for analysis
RISK_TIER_CHARACTERISTICS = {
    RiskTier.CRITICAL: {
        "name": "Critical",
        "description": "Direct causal agent — safe distance fully violated, impact occurring",
        "expected_intervention": "Emergency stop, aggressive steering",
        "counterfactual_expectation": "Single agent removal highly likely to resolve",
        "ttc_typical": "<0.5s",
    },
    RiskTier.HIGH: {
        "name": "High",
        "description": "Contributing agent — inside d_min, unsafe following distance violated",
        "expected_intervention": "Hard braking, significant path deviation",
        "counterfactual_expectation": "Single agent removal often sufficient",
        "ttc_typical": "0.5-2.0s",
    },
    RiskTier.MODERATE: {
        "name": "Moderate",
        "description": "Contextual risk — inside safety corridor but outside collision threshold",
        "expected_intervention": "Moderate braking, lane positioning",
        "counterfactual_expectation": "May require multiple agent analysis",
        "ttc_typical": "2.0-4.0s",
    },
    RiskTier.LOW: {
        "name": "Low",
        "description": "Background traffic — outside any reasonable safety envelope",
        "expected_intervention": "Speed adjustment, monitoring",
        "counterfactual_expectation": "Root cause unlikely from this agent",
        "ttc_typical": ">4.0s",
    },
}


@dataclass
class ScenarioRiskProfile:
    """Complete risk profile for a single scenario."""
    scenario_id: str
    typology: str
    risk_tier: str
    min_distance_m: float
    min_ttc_s: float
    safety_score: float
    closest_obj_id: str
    closest_obj_type: str
    num_objects: int
    avg_speed_ms: float
    # Diagnosis results per strategy
    diagnosis_results: Dict[str, dict] = field(default_factory=dict)
    # Additional metrics
    tier_percentile: float = 0.0  # Position within tier (0=safest, 1=most critical)


@dataclass
class TierAnalysisResult:
    """Aggregated analysis results for a risk tier."""
    tier: str
    tier_name: str
    threshold_range: Tuple[float, float]
    num_scenarios: int
    scenario_ids: List[str]
    # Distance metrics
    avg_min_distance: float
    distance_std: float
    # TTC metrics
    avg_min_ttc: float
    ttc_std: float
    # Typology distribution
    typology_distribution: Dict[str, int]
    object_type_distribution: Dict[str, int]
    # Counterfactual results per strategy
    strategy_results: Dict[str, dict] = field(default_factory=dict)
    # Diagnosis success metrics
    singleton_resolution_rate: float = 0.0  # % resolved by single agent removal
    avg_agents_tested: float = 0.0
    avg_validator_calls: float = 0.0


@dataclass
class CrossTierComparison:
    """Comparison of counterfactual outcomes across risk tiers."""
    comparison_timestamp: str
    collision_threshold_m: float
    tiers_analyzed: List[str]
    # Per-tier summaries
    tier_summaries: Dict[str, dict]
    # Cross-tier metrics
    resolution_rate_by_tier: Dict[str, float]
    avg_agents_tested_by_tier: Dict[str, float]
    strategy_effectiveness_by_tier: Dict[str, Dict[str, float]]
    # Correlation analysis
    distance_resolution_correlation: float
    ttc_resolution_correlation: float
    # Key findings
    key_findings: List[str]


def classify_scenario_risk_tier(min_distance: float) -> Optional[RiskTier]:
    """
    Classify a scenario into a risk tier based on minimum distance.

    With RSS-inspired thresholds covering [0, inf), returns None only for
    negative distances (which should not occur in practice).
    """
    for tier, (lower, upper) in RISK_TIER_THRESHOLDS.items():
        if lower <= min_distance < upper:
            return tier
    return None


def load_scenarios_with_risk_tiers(csv_path: str = "scenarios.csv") -> Dict[str, ScenarioRiskProfile]:
    """
    Load scenarios from CSV and classify into risk tiers.

    Returns dict mapping scenario_id to ScenarioRiskProfile.
    """
    scenarios = {}

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row['scenario_id'].strip()
            raw_dist = row.get('min_distance_m', '').strip()
            if not raw_dist:
                continue  # Skip scenarios with no distance data
            min_distance = float(raw_dist)

            risk_tier = classify_scenario_risk_tier(min_distance)
            if risk_tier is None:
                continue  # Skip scenarios outside defined tiers

            raw_ttc = row.get('min_ttc_s', '').strip()
            raw_safety = row.get('safety_score', '').strip()
            raw_num = row.get('num_objects', '').strip()
            raw_speed = row.get('avg_speed_ms', '').strip()

            scenarios[sid] = ScenarioRiskProfile(
                scenario_id=sid,
                typology=row.get('typology', 'Unknown'),
                risk_tier=risk_tier.value,
                min_distance_m=min_distance,
                min_ttc_s=float(raw_ttc) if raw_ttc else 0.0,
                safety_score=float(raw_safety) if raw_safety else 0.0,
                closest_obj_id=row.get('closest_obj_id', ''),
                closest_obj_type=row.get('closest_obj_type', ''),
                num_objects=int(raw_num) if raw_num else 0,
                avg_speed_ms=float(raw_speed) if raw_speed else 0.0,
            )

    # Compute tier percentiles
    for tier in RiskTier:
        tier_scenarios = [s for s in scenarios.values() if s.risk_tier == tier.value]
        if tier_scenarios:
            tier_scenarios.sort(key=lambda x: x.min_distance_m)
            for i, s in enumerate(tier_scenarios):
                s.tier_percentile = i / max(len(tier_scenarios) - 1, 1)

    return scenarios


def get_scenarios_by_tier(scenarios: Dict[str, ScenarioRiskProfile]) -> Dict[RiskTier, List[ScenarioRiskProfile]]:
    """Group scenarios by risk tier."""
    by_tier = {tier: [] for tier in RiskTier}

    for scenario in scenarios.values():
        tier = RiskTier(scenario.risk_tier)
        by_tier[tier].append(scenario)

    # Sort within each tier by distance (most critical first)
    for tier in by_tier:
        by_tier[tier].sort(key=lambda x: x.min_distance_m)

    return by_tier


def run_diagnosis_for_scenario(
    scenario: ScenarioRiskProfile,
    scenarios_dir: str = "scenarios",
    collision_threshold: float = 2.0,
    strategies: List[RankingStrategy] = None
) -> Dict[str, dict]:
    """
    Run counterfactual diagnosis for a scenario using all strategies.

    Returns dict mapping strategy name to diagnosis results.
    """
    if strategies is None:
        strategies = list(RankingStrategy)

    usd_path = os.path.join(scenarios_dir, f"{scenario.scenario_id}_base.usd")

    if not os.path.exists(usd_path):
        return {}

    results = {}

    for strategy in strategies:
        try:
            result = diagnose(
                usd_path,
                strategy=strategy,
                collision_threshold=collision_threshold,
                max_agents=15,
                verbose=False
            )

            results[strategy.value] = {
                'collision_detected': result.collision,
                'root_cause_identified': result.root_cause_agent is not None,
                'root_cause_agent': result.root_cause_agent,
                'agents_tested': result.agents_tested,
                'margin_dist': result.margin_dist,
                'margin_ttc': result.margin_ttc,
            }
        except Exception as e:
            results[strategy.value] = {
                'error': str(e),
                'collision_detected': None,
                'root_cause_identified': False,
            }

    return results


def analyze_tier(
    tier: RiskTier,
    scenarios: List[ScenarioRiskProfile],
    scenarios_dir: str = "scenarios",
    collision_threshold: float = 2.0,
    verbose: bool = True
) -> TierAnalysisResult:
    """
    Perform comprehensive analysis on a single risk tier.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Analyzing {RISK_TIER_CHARACTERISTICS[tier]['name']} Tier")
        upper = RISK_TIER_THRESHOLDS[tier][1]
        upper_str = f"{upper}m" if upper != float('inf') else "∞"
        print(f"Distance Range: {RISK_TIER_THRESHOLDS[tier][0]}-{upper_str}")
        print(f"Scenarios: {len(scenarios)}")
        print(f"{'='*70}")

    if not scenarios:
        return TierAnalysisResult(
            tier=tier.value,
            tier_name=RISK_TIER_CHARACTERISTICS[tier]['name'],
            threshold_range=RISK_TIER_THRESHOLDS[tier],
            num_scenarios=0,
            scenario_ids=[],
            avg_min_distance=0.0,
            distance_std=0.0,
            avg_min_ttc=0.0,
            ttc_std=0.0,
            typology_distribution={},
            object_type_distribution={},
        )

    # Collect basic metrics
    distances = [s.min_distance_m for s in scenarios]
    ttcs = [s.min_ttc_s for s in scenarios if s.min_ttc_s > 0]

    import statistics
    avg_distance = statistics.mean(distances)
    distance_std = statistics.stdev(distances) if len(distances) > 1 else 0.0
    avg_ttc = statistics.mean(ttcs) if ttcs else 0.0
    ttc_std = statistics.stdev(ttcs) if len(ttcs) > 1 else 0.0

    # Typology distribution
    typology_dist = defaultdict(int)
    object_type_dist = defaultdict(int)
    for s in scenarios:
        typology_dist[s.typology] += 1
        object_type_dist[s.closest_obj_type] += 1

    # Run diagnosis for each scenario
    strategy_results = defaultdict(lambda: {
        'total': 0,
        'collisions_detected': 0,
        'root_cause_found': 0,
        'total_agents_tested': 0,
        'successful_diagnoses': [],
    })

    for i, scenario in enumerate(scenarios):
        if verbose:
            print(f"\n  [{i+1}/{len(scenarios)}] {scenario.scenario_id}")
            print(f"      Typology: {scenario.typology}, Distance: {scenario.min_distance_m:.2f}m")

        diagnosis_results = run_diagnosis_for_scenario(
            scenario, scenarios_dir, collision_threshold
        )
        scenario.diagnosis_results = diagnosis_results

        for strategy, result in diagnosis_results.items():
            if 'error' in result:
                continue

            strategy_results[strategy]['total'] += 1

            if result.get('collision_detected'):
                strategy_results[strategy]['collisions_detected'] += 1

            if result.get('root_cause_identified'):
                strategy_results[strategy]['root_cause_found'] += 1
                strategy_results[strategy]['successful_diagnoses'].append({
                    'scenario_id': scenario.scenario_id,
                    'agent': result.get('root_cause_agent'),
                    'agents_tested': result.get('agents_tested', 0),
                })

            strategy_results[strategy]['total_agents_tested'] += result.get('agents_tested', 0)

        if verbose and diagnosis_results:
            for strategy, result in diagnosis_results.items():
                status = "ROOT CAUSE" if result.get('root_cause_identified') else "No root cause"
                agent = result.get('root_cause_agent', '-')
                tested = result.get('agents_tested', 0)
                print(f"      {strategy}: {status} (Agent: {agent}, Tested: {tested})")

    # Compute aggregate metrics
    singleton_resolution_rate = 0.0
    avg_agents_tested = 0.0

    # Use best-performing strategy for tier-level metrics
    best_strategy = None
    best_resolution = 0

    for strategy, data in strategy_results.items():
        if data['total'] > 0:
            resolution_rate = data['root_cause_found'] / data['total']
            if resolution_rate > best_resolution:
                best_resolution = resolution_rate
                best_strategy = strategy

            data['resolution_rate'] = round(resolution_rate, 3)
            data['avg_agents_tested'] = round(
                data['total_agents_tested'] / data['total'], 2
            ) if data['total'] > 0 else 0

    if best_strategy:
        singleton_resolution_rate = best_resolution
        avg_agents_tested = strategy_results[best_strategy]['avg_agents_tested']

    return TierAnalysisResult(
        tier=tier.value,
        tier_name=RISK_TIER_CHARACTERISTICS[tier]['name'],
        threshold_range=RISK_TIER_THRESHOLDS[tier],
        num_scenarios=len(scenarios),
        scenario_ids=[s.scenario_id for s in scenarios],
        avg_min_distance=round(avg_distance, 3),
        distance_std=round(distance_std, 3),
        avg_min_ttc=round(avg_ttc, 3),
        ttc_std=round(ttc_std, 3),
        typology_distribution=dict(typology_dist),
        object_type_distribution=dict(object_type_dist),
        strategy_results={k: dict(v) for k, v in strategy_results.items()},
        singleton_resolution_rate=round(singleton_resolution_rate, 3),
        avg_agents_tested=round(avg_agents_tested, 2),
    )


def compute_cross_tier_comparison(
    tier_results: Dict[RiskTier, TierAnalysisResult],
    scenarios: Dict[str, ScenarioRiskProfile]
) -> CrossTierComparison:
    """
    Compute cross-tier comparison of counterfactual outcomes.
    """
    tier_summaries = {}
    resolution_by_tier = {}
    agents_tested_by_tier = {}
    strategy_effectiveness = defaultdict(dict)

    for tier, result in tier_results.items():
        tier_summaries[tier.value] = {
            'name': result.tier_name,
            'num_scenarios': result.num_scenarios,
            'avg_distance': result.avg_min_distance,
            'avg_ttc': result.avg_min_ttc,
            'resolution_rate': result.singleton_resolution_rate,
            'avg_agents_tested': result.avg_agents_tested,
            'typology_distribution': result.typology_distribution,
        }

        resolution_by_tier[tier.value] = result.singleton_resolution_rate
        agents_tested_by_tier[tier.value] = result.avg_agents_tested

        # Strategy effectiveness per tier
        for strategy, data in result.strategy_results.items():
            strategy_effectiveness[strategy][tier.value] = data.get('resolution_rate', 0)

    # Compute correlations (simplified - using rank correlation concept)
    # Distance vs resolution rate
    tiers_ordered = [RiskTier.CRITICAL, RiskTier.HIGH, RiskTier.MODERATE, RiskTier.LOW]
    distances = []
    resolutions = []

    for tier in tiers_ordered:
        if tier in tier_results and tier_results[tier].num_scenarios > 0:
            distances.append(tier_results[tier].avg_min_distance)
            resolutions.append(tier_results[tier].singleton_resolution_rate)

    # Simple correlation calculation
    def compute_correlation(x, y):
        if len(x) < 2:
            return 0.0
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
        if denom_x * denom_y == 0:
            return 0.0
        return numerator / (denom_x * denom_y)

    distance_resolution_corr = compute_correlation(distances, resolutions)

    # TTC vs resolution (for scenarios with valid TTC)
    ttcs = []
    ttc_resolutions = []
    for tier in tiers_ordered:
        if tier in tier_results and tier_results[tier].avg_min_ttc > 0:
            ttcs.append(tier_results[tier].avg_min_ttc)
            ttc_resolutions.append(tier_results[tier].singleton_resolution_rate)

    ttc_resolution_corr = compute_correlation(ttcs, ttc_resolutions)

    # Generate key findings
    key_findings = []

    # Finding 1: Resolution rate trend
    if resolution_by_tier:
        best_tier = max(resolution_by_tier.items(), key=lambda x: x[1])
        worst_tier = min(resolution_by_tier.items(), key=lambda x: x[1])
        key_findings.append(
            f"Highest resolution rate in {best_tier[0]} tier ({best_tier[1]*100:.1f}%), "
            f"lowest in {worst_tier[0]} tier ({worst_tier[1]*100:.1f}%)"
        )

    # Finding 2: Agents tested trend
    if agents_tested_by_tier:
        avg_critical = agents_tested_by_tier.get('critical', 0)
        avg_low = agents_tested_by_tier.get('low', 0)
        if avg_critical > 0 and avg_low > 0:
            key_findings.append(
                f"Critical tier requires {avg_critical:.1f} agent tests on average vs "
                f"{avg_low:.1f} for low-risk tier"
            )

    # Finding 3: Correlation insight
    if abs(distance_resolution_corr) > 0.5:
        direction = "negative" if distance_resolution_corr < 0 else "positive"
        key_findings.append(
            f"Strong {direction} correlation ({distance_resolution_corr:.2f}) between "
            f"proximity and resolution success"
        )

    # Finding 4: Strategy effectiveness variation
    for strategy, tier_rates in strategy_effectiveness.items():
        if tier_rates:
            rates = list(tier_rates.values())
            if max(rates) - min(rates) > 0.3:
                key_findings.append(
                    f"Strategy '{strategy}' shows significant variation across tiers "
                    f"({min(rates)*100:.0f}%-{max(rates)*100:.0f}%)"
                )

    return CrossTierComparison(
        comparison_timestamp=datetime.now().isoformat(),
        collision_threshold_m=2.0,
        tiers_analyzed=[t.value for t in tier_results.keys()],
        tier_summaries=tier_summaries,
        resolution_rate_by_tier=resolution_by_tier,
        avg_agents_tested_by_tier=agents_tested_by_tier,
        strategy_effectiveness_by_tier=dict(strategy_effectiveness),
        distance_resolution_correlation=round(distance_resolution_corr, 3),
        ttc_resolution_correlation=round(ttc_resolution_corr, 3),
        key_findings=key_findings,
    )


def print_tier_distribution_summary(scenarios: Dict[str, ScenarioRiskProfile]):
    """Print summary of scenario distribution across risk tiers."""
    by_tier = get_scenarios_by_tier(scenarios)

    print("\n" + "=" * 70)
    print("GRADUATED RISK FRAMEWORK - TIER DISTRIBUTION")
    print("=" * 70)

    total = sum(len(s) for s in by_tier.values())

    for tier in RiskTier:
        tier_scenarios = by_tier[tier]
        count = len(tier_scenarios)
        pct = (count / total * 100) if total > 0 else 0

        char = RISK_TIER_CHARACTERISTICS[tier]
        bounds = RISK_TIER_THRESHOLDS[tier]

        upper_str = f"{bounds[1]}m" if bounds[1] != float('inf') else "∞"
        print(f"\n{char['name']} ({bounds[0]}-{upper_str})")
        print("-" * 50)
        print(f"  Count: {count} scenarios ({pct:.1f}%)")
        print(f"  Description: {char['description']}")
        print(f"  Expected TTC: {char['ttc_typical']}")

        if tier_scenarios:
            typologies = defaultdict(int)
            obj_types = defaultdict(int)
            for s in tier_scenarios:
                typologies[s.typology] += 1
                obj_types[s.closest_obj_type] += 1

            print(f"  Typologies: {dict(typologies)}")
            print(f"  Object Types: {dict(obj_types)}")
            print(f"  Scenarios: {[s.scenario_id[:8]+'...' for s in tier_scenarios[:5]]}")

    print(f"\n{'='*70}")
    print(f"Total scenarios in framework: {total}")


def print_analysis_report(
    tier_results: Dict[RiskTier, TierAnalysisResult],
    comparison: CrossTierComparison
):
    """Print comprehensive analysis report."""
    print("\n" + "=" * 70)
    print("COUNTERFACTUAL ANALYSIS BY RISK TIER")
    print("=" * 70)

    # Per-tier detailed results
    for tier in [RiskTier.CRITICAL, RiskTier.HIGH, RiskTier.MODERATE, RiskTier.LOW]:
        if tier not in tier_results:
            continue

        result = tier_results[tier]
        char = RISK_TIER_CHARACTERISTICS[tier]

        print(f"\n{'='*70}")
        upper_str = f"{result.threshold_range[1]}m" if result.threshold_range[1] != float('inf') else "∞"
        print(f"{char['name'].upper()} TIER ({result.threshold_range[0]}-{upper_str})")
        print(f"{'='*70}")
        print(f"Scenarios: {result.num_scenarios}")
        print(f"Avg Distance: {result.avg_min_distance:.3f}m (std: {result.distance_std:.3f})")
        print(f"Avg TTC: {result.avg_min_ttc:.3f}s (std: {result.ttc_std:.3f})")
        print(f"\nSingleton Resolution Rate: {result.singleton_resolution_rate*100:.1f}%")
        print(f"Avg Agents Tested: {result.avg_agents_tested:.1f}")

        print(f"\nTypology Distribution: {result.typology_distribution}")
        print(f"Object Types: {result.object_type_distribution}")

        print(f"\nStrategy Results:")
        print("-" * 50)
        print(f"{'Strategy':<12} {'Tested':<8} {'Resolved':<10} {'Rate':<8} {'Avg Tests':<10}")
        print("-" * 50)

        for strategy, data in result.strategy_results.items():
            if data.get('total', 0) > 0:
                print(f"{strategy:<12} {data['total']:<8} {data['root_cause_found']:<10} "
                      f"{data.get('resolution_rate', 0)*100:<7.1f}% {data.get('avg_agents_tested', 0):<10.1f}")

    # Cross-tier comparison
    print("\n" + "=" * 70)
    print("CROSS-TIER COMPARISON")
    print("=" * 70)

    print(f"\nResolution Rate by Tier:")
    for tier, rate in comparison.resolution_rate_by_tier.items():
        bar = '#' * int(rate * 30)
        print(f"  {tier:<10}: {bar:<30} {rate*100:.1f}%")

    print(f"\nAvg Agents Tested by Tier:")
    for tier, count in comparison.avg_agents_tested_by_tier.items():
        bar = '#' * int(count * 3)
        print(f"  {tier:<10}: {bar:<30} {count:.1f}")

    print(f"\nCorrelation Analysis:")
    print(f"  Distance vs Resolution: {comparison.distance_resolution_correlation:+.3f}")
    print(f"  TTC vs Resolution: {comparison.ttc_resolution_correlation:+.3f}")

    print(f"\nKey Findings:")
    for i, finding in enumerate(comparison.key_findings, 1):
        print(f"  {i}. {finding}")

    # Strategy effectiveness comparison
    print(f"\n{'='*70}")
    print("STRATEGY EFFECTIVENESS BY TIER")
    print("=" * 70)

    print(f"\n{'Strategy':<12}", end="")
    for tier in [RiskTier.CRITICAL, RiskTier.HIGH, RiskTier.MODERATE, RiskTier.LOW]:
        print(f"{tier.value:<12}", end="")
    print()
    print("-" * 60)

    for strategy, tier_rates in comparison.strategy_effectiveness_by_tier.items():
        print(f"{strategy:<12}", end="")
        for tier in [RiskTier.CRITICAL, RiskTier.HIGH, RiskTier.MODERATE, RiskTier.LOW]:
            rate = tier_rates.get(tier.value, 0)
            print(f"{rate*100:<11.1f}%", end="")
        print()


def save_analysis_results(
    scenarios: Dict[str, ScenarioRiskProfile],
    tier_results: Dict[RiskTier, TierAnalysisResult],
    comparison: CrossTierComparison,
    output_dir: str = "evaluation_output"
):
    """Save all analysis results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    def _sanitize_inf(obj):
        """Replace float('inf') with string 'inf' for JSON serialization."""
        if isinstance(obj, float) and obj == float('inf'):
            return "inf"
        if isinstance(obj, dict):
            return {k: _sanitize_inf(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_sanitize_inf(v) for v in obj)
        return obj

    # Scenario profiles
    scenarios_data = {
        'generated_at': datetime.now().isoformat(),
        'risk_tier_thresholds': _sanitize_inf({k.value: v for k, v in RISK_TIER_THRESHOLDS.items()}),
        'scenarios': {sid: asdict(s) for sid, s in scenarios.items()}
    }
    with open(os.path.join(output_dir, 'risk_tier_scenarios.json'), 'w') as f:
        json.dump(scenarios_data, f, indent=2)
    print(f"Saved: {output_dir}/risk_tier_scenarios.json")

    # Tier analysis results
    tier_data = {
        'generated_at': datetime.now().isoformat(),
        'tier_results': _sanitize_inf({tier.value: asdict(result) for tier, result in tier_results.items()})
    }
    with open(os.path.join(output_dir, 'risk_tier_analysis.json'), 'w') as f:
        json.dump(tier_data, f, indent=2)
    print(f"Saved: {output_dir}/risk_tier_analysis.json")

    # Cross-tier comparison
    comparison_data = asdict(comparison)
    with open(os.path.join(output_dir, 'cross_tier_comparison.json'), 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Saved: {output_dir}/cross_tier_comparison.json")


def run_full_analysis(
    scenarios_csv: str = "scenarios.csv",
    scenarios_dir: str = "scenarios",
    output_dir: str = "evaluation_output",
    collision_threshold: float = 2.0,
    verbose: bool = True
) -> Tuple[Dict[str, ScenarioRiskProfile], Dict[RiskTier, TierAnalysisResult], CrossTierComparison]:
    """
    Run the complete graduated risk framework analysis.
    """
    print("\n" + "=" * 70)
    print("GRADUATED RISK FRAMEWORK ANALYSIS")
    print("=" * 70)
    print(f"Collision Threshold: {collision_threshold}m")
    print(f"Risk Tiers: {[t.value for t in RiskTier]}")

    # Load and classify scenarios
    scenarios = load_scenarios_with_risk_tiers(scenarios_csv)
    print(f"\nLoaded {len(scenarios)} scenarios within risk tier bounds")

    # Print distribution summary
    print_tier_distribution_summary(scenarios)

    # Get scenarios grouped by tier
    by_tier = get_scenarios_by_tier(scenarios)

    # Analyze each tier
    tier_results = {}
    for tier in RiskTier:
        if by_tier[tier]:
            result = analyze_tier(
                tier,
                by_tier[tier],
                scenarios_dir,
                collision_threshold,
                verbose
            )
            tier_results[tier] = result

    # Compute cross-tier comparison
    comparison = compute_cross_tier_comparison(tier_results, scenarios)

    # Print report
    print_analysis_report(tier_results, comparison)

    # Save results
    save_analysis_results(scenarios, tier_results, comparison, output_dir)

    return scenarios, tier_results, comparison


# --- CLI ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Graduated Risk Framework for Counterfactual Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with verbose output
  python graduated_risk_framework.py --analyze --verbose

  # Quick distribution check
  python graduated_risk_framework.py --distribution

  # Custom collision threshold
  python graduated_risk_framework.py --analyze --threshold 1.5
        """
    )

    parser.add_argument("--analyze", action="store_true",
                        help="Run full graduated risk analysis")
    parser.add_argument("--distribution", action="store_true",
                        help="Show tier distribution only (no diagnosis)")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Collision detection threshold (meters)")
    parser.add_argument("--scenarios-csv", default="scenarios.csv",
                        help="Path to scenarios CSV file")
    parser.add_argument("--scenarios-dir", default="scenarios",
                        help="Path to scenarios USD directory")
    parser.add_argument("--output-dir", default="evaluation_output",
                        help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Default to distribution if no action specified
    if not args.analyze and not args.distribution:
        args.distribution = True

    if args.distribution:
        scenarios = load_scenarios_with_risk_tiers(args.scenarios_csv)
        print_tier_distribution_summary(scenarios)

    if args.analyze:
        run_full_analysis(
            scenarios_csv=args.scenarios_csv,
            scenarios_dir=args.scenarios_dir,
            output_dir=args.output_dir,
            collision_threshold=args.threshold,
            verbose=args.verbose
        )

    print("\nDone.")
