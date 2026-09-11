"""
generate_report.py - Unified Analysis Report Generator

Combines:
- Risk tier distribution and characteristics
- Strategy comparison results (resolution rate, search cost, MRR)
- Minimal intervention analysis (ego-resolvable vs non-ego-resolvable)
- Per-scenario natural language summaries

Outputs:
- Markdown report for thesis appendix
- JSON summary for programmatic access
- CSV tables for easy import into thesis

Usage:
    python generate_report.py --output thesis_results/
    python generate_report.py --scenarios scenarios/ --output thesis_results/ --verbose
"""

import csv
import json
import os
import statistics
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from graduated_risk_framework import (
    load_scenarios_with_risk_tiers, get_scenarios_by_tier,
    RiskTier, RISK_TIER_THRESHOLDS, RISK_TIER_CHARACTERISTICS,
    ScenarioRiskProfile, run_diagnosis_for_scenario
)
from evaluation import (
    load_existing_oracles, compute_metrics, load_scenarios_metadata,
    OracleTier1Result, StrategyEvalResult, OUTPUT_DIR as EVAL_OUTPUT_DIR
)
from minimal_intervention import (
    EgoInterventionEngine, MinimalInterventionExplanation,
    BRAKING_DECELERATION, BrakingProfile
)
from diagnosis_engine import RankingStrategy, DiagnosisResult, diagnose


@dataclass
class UnifiedScenarioReport:
    """Complete analysis for a single scenario."""
    scenario_id: str
    typology: str
    risk_tier: str

    # Baseline metrics
    min_distance_m: float
    min_ttc_s: float
    closest_agent_id: str
    closest_agent_type: str

    # Root-cause analysis (per strategy)
    root_cause_by_strategy: Dict[str, dict] = field(default_factory=dict)

    # Minimal intervention
    ego_resolvable: bool = False
    min_speed_reduction_pct: Optional[float] = None
    min_braking_profile: Optional[str] = None
    min_braking_decel: Optional[float] = None

    # Natural language summaries
    root_cause_explanation: str = ""
    intervention_explanation: str = ""
    combined_explanation: str = ""  # Comprehensive explanation combining both analyses


@dataclass
class TierSummary:
    """Summary statistics for a risk tier."""
    tier: str
    tier_name: str
    num_scenarios: int
    resolution_rate_by_strategy: Dict[str, float]
    avg_search_cost_by_strategy: Dict[str, float]
    ego_resolvable_fraction: float
    avg_speed_reduction_needed: Optional[float]
    avg_braking_decel_needed: Optional[float]
    common_root_cause_types: Dict[str, int]
    typology_distribution: Dict[str, int]


def generate_combined_explanation(
    scenario_id: str,
    typology: str,
    risk_tier: str,
    min_distance: float,
    root_cause_agent: Optional[str],
    root_cause_type: Optional[str],
    root_cause_explanation: str,
    new_distance_after_removal: Optional[float],
    ego_resolvable: bool,
    speed_reduction_pct: Optional[float],
    braking_profile: Optional[str],
    braking_decel: Optional[float],
    new_distance_after_intervention: Optional[float],
    original_speed: Optional[float] = None
) -> str:
    """
    Generate comprehensive natural language explanation combining root-cause
    analysis with ego intervention analysis.

    Args:
        scenario_id: Unique scenario identifier
        typology: Scenario typology (e.g., "Highway_Merge")
        risk_tier: Risk tier classification
        min_distance: Minimum distance in baseline scenario (meters)
        root_cause_agent: ID of identified root cause agent (if found)
        root_cause_type: Type of root cause agent (e.g., "TYPE_VEHICLE")
        root_cause_explanation: Explanation from diagnosis engine
        new_distance_after_removal: Min distance after removing root cause agent
        ego_resolvable: Whether ego intervention can resolve the situation
        speed_reduction_pct: Speed reduction percentage needed (if applicable)
        braking_profile: Braking profile name (if applicable)
        braking_decel: Braking deceleration in m/s² (if applicable)
        new_distance_after_intervention: Min distance after ego intervention
        original_speed: Original ego speed in m/s (if known)

    Returns:
        Comprehensive natural language explanation string

    Example output:
    "Scenario abc123 (Highway_Merge, High Risk tier):

     Root Cause: The near-miss (min distance 2.45m) was caused by Agent_717
     (TYPE_VEHICLE) cutting into the ego lane. Removing this agent resolves
     the safety violation, increasing minimum distance to 5.2m.

     Ego Alternative: The ego vehicle could also have avoided this situation
     by reducing speed by 15% (from 22 m/s to 18.7 m/s), which would increase
     the minimum distance to 3.1m. This is a physically feasible intervention
     requiring only moderate speed adjustment.

     Recommendation: Both actor removal and ego speed reduction are viable.
     The ego intervention is less disruptive as it doesn't require predicting
     other agents' absence."
    """
    lines = []

    # Header with scenario context
    tier_display = risk_tier.replace("_", " ").title()
    lines.append(f"Scenario {scenario_id} ({typology}, {tier_display} tier):")
    lines.append("")

    # Root Cause Analysis section
    lines.append("Root Cause Analysis:")
    if root_cause_agent:
        # Format agent type for readability
        type_display = root_cause_type.replace("TYPE_", "").lower() if root_cause_type else "agent"

        lines.append(f"  The near-miss (minimum distance {min_distance:.2f}m) was caused by "
                    f"Agent_{root_cause_agent} ({root_cause_type or 'unknown type'}).")

        if new_distance_after_removal is not None and new_distance_after_removal > min_distance:
            lines.append(f"  Removing this {type_display} resolves the safety violation, "
                        f"increasing minimum distance to {new_distance_after_removal:.2f}m.")
        else:
            lines.append(f"  Removing this {type_display} improves the safety margin.")
    else:
        lines.append(f"  No single agent identified as root cause for the near-miss "
                    f"(minimum distance {min_distance:.2f}m).")
        lines.append("  The situation may involve multiple interacting agents or "
                    "require pairwise analysis.")

    lines.append("")

    # Ego Intervention section
    lines.append("Ego Intervention Analysis:")
    if ego_resolvable:
        if speed_reduction_pct is not None:
            # Calculate new speed if original is known
            if original_speed:
                new_speed = original_speed * (1 - speed_reduction_pct / 100)
                lines.append(f"  The ego vehicle could avoid this situation by reducing "
                            f"speed by {speed_reduction_pct:.0f}% "
                            f"(from {original_speed:.1f} m/s to {new_speed:.1f} m/s).")
            else:
                lines.append(f"  The ego vehicle could avoid this situation by reducing "
                            f"speed by {speed_reduction_pct:.0f}%.")

            if new_distance_after_intervention:
                lines.append(f"  This would increase minimum distance to "
                            f"{new_distance_after_intervention:.2f}m.")

            # Assess feasibility
            if speed_reduction_pct <= 15:
                lines.append("  This is a minor speed adjustment, easily achievable.")
            elif speed_reduction_pct <= 30:
                lines.append("  This is a moderate speed adjustment, physically feasible.")
            else:
                lines.append("  This requires significant speed reduction, which may "
                            "impact traffic flow.")

        elif braking_profile and braking_decel:
            lines.append(f"  The ego vehicle could avoid this situation by applying "
                        f"{braking_profile} braking ({braking_decel:.1f} m/s²).")

            if new_distance_after_intervention:
                lines.append(f"  This would increase minimum distance to "
                            f"{new_distance_after_intervention:.2f}m.")

            # Assess feasibility based on braking profile
            if braking_profile == 'comfortable':
                lines.append("  This is comfortable braking within normal driving parameters.")
            elif braking_profile == 'firm':
                lines.append("  This is firm braking, noticeable but controlled.")
            elif braking_profile == 'hard':
                lines.append("  This requires hard braking, which may startle passengers.")
            else:  # emergency
                lines.append("  This requires emergency braking at friction limits.")
    else:
        lines.append("  Ego speed reduction or braking alone is insufficient to resolve "
                    "this near-miss.")
        lines.append("  The situation requires either agent removal or more aggressive "
                    "maneuvers (e.g., steering).")

    lines.append("")

    # Recommendation section
    lines.append("Recommendation:")
    if root_cause_agent and ego_resolvable:
        # Both options available
        if speed_reduction_pct and speed_reduction_pct <= 20:
            lines.append("  Both actor removal and ego speed reduction are viable solutions.")
            lines.append("  The ego intervention (speed reduction) is less disruptive as it "
                        "doesn't require predicting other agents' absence and represents "
                        "a defensive driving approach.")
        elif braking_profile in ['comfortable', 'firm']:
            lines.append("  Both actor removal and ego braking intervention are viable.")
            lines.append("  The ego intervention represents a standard defensive response "
                        "that doesn't depend on counterfactual agent behavior.")
        else:
            lines.append("  Actor removal provides a cleaner resolution, while ego intervention "
                        "requires more aggressive action.")
            lines.append("  For AV planning, identifying and tracking the root cause agent "
                        "may enable earlier, gentler interventions.")

    elif root_cause_agent and not ego_resolvable:
        lines.append("  Actor removal is the primary resolution path.")
        lines.append("  For AV systems, early detection and tracking of "
                    f"Agent_{root_cause_agent} is critical for collision avoidance.")

    elif not root_cause_agent and ego_resolvable:
        lines.append("  Ego intervention is the recommended approach.")
        lines.append("  Since no single agent is responsible, defensive driving "
                    "(speed management) is the most reliable strategy.")

    else:
        lines.append("  This scenario requires advanced intervention strategies.")
        lines.append("  Consider: (1) pairwise agent analysis, (2) steering maneuvers, "
                    "or (3) earlier speed reduction before the critical point.")

    return "\n".join(lines)


def generate_scenario_report(
    scenario_id: str,
    scenarios_dir: str = "scenarios",
    collision_threshold: float = 2.0,
    safety_threshold: float = 2.5,
    risk_profile: Optional[ScenarioRiskProfile] = None,
    oracle: Optional[OracleTier1Result] = None,
    verbose: bool = False
) -> Optional[UnifiedScenarioReport]:
    """Generate complete report for a single scenario."""
    usd_path = os.path.join(scenarios_dir, f"{scenario_id}_base.usd")

    if not os.path.exists(usd_path):
        if verbose:
            print(f"  Warning: USD file not found for {scenario_id}")
        return None

    # Get baseline metrics from risk profile or compute fresh
    if risk_profile:
        min_distance = risk_profile.min_distance_m
        min_ttc = risk_profile.min_ttc_s
        closest_agent_id = risk_profile.closest_obj_id
        closest_agent_type = risk_profile.closest_obj_type
        typology = risk_profile.typology
        risk_tier = risk_profile.risk_tier
    else:
        # Load from metadata
        metadata = load_scenarios_metadata()
        meta = metadata.get(scenario_id, {})
        min_distance = meta.get('min_distance_m', 0)
        min_ttc = meta.get('min_ttc_s', 0) or 0
        closest_agent_id = meta.get('closest_obj_id', '')
        closest_agent_type = ''
        typology = meta.get('typology', 'Unknown')
        # Classify risk tier
        risk_tier = "unknown"
        for tier, (lower, upper) in RISK_TIER_THRESHOLDS.items():
            if lower <= min_distance < upper:
                risk_tier = tier.value
                break

    report = UnifiedScenarioReport(
        scenario_id=scenario_id,
        typology=typology,
        risk_tier=risk_tier,
        min_distance_m=min_distance,
        min_ttc_s=min_ttc,
        closest_agent_id=closest_agent_id,
        closest_agent_type=closest_agent_type
    )

    # Root-cause analysis per strategy
    best_explanation = ""
    for strategy in RankingStrategy:
        try:
            result = diagnose(
                usd_path,
                strategy=strategy,
                collision_threshold=collision_threshold,
                max_agents=15,
                verbose=False
            )

            report.root_cause_by_strategy[strategy.value] = {
                'agent_id': result.root_cause_agent,
                'agents_tested': result.agents_tested,
                'found': result.root_cause_agent is not None,
                'margin_dist': result.margin_dist,
                'explanation': result.explanation
            }

            # Keep best explanation (from semantic strategy preferentially)
            if result.explanation and (strategy == RankingStrategy.SEMANTIC or not best_explanation):
                best_explanation = result.explanation

        except Exception as e:
            if verbose:
                print(f"  Warning: Strategy {strategy.value} failed: {e}")
            report.root_cause_by_strategy[strategy.value] = {
                'agent_id': None,
                'agents_tested': 0,
                'found': False,
                'error': str(e)
            }

    report.root_cause_explanation = best_explanation

    # Minimal intervention analysis
    try:
        engine = EgoInterventionEngine(usd_path, safety_threshold)

        # Test uniform speed reduction
        speed_result = engine.find_minimal_speed_reduction(step_size=5.0, max_reduction=50.0)

        # Test targeted braking
        braking_result = engine.find_minimal_braking(max_lead_time_s=3.0, time_step_s=0.5)

        # Determine if ego-resolvable
        speed_resolved = (speed_result.minimal_intervention is not None and
                         speed_result.minimal_intervention.threshold_achieved)
        braking_resolved = (braking_result.minimal_intervention is not None and
                           braking_result.minimal_intervention.threshold_achieved)

        report.ego_resolvable = speed_resolved or braking_resolved

        if speed_resolved:
            report.min_speed_reduction_pct = speed_result.minimal_intervention.intervention_value

        if braking_resolved:
            report.min_braking_decel = braking_result.minimal_intervention.intervention_value
            # Map deceleration to profile name
            for profile, decel in BRAKING_DECELERATION.items():
                if abs(decel - report.min_braking_decel) < 0.1:
                    report.min_braking_profile = profile.value
                    break

        # Generate intervention explanation
        if speed_resolved and braking_resolved:
            report.intervention_explanation = (
                f"Ego-resolvable via speed reduction ({report.min_speed_reduction_pct:.0f}%) "
                f"or {report.min_braking_profile} braking ({report.min_braking_decel:.1f} m/s²). "
                f"{speed_result.explanation}"
            )
            new_distance_after_intervention = speed_result.minimal_intervention.new_min_distance
        elif speed_resolved:
            report.intervention_explanation = speed_result.explanation
            new_distance_after_intervention = speed_result.minimal_intervention.new_min_distance
        elif braking_resolved:
            report.intervention_explanation = braking_result.explanation
            new_distance_after_intervention = braking_result.minimal_intervention.new_min_distance
        else:
            report.intervention_explanation = (
                f"Not ego-resolvable with speed reduction (up to 50%) or braking alone. "
                f"Requires agent removal or steering intervention."
            )
            new_distance_after_intervention = None

        # Get original ego speed for combined explanation
        original_speed = engine.get_average_ego_speed() if hasattr(engine, 'get_average_ego_speed') else None

    except Exception as e:
        if verbose:
            print(f"  Warning: Intervention analysis failed: {e}")
        report.intervention_explanation = f"Intervention analysis failed: {e}"
        new_distance_after_intervention = None
        original_speed = None

    # Generate combined explanation
    # Get root cause info from semantic strategy (preferred) or first available
    semantic_result = report.root_cause_by_strategy.get('semantic', {})
    root_cause_agent = semantic_result.get('agent_id')
    new_distance_after_removal = semantic_result.get('margin_dist') if root_cause_agent else None

    report.combined_explanation = generate_combined_explanation(
        scenario_id=scenario_id,
        typology=typology,
        risk_tier=risk_tier,
        min_distance=min_distance,
        root_cause_agent=root_cause_agent,
        root_cause_type=closest_agent_type,
        root_cause_explanation=report.root_cause_explanation,
        new_distance_after_removal=new_distance_after_removal,
        ego_resolvable=report.ego_resolvable,
        speed_reduction_pct=report.min_speed_reduction_pct,
        braking_profile=report.min_braking_profile,
        braking_decel=report.min_braking_decel,
        new_distance_after_intervention=new_distance_after_intervention,
        original_speed=original_speed
    )

    return report


def generate_tier_summary(tier: RiskTier, scenarios: List[UnifiedScenarioReport]) -> TierSummary:
    """
    Generate summary statistics for a risk tier.

    Returns TierSummary with:
    - num_scenarios
    - resolution_rate_by_strategy: {strategy: rate}
    - avg_search_cost_by_strategy: {strategy: avg_agents_tested}
    - ego_resolvable_fraction
    - avg_speed_reduction_needed (for ego-resolvable)
    - common_root_cause_types: {type: count}
    """
    if not scenarios:
        return TierSummary(
            tier=tier.value,
            tier_name=RISK_TIER_CHARACTERISTICS[tier]['name'],
            num_scenarios=0,
            resolution_rate_by_strategy={},
            avg_search_cost_by_strategy={},
            ego_resolvable_fraction=0.0,
            avg_speed_reduction_needed=None,
            avg_braking_decel_needed=None,
            common_root_cause_types={},
            typology_distribution={}
        )

    # Resolution rates and search costs per strategy
    resolution_rates = {}
    search_costs = {}

    for strategy in RankingStrategy:
        found_count = 0
        total_agents_tested = 0
        valid_count = 0

        for s in scenarios:
            strat_result = s.root_cause_by_strategy.get(strategy.value, {})
            if 'error' not in strat_result:
                valid_count += 1
                if strat_result.get('found'):
                    found_count += 1
                total_agents_tested += strat_result.get('agents_tested', 0)

        if valid_count > 0:
            resolution_rates[strategy.value] = round(found_count / valid_count, 3)
            search_costs[strategy.value] = round(total_agents_tested / valid_count, 2)

    # Ego-resolvable fraction
    ego_resolvable_count = sum(1 for s in scenarios if s.ego_resolvable)
    ego_resolvable_fraction = ego_resolvable_count / len(scenarios) if scenarios else 0.0

    # Average speed reduction (for ego-resolvable scenarios)
    speed_reductions = [s.min_speed_reduction_pct for s in scenarios
                       if s.min_speed_reduction_pct is not None]
    avg_speed_reduction = statistics.mean(speed_reductions) if speed_reductions else None

    # Average braking deceleration
    braking_decels = [s.min_braking_decel for s in scenarios
                     if s.min_braking_decel is not None]
    avg_braking_decel = statistics.mean(braking_decels) if braking_decels else None

    # Common root cause types
    root_cause_types = defaultdict(int)
    for s in scenarios:
        # Use semantic strategy result as primary
        semantic_result = s.root_cause_by_strategy.get('semantic', {})
        if semantic_result.get('found') and s.closest_agent_type:
            root_cause_types[s.closest_agent_type] += 1

    # Typology distribution
    typology_dist = defaultdict(int)
    for s in scenarios:
        typology_dist[s.typology] += 1

    return TierSummary(
        tier=tier.value,
        tier_name=RISK_TIER_CHARACTERISTICS[tier]['name'],
        num_scenarios=len(scenarios),
        resolution_rate_by_strategy=resolution_rates,
        avg_search_cost_by_strategy=search_costs,
        ego_resolvable_fraction=round(ego_resolvable_fraction, 3),
        avg_speed_reduction_needed=round(avg_speed_reduction, 1) if avg_speed_reduction else None,
        avg_braking_decel_needed=round(avg_braking_decel, 2) if avg_braking_decel else None,
        common_root_cause_types=dict(root_cause_types),
        typology_distribution=dict(typology_dist)
    )


def generate_cross_tier_comparison(tier_summaries: Dict[RiskTier, TierSummary]) -> dict:
    """
    Compare metrics across risk tiers.

    Key insights to surface:
    - How resolution rate changes with proximity (Critical vs Low)
    - Which strategies perform best in each tier
    - Whether ego interventions are sufficient by tier
    """
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'tiers_analyzed': [t.value for t in tier_summaries.keys()],
        'resolution_comparison': {},
        'search_cost_comparison': {},
        'ego_intervention_comparison': {},
        'best_strategy_by_tier': {},
        'key_insights': []
    }

    # Build comparison tables
    for tier, summary in tier_summaries.items():
        comparison['resolution_comparison'][tier.value] = summary.resolution_rate_by_strategy
        comparison['search_cost_comparison'][tier.value] = summary.avg_search_cost_by_strategy
        comparison['ego_intervention_comparison'][tier.value] = {
            'ego_resolvable_fraction': summary.ego_resolvable_fraction,
            'avg_speed_reduction': summary.avg_speed_reduction_needed,
            'avg_braking_decel': summary.avg_braking_decel_needed
        }

        # Best strategy for this tier
        if summary.resolution_rate_by_strategy:
            best_strategy = max(summary.resolution_rate_by_strategy.items(), key=lambda x: x[1])
            comparison['best_strategy_by_tier'][tier.value] = {
                'strategy': best_strategy[0],
                'resolution_rate': best_strategy[1]
            }

    # Generate key insights
    insights = []

    # Insight 1: Resolution rate trend across tiers
    tier_order = [RiskTier.CRITICAL, RiskTier.HIGH, RiskTier.MODERATE, RiskTier.LOW]
    rates_by_tier = []
    for tier in tier_order:
        if tier in tier_summaries:
            rates = tier_summaries[tier].resolution_rate_by_strategy
            if rates:
                avg_rate = statistics.mean(rates.values())
                rates_by_tier.append((tier.value, avg_rate))

    if len(rates_by_tier) >= 2:
        first_tier, first_rate = rates_by_tier[0]
        last_tier, last_rate = rates_by_tier[-1]
        if first_rate > last_rate:
            insights.append(
                f"Resolution rate decreases from {first_tier} ({first_rate*100:.0f}%) "
                f"to {last_tier} ({last_rate*100:.0f}%) as distance increases."
            )
        elif last_rate > first_rate:
            insights.append(
                f"Resolution rate increases from {first_tier} ({first_rate*100:.0f}%) "
                f"to {last_tier} ({last_rate*100:.0f}%) as distance increases."
            )

    # Insight 2: Ego intervention sufficiency
    ego_by_tier = []
    for tier in tier_order:
        if tier in tier_summaries:
            ego_by_tier.append((tier.value, tier_summaries[tier].ego_resolvable_fraction))

    if ego_by_tier:
        critical_ego = next((rate for name, rate in ego_by_tier if name == 'critical'), None)
        low_ego = next((rate for name, rate in ego_by_tier if name == 'low'), None)

        if critical_ego is not None and low_ego is not None:
            if critical_ego < low_ego:
                insights.append(
                    f"Critical tier has lower ego-resolvability ({critical_ego*100:.0f}%) "
                    f"than low-risk tier ({low_ego*100:.0f}%), suggesting multi-agent dynamics."
                )

    # Insight 3: Strategy effectiveness variation
    strategy_variation = defaultdict(list)
    for tier in tier_order:
        if tier in tier_summaries:
            for strategy, rate in tier_summaries[tier].resolution_rate_by_strategy.items():
                strategy_variation[strategy].append(rate)

    for strategy, rates in strategy_variation.items():
        if len(rates) >= 2:
            rate_range = max(rates) - min(rates)
            if rate_range > 0.2:
                insights.append(
                    f"Strategy '{strategy}' shows significant variation across tiers "
                    f"({min(rates)*100:.0f}%-{max(rates)*100:.0f}%)."
                )

    comparison['key_insights'] = insights

    return comparison


def generate_markdown_report(
    scenario_reports: List[UnifiedScenarioReport],
    tier_summaries: Dict[RiskTier, TierSummary],
    cross_tier: dict,
    output_path: str
):
    """
    Generate thesis-ready Markdown report.

    Structure:
    1. Executive Summary
    2. Dataset Overview (scenarios by typology and tier)
    3. Strategy Comparison Results
       - Table: Strategy x Tier resolution rates
       - Table: Strategy x Tier search costs
    4. Minimal Intervention Analysis
       - Ego-resolvable vs non-ego-resolvable breakdown
       - Speed reduction / braking requirements by tier
    5. Per-Scenario Details (appendix-style)
    """
    lines = []

    # Header
    lines.append("# Counterfactual Analysis Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Executive Summary
    lines.append("## 1. Executive Summary\n")

    total_scenarios = len(scenario_reports)
    total_resolved = sum(
        1 for s in scenario_reports
        if any(r.get('found') for r in s.root_cause_by_strategy.values())
    )
    total_ego_resolvable = sum(1 for s in scenario_reports if s.ego_resolvable)

    lines.append(f"- **Total Scenarios Analyzed**: {total_scenarios}")
    lines.append(f"- **Root Cause Identified**: {total_resolved} ({total_resolved/total_scenarios*100:.1f}%)")
    lines.append(f"- **Ego-Resolvable**: {total_ego_resolvable} ({total_ego_resolvable/total_scenarios*100:.1f}%)")
    lines.append("")

    if cross_tier.get('key_insights'):
        lines.append("### Key Insights\n")
        for insight in cross_tier['key_insights']:
            lines.append(f"- {insight}")
        lines.append("")

    # Dataset Overview
    lines.append("## 2. Dataset Overview\n")

    lines.append("### Scenarios by Risk Tier\n")
    lines.append("| Tier | Distance Range | Count | Ego-Resolvable |")
    lines.append("|------|---------------|-------|----------------|")

    for tier in [RiskTier.CRITICAL, RiskTier.HIGH, RiskTier.MODERATE, RiskTier.LOW]:
        if tier in tier_summaries:
            summary = tier_summaries[tier]
            bounds = RISK_TIER_THRESHOLDS[tier]
            lines.append(
                f"| {summary.tier_name} | {bounds[0]}-{bounds[1]}m | "
                f"{summary.num_scenarios} | {summary.ego_resolvable_fraction*100:.0f}% |"
            )
    lines.append("")

    # Typology distribution
    typology_counts = defaultdict(int)
    for s in scenario_reports:
        typology_counts[s.typology] += 1

    lines.append("### Scenarios by Typology\n")
    lines.append("| Typology | Count |")
    lines.append("|----------|-------|")
    for typology, count in sorted(typology_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {typology} | {count} |")
    lines.append("")

    # Strategy Comparison
    lines.append("## 3. Strategy Comparison Results\n")

    lines.append("### Resolution Rate by Strategy and Tier\n")

    # Build strategy list
    strategies = list(RankingStrategy)
    strategy_names = [s.value for s in strategies]

    # Header row
    header = "| Tier |" + "|".join(f" {s} " for s in strategy_names) + "|"
    separator = "|------|" + "|".join("-------" for _ in strategy_names) + "|"
    lines.append(header)
    lines.append(separator)

    for tier in [RiskTier.CRITICAL, RiskTier.HIGH, RiskTier.MODERATE, RiskTier.LOW]:
        if tier in tier_summaries:
            summary = tier_summaries[tier]
            row = f"| {summary.tier_name} |"
            for strat in strategy_names:
                rate = summary.resolution_rate_by_strategy.get(strat, 0)
                row += f" {rate*100:.0f}% |"
            lines.append(row)
    lines.append("")

    lines.append("### Average Search Cost (Agents Tested) by Strategy and Tier\n")
    lines.append(header)
    lines.append(separator)

    for tier in [RiskTier.CRITICAL, RiskTier.HIGH, RiskTier.MODERATE, RiskTier.LOW]:
        if tier in tier_summaries:
            summary = tier_summaries[tier]
            row = f"| {summary.tier_name} |"
            for strat in strategy_names:
                cost = summary.avg_search_cost_by_strategy.get(strat, 0)
                row += f" {cost:.1f} |"
            lines.append(row)
    lines.append("")

    # Best strategy by tier
    lines.append("### Best Strategy by Tier\n")
    lines.append("| Tier | Best Strategy | Resolution Rate |")
    lines.append("|------|---------------|-----------------|")
    for tier_name, data in cross_tier.get('best_strategy_by_tier', {}).items():
        lines.append(f"| {tier_name.title()} | {data['strategy']} | {data['resolution_rate']*100:.0f}% |")
    lines.append("")

    # Minimal Intervention Analysis
    lines.append("## 4. Minimal Intervention Analysis\n")

    lines.append("### Ego-Resolvability by Tier\n")
    lines.append("| Tier | Ego-Resolvable | Avg Speed Reduction | Avg Braking |")
    lines.append("|------|----------------|---------------------|-------------|")

    for tier in [RiskTier.CRITICAL, RiskTier.HIGH, RiskTier.MODERATE, RiskTier.LOW]:
        if tier in tier_summaries:
            summary = tier_summaries[tier]
            speed_str = f"{summary.avg_speed_reduction_needed:.0f}%" if summary.avg_speed_reduction_needed else "N/A"
            brake_str = f"{summary.avg_braking_decel_needed:.1f} m/s²" if summary.avg_braking_decel_needed else "N/A"
            lines.append(
                f"| {summary.tier_name} | {summary.ego_resolvable_fraction*100:.0f}% | "
                f"{speed_str} | {brake_str} |"
            )
    lines.append("")

    # Breakdown by resolution method
    speed_only = sum(1 for s in scenario_reports
                    if s.min_speed_reduction_pct and not s.min_braking_decel)
    brake_only = sum(1 for s in scenario_reports
                    if s.min_braking_decel and not s.min_speed_reduction_pct)
    both = sum(1 for s in scenario_reports
               if s.min_speed_reduction_pct and s.min_braking_decel)
    neither = sum(1 for s in scenario_reports
                  if not s.min_speed_reduction_pct and not s.min_braking_decel)

    lines.append("### Resolution Method Breakdown\n")
    lines.append(f"- Speed reduction sufficient: {speed_only + both} scenarios")
    lines.append(f"- Braking sufficient: {brake_only + both} scenarios")
    lines.append(f"- Both methods work: {both} scenarios")
    lines.append(f"- Neither sufficient (requires agent removal): {neither} scenarios")
    lines.append("")

    # Per-Scenario Details (Appendix)
    lines.append("## 5. Per-Scenario Details\n")
    lines.append("*Detailed analysis for each scenario.*\n")

    for i, scenario in enumerate(scenario_reports[:50], 1):  # Limit to first 50
        lines.append(f"### {i}. Scenario: {scenario.scenario_id}\n")
        lines.append(f"- **Typology**: {scenario.typology}")
        lines.append(f"- **Risk Tier**: {scenario.risk_tier}")
        lines.append(f"- **Min Distance**: {scenario.min_distance_m:.2f}m")
        lines.append(f"- **Min TTC**: {scenario.min_ttc_s:.2f}s")
        lines.append(f"- **Ego-Resolvable**: {'Yes' if scenario.ego_resolvable else 'No'}")

        # Use combined explanation if available, otherwise fall back to individual explanations
        if scenario.combined_explanation:
            lines.append(f"\n**Analysis:**\n```")
            lines.append(scenario.combined_explanation)
            lines.append("```")
        else:
            if scenario.root_cause_explanation:
                lines.append(f"\n**Root Cause**: {scenario.root_cause_explanation}")

            if scenario.intervention_explanation:
                lines.append(f"\n**Intervention**: {scenario.intervention_explanation}")

        lines.append("")

    if len(scenario_reports) > 50:
        lines.append(f"\n*... and {len(scenario_reports) - 50} more scenarios (see CSV for full details)*\n")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved: {output_path}")


def generate_csv_tables(
    scenario_reports: List[UnifiedScenarioReport],
    tier_summaries: Dict[RiskTier, TierSummary],
    output_dir: str
):
    """
    Generate CSV files for thesis tables:
    - scenarios_summary.csv: All scenarios with key metrics
    - strategy_comparison.csv: Strategy x Tier performance
    - intervention_requirements.csv: Ego intervention details
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. scenarios_summary.csv
    scenarios_csv_path = os.path.join(output_dir, 'scenarios_summary.csv')
    with open(scenarios_csv_path, 'w', newline='') as f:
        fieldnames = [
            'scenario_id', 'typology', 'risk_tier', 'min_distance_m', 'min_ttc_s',
            'closest_agent_id', 'closest_agent_type', 'ego_resolvable',
            'min_speed_reduction_pct', 'min_braking_profile', 'min_braking_decel',
            'semantic_resolved', 'semantic_agents_tested',
            'distance_resolved', 'distance_agents_tested',
            'ttc_resolved', 'ttc_agents_tested',
            'random_resolved', 'random_agents_tested'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for s in scenario_reports:
            row = {
                'scenario_id': s.scenario_id,
                'typology': s.typology,
                'risk_tier': s.risk_tier,
                'min_distance_m': s.min_distance_m,
                'min_ttc_s': s.min_ttc_s,
                'closest_agent_id': s.closest_agent_id,
                'closest_agent_type': s.closest_agent_type,
                'ego_resolvable': s.ego_resolvable,
                'min_speed_reduction_pct': s.min_speed_reduction_pct,
                'min_braking_profile': s.min_braking_profile,
                'min_braking_decel': s.min_braking_decel,
            }

            for strat in ['semantic', 'distance', 'ttc', 'random']:
                result = s.root_cause_by_strategy.get(strat, {})
                row[f'{strat}_resolved'] = result.get('found', False)
                row[f'{strat}_agents_tested'] = result.get('agents_tested', 0)

            writer.writerow(row)

    print(f"Saved: {scenarios_csv_path}")

    # 2. strategy_comparison.csv
    strategy_csv_path = os.path.join(output_dir, 'strategy_comparison.csv')
    with open(strategy_csv_path, 'w', newline='') as f:
        fieldnames = ['tier', 'tier_name', 'num_scenarios',
                     'semantic_rate', 'semantic_cost',
                     'distance_rate', 'distance_cost',
                     'ttc_rate', 'ttc_cost',
                     'random_rate', 'random_cost']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tier in [RiskTier.CRITICAL, RiskTier.HIGH, RiskTier.MODERATE, RiskTier.LOW]:
            if tier in tier_summaries:
                summary = tier_summaries[tier]
                row = {
                    'tier': tier.value,
                    'tier_name': summary.tier_name,
                    'num_scenarios': summary.num_scenarios,
                }
                for strat in ['semantic', 'distance', 'ttc', 'random']:
                    row[f'{strat}_rate'] = summary.resolution_rate_by_strategy.get(strat, 0)
                    row[f'{strat}_cost'] = summary.avg_search_cost_by_strategy.get(strat, 0)
                writer.writerow(row)

    print(f"Saved: {strategy_csv_path}")

    # 3. intervention_requirements.csv
    intervention_csv_path = os.path.join(output_dir, 'intervention_requirements.csv')
    with open(intervention_csv_path, 'w', newline='') as f:
        fieldnames = ['tier', 'tier_name', 'num_scenarios', 'ego_resolvable_pct',
                     'avg_speed_reduction_pct', 'avg_braking_decel']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tier in [RiskTier.CRITICAL, RiskTier.HIGH, RiskTier.MODERATE, RiskTier.LOW]:
            if tier in tier_summaries:
                summary = tier_summaries[tier]
                writer.writerow({
                    'tier': tier.value,
                    'tier_name': summary.tier_name,
                    'num_scenarios': summary.num_scenarios,
                    'ego_resolvable_pct': summary.ego_resolvable_fraction * 100,
                    'avg_speed_reduction_pct': summary.avg_speed_reduction_needed,
                    'avg_braking_decel': summary.avg_braking_decel_needed
                })

    print(f"Saved: {intervention_csv_path}")


def generate_json_summary(
    scenario_reports: List[UnifiedScenarioReport],
    tier_summaries: Dict[RiskTier, TierSummary],
    cross_tier: dict,
    output_path: str
):
    """Generate JSON summary for programmatic access."""
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_scenarios': len(scenario_reports),
        'tier_summaries': {
            tier.value: asdict(summary)
            for tier, summary in tier_summaries.items()
        },
        'cross_tier_comparison': cross_tier,
        'scenarios': [asdict(s) for s in scenario_reports]
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Saved: {output_path}")


def run_full_report(
    scenarios_dir: str = "scenarios",
    scenarios_csv: str = "scenarios.csv",
    output_dir: str = "thesis_results",
    collision_threshold: float = 2.0,
    safety_threshold: float = 2.5,
    max_scenarios: int = None,
    verbose: bool = True
):
    """
    Run the complete unified analysis and generate all reports.
    """
    print("\n" + "=" * 70)
    print("UNIFIED ANALYSIS REPORT GENERATOR")
    print("=" * 70)
    print(f"Scenarios Directory: {scenarios_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Collision Threshold: {collision_threshold}m")
    print(f"Safety Threshold: {safety_threshold}m")

    os.makedirs(output_dir, exist_ok=True)

    # Load scenarios with risk tiers
    print("\nLoading scenarios...")
    scenarios = load_scenarios_with_risk_tiers(scenarios_csv)
    by_tier = get_scenarios_by_tier(scenarios)

    total = sum(len(s) for s in by_tier.values())
    print(f"Found {total} scenarios within risk tier bounds")

    for tier in RiskTier:
        print(f"  {tier.value}: {len(by_tier[tier])} scenarios")

    # Generate per-scenario reports
    print("\nGenerating scenario reports...")
    scenario_reports = []

    all_scenarios = list(scenarios.values())
    if max_scenarios:
        all_scenarios = all_scenarios[:max_scenarios]

    for i, risk_profile in enumerate(all_scenarios):
        if verbose:
            print(f"  [{i+1}/{len(all_scenarios)}] {risk_profile.scenario_id}")

        report = generate_scenario_report(
            risk_profile.scenario_id,
            scenarios_dir=scenarios_dir,
            collision_threshold=collision_threshold,
            safety_threshold=safety_threshold,
            risk_profile=risk_profile,
            verbose=verbose
        )

        if report:
            scenario_reports.append(report)

    print(f"\nGenerated {len(scenario_reports)} scenario reports")

    # Group reports by tier
    reports_by_tier = {tier: [] for tier in RiskTier}
    for report in scenario_reports:
        try:
            tier = RiskTier(report.risk_tier)
            reports_by_tier[tier].append(report)
        except ValueError:
            pass  # Skip unknown tiers

    # Generate tier summaries
    print("\nGenerating tier summaries...")
    tier_summaries = {}
    for tier in RiskTier:
        tier_summaries[tier] = generate_tier_summary(tier, reports_by_tier[tier])
        if verbose:
            summary = tier_summaries[tier]
            print(f"  {tier.value}: {summary.num_scenarios} scenarios, "
                  f"ego-resolvable: {summary.ego_resolvable_fraction*100:.0f}%")

    # Generate cross-tier comparison
    print("\nGenerating cross-tier comparison...")
    cross_tier = generate_cross_tier_comparison(tier_summaries)

    if cross_tier.get('key_insights'):
        print("\nKey Insights:")
        for insight in cross_tier['key_insights']:
            print(f"  - {insight}")

    # Generate outputs
    print("\nGenerating output files...")

    # Markdown report
    markdown_path = os.path.join(output_dir, 'analysis_report.md')
    generate_markdown_report(scenario_reports, tier_summaries, cross_tier, markdown_path)

    # CSV tables
    generate_csv_tables(scenario_reports, tier_summaries, output_dir)

    # JSON summary
    json_path = os.path.join(output_dir, 'analysis_summary.json')
    generate_json_summary(scenario_reports, tier_summaries, cross_tier, json_path)

    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - {markdown_path}")
    print(f"  - {os.path.join(output_dir, 'scenarios_summary.csv')}")
    print(f"  - {os.path.join(output_dir, 'strategy_comparison.csv')}")
    print(f"  - {os.path.join(output_dir, 'intervention_requirements.csv')}")
    print(f"  - {json_path}")

    return scenario_reports, tier_summaries, cross_tier


# --- CLI ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate unified analysis report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full report with default settings
  python generate_report.py --output thesis_results/

  # Generate report for specific scenarios directory
  python generate_report.py --scenarios-dir scenarios/ --output thesis_results/

  # Quick test with limited scenarios
  python generate_report.py --output thesis_results/ --max-scenarios 5 --verbose
        """
    )

    parser.add_argument("--scenarios-dir", default="scenarios",
                        help="Directory containing USD scenario files")
    parser.add_argument("--scenarios-csv", default="scenarios.csv",
                        help="CSV file with scenario metadata")
    parser.add_argument("--output", default="thesis_results/",
                        help="Output directory for reports")
    parser.add_argument("--collision-threshold", type=float, default=2.0,
                        help="Collision detection threshold (meters)")
    parser.add_argument("--safety-threshold", type=float, default=2.5,
                        help="Target safety distance for interventions (meters)")
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="Maximum scenarios to process (for testing)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    run_full_report(
        scenarios_dir=args.scenarios_dir,
        scenarios_csv=args.scenarios_csv,
        output_dir=args.output,
        collision_threshold=args.collision_threshold,
        safety_threshold=args.safety_threshold,
        max_scenarios=args.max_scenarios,
        verbose=args.verbose
    )

    print("\nDone.")
