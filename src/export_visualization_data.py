"""
export_visualization_data.py - Export data for thesis figures

Generates JSON/CSV files formatted for common visualization tools:
- Plotly / Matplotlib charts
- LaTeX tables
- Thesis figures

Usage:
    python export_visualization_data.py --output figures_data/
    python export_visualization_data.py --output figures_data/ --format all
"""

import csv
import json
import os
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from graduated_risk_framework import (
    load_scenarios_with_risk_tiers, get_scenarios_by_tier,
    RiskTier, RISK_TIER_THRESHOLDS, RISK_TIER_CHARACTERISTICS,
    ScenarioRiskProfile
)
from evaluation import (
    load_existing_oracles, compute_metrics, compute_metrics_by_risk_tier,
    load_scenarios_metadata, OracleTier1Result, StrategyEvalResult,
    OUTPUT_DIR as EVAL_OUTPUT_DIR
)
from minimal_intervention import (
    compute_ego_intervention_summary, BRAKING_DECELERATION, BrakingProfile
)
from diagnosis_engine import RankingStrategy


# Strategy display names for figures
STRATEGY_DISPLAY_NAMES = {
    'semantic': 'Semantic Priority',
    'distance': 'Distance-Based',
    'ttc': 'TTC-Based',
    'random': 'Random Baseline'
}

# Tier display names
TIER_DISPLAY_NAMES = {
    'critical': 'Critical (1.5-2.5m)',
    'high': 'High (2.5-3.0m)',
    'moderate': 'Moderate (3.0-3.5m)',
    'low': 'Low (3.5-4.0m)'
}

# Color schemes for charts
STRATEGY_COLORS = {
    'semantic': '#2ecc71',    # Green
    'distance': '#3498db',    # Blue
    'ttc': '#e74c3c',         # Red
    'random': '#95a5a6'       # Gray
}

TIER_COLORS = {
    'critical': '#e74c3c',    # Red
    'high': '#f39c12',        # Orange
    'moderate': '#f1c40f',    # Yellow
    'low': '#2ecc71'          # Green
}


def export_strategy_comparison_chart_data(
    tier_metrics: dict,
    output_dir: str,
    metric_types: List[str] = None
):
    """
    Export data for bar chart comparing strategies across risk tiers.

    Args:
        tier_metrics: Output from compute_metrics_by_risk_tier()
        output_dir: Directory to save output files
        metric_types: List of metrics to export ['resolution_rate', 'mean_calls', 'mrr']

    Generates:
    - strategy_comparison_resolution.json: Resolution rate by tier and strategy
    - strategy_comparison_search_cost.json: Search cost by tier and strategy
    - strategy_comparison_mrr.json: MRR by tier and strategy
    - strategy_comparison.csv: Combined CSV for all metrics
    """
    if metric_types is None:
        metric_types = ['resolution_rate', 'mean_calls', 'mrr', 'top_1', 'top_3']

    os.makedirs(output_dir, exist_ok=True)

    tiers = ['critical', 'high', 'moderate', 'low']
    strategies = ['semantic', 'distance', 'ttc', 'random']

    # Build data structure for each metric
    for metric in metric_types:
        chart_data = {
            'title': f'Strategy Comparison: {metric.replace("_", " ").title()}',
            'x_axis': 'Risk Tier',
            'y_axis': metric.replace("_", " ").title(),
            'categories': [TIER_DISPLAY_NAMES.get(t, t) for t in tiers],
            'series': []
        }

        for strategy in strategies:
            series_data = {
                'name': STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
                'color': STRATEGY_COLORS.get(strategy, '#333333'),
                'values': []
            }

            for tier in tiers:
                tier_data = tier_metrics.get('per_tier', {}).get(tier, {})
                strat_data = tier_data.get('per_strategy', {}).get(strategy, {})
                value = strat_data.get(metric, 0)

                # Convert rate to percentage for display
                if metric in ['resolution_rate', 'top_1', 'top_3', 'top_5']:
                    value = value * 100

                series_data['values'].append(round(value, 2))

            chart_data['series'].append(series_data)

        # Save JSON
        json_path = os.path.join(output_dir, f'strategy_comparison_{metric}.json')
        with open(json_path, 'w') as f:
            json.dump(chart_data, f, indent=2)
        print(f"Saved: {json_path}")

    # Save combined CSV
    csv_path = os.path.join(output_dir, 'strategy_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['tier', 'tier_display', 'strategy', 'strategy_display',
                     'resolution_rate', 'mean_calls', 'mrr', 'top_1', 'top_3', 'num_scenarios']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tier in tiers:
            tier_data = tier_metrics.get('per_tier', {}).get(tier, {})
            for strategy in strategies:
                strat_data = tier_data.get('per_strategy', {}).get(strategy, {})
                writer.writerow({
                    'tier': tier,
                    'tier_display': TIER_DISPLAY_NAMES.get(tier, tier),
                    'strategy': strategy,
                    'strategy_display': STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
                    'resolution_rate': strat_data.get('resolution_rate', 0),
                    'mean_calls': strat_data.get('mean_calls', 0),
                    'mrr': strat_data.get('mrr', 0),
                    'top_1': strat_data.get('top_1', 0),
                    'top_3': strat_data.get('top_3', 0),
                    'num_scenarios': strat_data.get('num_scenarios', 0)
                })

    print(f"Saved: {csv_path}")

    # Generate Plotly-ready format
    plotly_data = {
        'resolution_rate': _generate_plotly_grouped_bar(
            tier_metrics, tiers, strategies, 'resolution_rate',
            'Resolution Rate by Strategy and Risk Tier', 'Resolution Rate (%)'
        ),
        'mean_calls': _generate_plotly_grouped_bar(
            tier_metrics, tiers, strategies, 'mean_calls',
            'Search Cost by Strategy and Risk Tier', 'Agents Tested'
        ),
        'mrr': _generate_plotly_grouped_bar(
            tier_metrics, tiers, strategies, 'mrr',
            'MRR by Strategy and Risk Tier', 'Mean Reciprocal Rank'
        )
    }

    plotly_path = os.path.join(output_dir, 'strategy_comparison_plotly.json')
    with open(plotly_path, 'w') as f:
        json.dump(plotly_data, f, indent=2)
    print(f"Saved: {plotly_path}")


def _generate_plotly_grouped_bar(
    tier_metrics: dict,
    tiers: List[str],
    strategies: List[str],
    metric: str,
    title: str,
    y_label: str
) -> dict:
    """Generate Plotly-compatible grouped bar chart data."""
    traces = []

    for strategy in strategies:
        y_values = []
        for tier in tiers:
            tier_data = tier_metrics.get('per_tier', {}).get(tier, {})
            strat_data = tier_data.get('per_strategy', {}).get(strategy, {})
            value = strat_data.get(metric, 0)
            if metric in ['resolution_rate', 'top_1', 'top_3', 'top_5']:
                value = value * 100
            y_values.append(round(value, 2))

        traces.append({
            'x': [TIER_DISPLAY_NAMES.get(t, t) for t in tiers],
            'y': y_values,
            'name': STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
            'type': 'bar',
            'marker': {'color': STRATEGY_COLORS.get(strategy, '#333333')}
        })

    return {
        'data': traces,
        'layout': {
            'title': title,
            'barmode': 'group',
            'xaxis': {'title': 'Risk Tier'},
            'yaxis': {'title': y_label},
            'legend': {'orientation': 'h', 'y': -0.2}
        }
    }


def export_intervention_requirements_chart_data(
    intervention_results: Dict[str, Dict],
    scenario_profiles: Dict[str, ScenarioRiskProfile],
    output_dir: str
):
    """
    Export data for intervention analysis charts.

    Args:
        intervention_results: Output from batch_analyze_scenarios()
        scenario_profiles: Scenario risk profiles for tier classification
        output_dir: Directory to save output files

    Generates:
    - ego_resolvable_by_tier.json: Pie/bar chart data for ego-resolvable by tier
    - speed_reduction_distribution.json: Histogram data for speed reductions
    - braking_profile_distribution.json: Bar chart for braking profiles by tier
    - intervention_requirements.csv: Combined data
    """
    os.makedirs(output_dir, exist_ok=True)

    tiers = ['critical', 'high', 'moderate', 'low']

    # Classify scenarios by tier
    by_tier = {tier: {'ego_resolvable': 0, 'non_resolvable': 0, 'total': 0,
                     'speed_reductions': [], 'braking_decels': [],
                     'braking_profiles': defaultdict(int)}
              for tier in tiers}

    for sid, data in intervention_results.items():
        # Get tier from profile
        profile = scenario_profiles.get(sid)
        if not profile:
            continue

        tier = profile.risk_tier
        if tier not in by_tier:
            continue

        by_tier[tier]['total'] += 1

        classification = data.get('classification', 'non_ego_resolvable')
        if classification == 'ego_resolvable':
            by_tier[tier]['ego_resolvable'] += 1

            # Track intervention details
            if data.get('min_intervention_type') == 'speed_reduction':
                val = data.get('min_intervention_value')
                if val:
                    by_tier[tier]['speed_reductions'].append(val)
            elif data.get('min_intervention_type') == 'braking':
                val = data.get('min_intervention_value')
                profile_name = data.get('braking_profile')
                if val:
                    by_tier[tier]['braking_decels'].append(val)
                if profile_name:
                    by_tier[tier]['braking_profiles'][profile_name] += 1
        else:
            by_tier[tier]['non_resolvable'] += 1

    # 1. Ego-resolvable by tier (stacked bar chart)
    ego_resolvable_data = {
        'title': 'Ego-Resolvability by Risk Tier',
        'x_axis': 'Risk Tier',
        'y_axis': 'Number of Scenarios',
        'categories': [TIER_DISPLAY_NAMES.get(t, t) for t in tiers],
        'series': [
            {
                'name': 'Ego-Resolvable',
                'color': '#2ecc71',
                'values': [by_tier[t]['ego_resolvable'] for t in tiers]
            },
            {
                'name': 'Non-Resolvable',
                'color': '#e74c3c',
                'values': [by_tier[t]['non_resolvable'] for t in tiers]
            }
        ],
        'percentages': {
            tier: round(by_tier[tier]['ego_resolvable'] / by_tier[tier]['total'] * 100, 1)
            if by_tier[tier]['total'] > 0 else 0
            for tier in tiers
        }
    }

    with open(os.path.join(output_dir, 'ego_resolvable_by_tier.json'), 'w') as f:
        json.dump(ego_resolvable_data, f, indent=2)
    print(f"Saved: {output_dir}/ego_resolvable_by_tier.json")

    # 2. Speed reduction distribution (histogram)
    all_speed_reductions = []
    for tier in tiers:
        all_speed_reductions.extend(by_tier[tier]['speed_reductions'])

    # Create histogram bins
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    histogram_counts = [0] * (len(bins) - 1)
    for val in all_speed_reductions:
        for i in range(len(bins) - 1):
            if bins[i] <= val < bins[i + 1]:
                histogram_counts[i] += 1
                break

    speed_reduction_data = {
        'title': 'Distribution of Required Speed Reductions',
        'x_axis': 'Speed Reduction (%)',
        'y_axis': 'Number of Scenarios',
        'bins': [f'{bins[i]}-{bins[i+1]}%' for i in range(len(bins) - 1)],
        'counts': histogram_counts,
        'statistics': {
            'count': len(all_speed_reductions),
            'mean': round(sum(all_speed_reductions) / len(all_speed_reductions), 2) if all_speed_reductions else 0,
            'min': round(min(all_speed_reductions), 2) if all_speed_reductions else 0,
            'max': round(max(all_speed_reductions), 2) if all_speed_reductions else 0
        },
        'by_tier': {
            tier: {
                'values': by_tier[tier]['speed_reductions'],
                'count': len(by_tier[tier]['speed_reductions']),
                'mean': round(sum(by_tier[tier]['speed_reductions']) / len(by_tier[tier]['speed_reductions']), 2)
                        if by_tier[tier]['speed_reductions'] else 0
            }
            for tier in tiers
        }
    }

    with open(os.path.join(output_dir, 'speed_reduction_distribution.json'), 'w') as f:
        json.dump(speed_reduction_data, f, indent=2)
    print(f"Saved: {output_dir}/speed_reduction_distribution.json")

    # 3. Braking profile distribution by tier
    braking_profiles = ['comfortable', 'firm', 'hard', 'emergency']

    braking_profile_data = {
        'title': 'Braking Profile Requirements by Risk Tier',
        'x_axis': 'Risk Tier',
        'y_axis': 'Number of Scenarios',
        'categories': [TIER_DISPLAY_NAMES.get(t, t) for t in tiers],
        'series': []
    }

    profile_colors = {
        'comfortable': '#2ecc71',
        'firm': '#f1c40f',
        'hard': '#e67e22',
        'emergency': '#e74c3c'
    }

    for profile in braking_profiles:
        braking_profile_data['series'].append({
            'name': profile.title(),
            'color': profile_colors.get(profile, '#333333'),
            'values': [by_tier[t]['braking_profiles'].get(profile, 0) for t in tiers]
        })

    with open(os.path.join(output_dir, 'braking_profile_distribution.json'), 'w') as f:
        json.dump(braking_profile_data, f, indent=2)
    print(f"Saved: {output_dir}/braking_profile_distribution.json")

    # 4. Combined CSV
    csv_path = os.path.join(output_dir, 'intervention_requirements.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['scenario_id', 'typology', 'risk_tier', 'min_distance_m',
                     'classification', 'intervention_type', 'intervention_value',
                     'braking_profile', 'achievable_margin_m']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sid, data in intervention_results.items():
            profile = scenario_profiles.get(sid)
            writer.writerow({
                'scenario_id': sid,
                'typology': profile.typology if profile else '',
                'risk_tier': profile.risk_tier if profile else '',
                'min_distance_m': profile.min_distance_m if profile else '',
                'classification': data.get('classification', ''),
                'intervention_type': data.get('min_intervention_type', ''),
                'intervention_value': data.get('min_intervention_value', ''),
                'braking_profile': data.get('braking_profile', ''),
                'achievable_margin_m': data.get('achievable_margin_m', '')
            })

    print(f"Saved: {csv_path}")


def export_dataset_summary_chart_data(
    scenario_profiles: Dict[str, ScenarioRiskProfile],
    output_dir: str
):
    """
    Export dataset summary data for overview charts.

    Generates:
    - dataset_by_tier.json: Scenario distribution by risk tier
    - dataset_by_typology.json: Scenario distribution by typology
    - dataset_heatmap.json: Tier x Typology heatmap data
    """
    os.makedirs(output_dir, exist_ok=True)

    tiers = ['critical', 'high', 'moderate', 'low']

    # Count by tier
    tier_counts = defaultdict(int)
    typology_counts = defaultdict(int)
    tier_typology_counts = defaultdict(lambda: defaultdict(int))

    for sid, profile in scenario_profiles.items():
        tier_counts[profile.risk_tier] += 1
        typology_counts[profile.typology] += 1
        tier_typology_counts[profile.risk_tier][profile.typology] += 1

    # 1. By tier (pie chart)
    tier_data = {
        'title': 'Scenarios by Risk Tier',
        'type': 'pie',
        'labels': [TIER_DISPLAY_NAMES.get(t, t) for t in tiers],
        'values': [tier_counts.get(t, 0) for t in tiers],
        'colors': [TIER_COLORS.get(t, '#333333') for t in tiers],
        'total': sum(tier_counts.values())
    }

    with open(os.path.join(output_dir, 'dataset_by_tier.json'), 'w') as f:
        json.dump(tier_data, f, indent=2)
    print(f"Saved: {output_dir}/dataset_by_tier.json")

    # 2. By typology (bar chart)
    typologies = sorted(typology_counts.keys(), key=lambda x: -typology_counts[x])

    typology_data = {
        'title': 'Scenarios by Typology',
        'x_axis': 'Typology',
        'y_axis': 'Number of Scenarios',
        'categories': typologies,
        'values': [typology_counts[t] for t in typologies],
        'total': sum(typology_counts.values())
    }

    with open(os.path.join(output_dir, 'dataset_by_typology.json'), 'w') as f:
        json.dump(typology_data, f, indent=2)
    print(f"Saved: {output_dir}/dataset_by_typology.json")

    # 3. Heatmap (tier x typology)
    heatmap_data = {
        'title': 'Scenario Distribution: Risk Tier × Typology',
        'x_labels': typologies,
        'y_labels': [TIER_DISPLAY_NAMES.get(t, t) for t in tiers],
        'values': [
            [tier_typology_counts[tier].get(typology, 0) for typology in typologies]
            for tier in tiers
        ],
        'raw_labels': {
            'x': typologies,
            'y': tiers
        }
    }

    with open(os.path.join(output_dir, 'dataset_heatmap.json'), 'w') as f:
        json.dump(heatmap_data, f, indent=2)
    print(f"Saved: {output_dir}/dataset_heatmap.json")


def export_latex_tables(
    tier_metrics: dict,
    intervention_summary: dict,
    scenario_profiles: Dict[str, ScenarioRiskProfile],
    output_dir: str
):
    """
    Generate LaTeX table code for thesis.

    Args:
        tier_metrics: Output from compute_metrics_by_risk_tier()
        intervention_summary: Output from compute_ego_intervention_summary()
        scenario_profiles: Scenario risk profiles
        output_dir: Directory to save output files

    Tables:
    1. Dataset summary (scenarios × typologies × tiers)
    2. Strategy comparison (resolution rate, MRR, search cost)
    3. Ego intervention requirements
    """
    os.makedirs(output_dir, exist_ok=True)

    tiers = ['critical', 'high', 'moderate', 'low']
    strategies = ['semantic', 'distance', 'ttc', 'random']

    # 1. Dataset Summary Table
    tier_counts = defaultdict(int)
    typology_counts = defaultdict(int)

    for profile in scenario_profiles.values():
        tier_counts[profile.risk_tier] += 1
        typology_counts[profile.typology] += 1

    dataset_table = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Dataset Summary by Risk Tier}",
        r"\label{tab:dataset-summary}",
        r"\begin{tabular}{lcrr}",
        r"\toprule",
        r"Risk Tier & Distance Range & Scenarios & Percentage \\",
        r"\midrule"
    ]

    total = sum(tier_counts.values())
    for tier in tiers:
        count = tier_counts.get(tier, 0)
        pct = count / total * 100 if total > 0 else 0
        bounds = RISK_TIER_THRESHOLDS[RiskTier(tier)]
        tier_display = tier.replace('_', ' ').title()
        dataset_table.append(
            f"{tier_display} & {bounds[0]}--{bounds[1]}m & {count} & {pct:.1f}\\% \\\\"
        )

    dataset_table.extend([
        r"\midrule",
        f"Total & -- & {total} & 100.0\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    with open(os.path.join(output_dir, 'table_dataset_summary.tex'), 'w') as f:
        f.write('\n'.join(dataset_table))
    print(f"Saved: {output_dir}/table_dataset_summary.tex")

    # 2. Strategy Comparison Table
    strategy_table = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Strategy Comparison by Risk Tier}",
        r"\label{tab:strategy-comparison}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Tier & Strategy & Resolution (\%) & MRR & Avg. Calls \\",
        r"\midrule"
    ]

    for tier in tiers:
        tier_data = tier_metrics.get('per_tier', {}).get(tier, {})
        tier_display = tier.replace('_', ' ').title()
        first_row = True

        for strategy in strategies:
            strat_data = tier_data.get('per_strategy', {}).get(strategy, {})
            resolution = strat_data.get('resolution_rate', 0) * 100
            mrr = strat_data.get('mrr', 0)
            calls = strat_data.get('mean_calls', 0)

            strat_display = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)

            if first_row:
                strategy_table.append(
                    f"{tier_display} & {strat_display} & {resolution:.1f} & {mrr:.3f} & {calls:.1f} \\\\"
                )
                first_row = False
            else:
                strategy_table.append(
                    f" & {strat_display} & {resolution:.1f} & {mrr:.3f} & {calls:.1f} \\\\"
                )

        strategy_table.append(r"\midrule")

    # Remove last midrule and add bottomrule
    strategy_table[-1] = r"\bottomrule"

    strategy_table.extend([
        r"\end{tabular}",
        r"\end{table}"
    ])

    with open(os.path.join(output_dir, 'table_strategy_comparison.tex'), 'w') as f:
        f.write('\n'.join(strategy_table))
    print(f"Saved: {output_dir}/table_strategy_comparison.tex")

    # 3. Ego Intervention Requirements Table
    intervention_table = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ego Intervention Requirements}",
        r"\label{tab:ego-intervention}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Metric & Value & Unit & Notes \\",
        r"\midrule"
    ]

    total_scenarios = intervention_summary.get('total_scenarios', 0)
    ego_count = intervention_summary.get('ego_resolvable_count', 0)
    ego_rate = intervention_summary.get('ego_resolvable_rate', 0) * 100
    avg_speed = intervention_summary.get('avg_speed_reduction_needed')
    avg_brake = intervention_summary.get('avg_braking_decel_needed')

    intervention_table.append(
        f"Total Scenarios & {total_scenarios} & -- & -- \\\\"
    )
    intervention_table.append(
        f"Ego-Resolvable & {ego_count} & ({ego_rate:.1f}\\%) & -- \\\\"
    )

    if avg_speed:
        intervention_table.append(
            f"Avg. Speed Reduction & {avg_speed:.1f} & \\% & For speed-resolvable \\\\"
        )

    if avg_brake:
        intervention_table.append(
            f"Avg. Braking & {avg_brake:.1f} & m/s\\textsuperscript{{2}} & For brake-resolvable \\\\"
        )

    # Braking profile breakdown
    profile_dist = intervention_summary.get('braking_profile_distribution', {})
    for profile in ['comfortable', 'firm', 'hard', 'emergency']:
        count = profile_dist.get(profile, 0)
        if count > 0:
            decel = BRAKING_DECELERATION.get(BrakingProfile(profile), 0)
            intervention_table.append(
                f"\\quad {profile.title()} Braking & {count} & ({decel} m/s\\textsuperscript{{2}}) & -- \\\\"
            )

    intervention_table.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    with open(os.path.join(output_dir, 'table_ego_intervention.tex'), 'w') as f:
        f.write('\n'.join(intervention_table))
    print(f"Saved: {output_dir}/table_ego_intervention.tex")

    # 4. Combined results table (resolution rate matrix)
    matrix_table = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Resolution Rate (\%) by Strategy and Risk Tier}",
        r"\label{tab:resolution-matrix}",
        r"\begin{tabular}{l" + "r" * len(strategies) + "}",
        r"\toprule",
        "Tier & " + " & ".join(STRATEGY_DISPLAY_NAMES.get(s, s) for s in strategies) + r" \\",
        r"\midrule"
    ]

    for tier in tiers:
        tier_data = tier_metrics.get('per_tier', {}).get(tier, {})
        tier_display = tier.replace('_', ' ').title()

        values = []
        for strategy in strategies:
            strat_data = tier_data.get('per_strategy', {}).get(strategy, {})
            resolution = strat_data.get('resolution_rate', 0) * 100
            values.append(f"{resolution:.1f}")

        matrix_table.append(f"{tier_display} & " + " & ".join(values) + r" \\")

    matrix_table.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    with open(os.path.join(output_dir, 'table_resolution_matrix.tex'), 'w') as f:
        f.write('\n'.join(matrix_table))
    print(f"Saved: {output_dir}/table_resolution_matrix.tex")


def export_matplotlib_scripts(output_dir: str):
    """
    Generate Python scripts for creating publication-quality figures.

    Generates ready-to-run Matplotlib/Seaborn scripts that read the exported data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Script for strategy comparison bar chart
    bar_chart_script = '''"""
Generated Matplotlib script for Strategy Comparison Bar Chart.
Run this script after exporting visualization data.
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('strategy_comparison_resolution_rate.json') as f:
    data = json.load(f)

# Setup
categories = data['categories']
x = np.arange(len(categories))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
for i, series in enumerate(data['series']):
    offset = (i - len(data['series'])/2 + 0.5) * width
    bars = ax.bar(x + offset, series['values'], width,
                  label=series['name'], color=series['color'])

# Labels and formatting
ax.set_xlabel(data['x_axis'])
ax.set_ylabel(data['y_axis'])
ax.set_title(data['title'])
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('strategy_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
'''

    with open(os.path.join(output_dir, 'plot_strategy_comparison.py'), 'w') as f:
        f.write(bar_chart_script)
    print(f"Saved: {output_dir}/plot_strategy_comparison.py")

    # Script for ego-resolvable stacked bar
    stacked_bar_script = '''"""
Generated Matplotlib script for Ego-Resolvability Stacked Bar Chart.
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('ego_resolvable_by_tier.json') as f:
    data = json.load(f)

categories = data['categories']
ego_resolvable = data['series'][0]['values']
non_resolvable = data['series'][1]['values']

x = np.arange(len(categories))
width = 0.6

fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x, ego_resolvable, width, label='Ego-Resolvable',
               color=data['series'][0]['color'])
bars2 = ax.bar(x, non_resolvable, width, bottom=ego_resolvable,
               label='Non-Resolvable', color=data['series'][1]['color'])

# Add percentage labels
for i, (e, n) in enumerate(zip(ego_resolvable, non_resolvable)):
    total = e + n
    if total > 0:
        pct = e / total * 100
        ax.annotate(f'{pct:.0f}%', xy=(i, e/2), ha='center', va='center',
                   color='white', fontweight='bold')

ax.set_xlabel(data['x_axis'])
ax.set_ylabel(data['y_axis'])
ax.set_title(data['title'])
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('ego_resolvable.pdf', dpi=300, bbox_inches='tight')
plt.savefig('ego_resolvable.png', dpi=300, bbox_inches='tight')
plt.show()
'''

    with open(os.path.join(output_dir, 'plot_ego_resolvable.py'), 'w') as f:
        f.write(stacked_bar_script)
    print(f"Saved: {output_dir}/plot_ego_resolvable.py")


def run_full_export(
    output_dir: str = "figures_data",
    scenarios_csv: str = "scenarios.csv",
    verbose: bool = True
):
    """
    Run full visualization data export pipeline.
    """
    print("\n" + "=" * 70)
    print("VISUALIZATION DATA EXPORT")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Load scenario profiles
    print("\nLoading scenario data...")
    scenario_profiles = load_scenarios_with_risk_tiers(scenarios_csv)
    print(f"Loaded {len(scenario_profiles)} scenarios")

    # Try to load existing evaluation results
    tier_metrics = None
    eval_results_path = os.path.join(EVAL_OUTPUT_DIR, 'risk_tier_metrics.json')
    if os.path.exists(eval_results_path):
        print(f"Loading tier metrics from {eval_results_path}")
        with open(eval_results_path) as f:
            tier_metrics = json.load(f)

    # Try to load intervention results
    intervention_results = None
    intervention_path = os.path.join(EVAL_OUTPUT_DIR, 'minimal_interventions.json')
    if os.path.exists(intervention_path):
        print(f"Loading intervention results from {intervention_path}")
        with open(intervention_path) as f:
            intervention_results = json.load(f)

    # Export dataset summary
    print("\n--- Exporting Dataset Summary ---")
    export_dataset_summary_chart_data(scenario_profiles, output_dir)

    # Export strategy comparison (if available)
    if tier_metrics:
        print("\n--- Exporting Strategy Comparison ---")
        export_strategy_comparison_chart_data(tier_metrics, output_dir)
    else:
        print("\nSkipping strategy comparison (no tier metrics available)")

    # Export intervention data (if available)
    if intervention_results:
        print("\n--- Exporting Intervention Requirements ---")
        export_intervention_requirements_chart_data(
            intervention_results, scenario_profiles, output_dir
        )

        # Compute summary for LaTeX tables
        intervention_summary = compute_ego_intervention_summary(intervention_results)
    else:
        print("\nSkipping intervention charts (no intervention results available)")
        intervention_summary = {}

    # Export LaTeX tables
    if tier_metrics or intervention_summary:
        print("\n--- Exporting LaTeX Tables ---")
        export_latex_tables(
            tier_metrics or {},
            intervention_summary,
            scenario_profiles,
            output_dir
        )

    # Export Matplotlib scripts
    print("\n--- Exporting Matplotlib Scripts ---")
    export_matplotlib_scripts(output_dir)

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nAll files saved to: {output_dir}/")

    return {
        'output_dir': output_dir,
        'scenarios_exported': len(scenario_profiles),
        'has_tier_metrics': tier_metrics is not None,
        'has_intervention_data': intervention_results is not None
    }


# --- CLI ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export data for thesis figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all visualization data
  python export_visualization_data.py --output figures_data/

  # Export with verbose output
  python export_visualization_data.py --output figures_data/ --verbose
        """
    )

    parser.add_argument("--output", default="figures_data/",
                        help="Output directory for visualization data")
    parser.add_argument("--scenarios-csv", default="scenarios.csv",
                        help="Path to scenarios CSV file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    run_full_export(
        output_dir=args.output,
        scenarios_csv=args.scenarios_csv,
        verbose=args.verbose
    )

    print("\nDone.")
