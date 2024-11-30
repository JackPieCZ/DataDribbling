import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Union


def evaluate_strategy(bankroll_history: Union[list, np.ndarray]) -> float:
    """
    Evaluates a betting strategy based on its bankroll history.

    Args:
        bankroll_history: Array-like object containing bankroll values over time

    Returns:
        float: Fitness score for the strategy
    """
    bankrolls = np.array(bankroll_history)
    initial_bankroll = 1000

    # Early exit with severe penalty if bankroll drops below critical threshold
    # if np.min(bankrolls) < 300:
    #     return -100

    # Calculate returns
    returns = np.diff(bankrolls) / bankrolls[:-1]  # Percentage returns

    # 1. Basic Performance Metrics
    final_bankroll = bankrolls[-1]
    total_return = (final_bankroll - initial_bankroll) / initial_bankroll

    # 2. Risk Metrics
    # Calculate maximum drawdown
    peak = np.maximum.accumulate(bankrolls)
    drawdowns = (peak - bankrolls) / peak
    max_drawdown = np.max(drawdowns)

    # Calculate volatility (annualized, assuming daily data)
    volatility = np.std(returns) * np.sqrt(252)

    # 3. Risk-Adjusted Returns
    # Calculate Sharpe ratio (annualized, assuming risk-free rate of 0.02)
    excess_returns = returns - 0.02 / 252  # Daily risk-free rate
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    # 4. Consistency Metrics
    # Calculate recovery factor
    recovery_factor = total_return / max_drawdown if max_drawdown > 0 else total_return

    # Calculate return consistency using skewness and kurtosis
    return_skew = skew(returns)
    return_kurt = kurtosis(returns, fisher=False)  # Excess kurtosis

    # 5. Growth Trajectory
    # Calculate the growth trend using linear regression
    time_indices = np.arange(len(bankrolls))
    growth_coefficient = np.polyfit(time_indices, bankrolls, 1)[0]

    print(
        f"Parameters without weights and normalization: {final_bankroll=}, {sharpe=}, {max_drawdown=}, {recovery_factor=}, {growth_coefficient=}, {return_skew=}, {return_kurt=} {volatility=}"
    )

    # 6. Composite Scoring
    # Weight the components based on importance
    weights = {
        "final_bankroll": 1.0,
        "sharpe": 0.0,
        "max_drawdown": -2.0,
        "recovery_factor": 0.0,
        "growth_coef": 1.0,
        "return_skew": 0.0,  # Prefer positive skew
        "consistency": 0.0,  # Penalize high kurtosis (fat tails)
        "volatility": -2.0,
    }

    # Normalize the metrics to similar scales
    normalized_metrics = {
        "final_bankroll": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "recovery_factor": recovery_factor,
        "growth_coef": growth_coefficient,
        "return_skew": return_skew,
        # "consistency": return_kurt,
        # "volatility": volatility,
        # "sharpe": np.clip(sharpe, -3, 3) / 3,  # Clip extreme values
        # "max_drawdown": np.clip(max_drawdown, 0, 10),
        # "recovery_factor": np.clip(recovery_factor, 0, 10) / 10,
        # "growth_coef": np.clip(growth_coefficient, -30, 30) / 30,
        # "return_skew": np.clip(return_skew, -3, 3) / 3,
        "consistency": np.clip(return_kurt, -1, 15) / 15,
        "volatility": np.clip(volatility, 0, 10),
    }
    print(f"Parameters with weights and normalization: {normalized_metrics=}")
    print()
    print()

    # Calculate final fitness score
    fitness_score = sum(weights[key] * normalized_metrics[key] for key in weights.keys())

    # Add survival bonus for maintaining healthy bankroll
    min_bankroll_ratio = np.min(bankrolls) / initial_bankroll
    if min_bankroll_ratio > 0.5:  # Bonus for never dropping below 50%
        fitness_score *= 1.2

    return fitness_score


import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class BettingScenario:
    name: str
    bankroll_history: np.ndarray
    expected_performance: str


def generate_synthetic_data(days: int = 365, initial_bankroll: float = 1000) -> List[BettingScenario]:
    """
    Generates 10 diverse synthetic bankroll histories to test the evaluation function.

    Args:
        days: Number of days to simulate
        initial_bankroll: Starting bankroll amount

    Returns:
        List of BettingScenario objects containing different patterns
    """
    t = np.linspace(0, days/365, days)
    scenarios = []

    # # 1. Conservative Steady Growth (low risk, modest returns)
    conservative = initial_bankroll * (1 + 0.3 * t + 0.03 * np.random.randn(days))
    scenarios.append(
        BettingScenario(
            name="Conservative Growth",
            bankroll_history=conservative,
            expected_performance="Stable but modest growth with minimal volatility",
        )
    )

    # 2. Aggressive Steady Growth (higher risk, higher returns)
    aggressive = initial_bankroll * (1 + 3 * t + 0.15 * np.random.randn(days))
    scenarios.append(
        BettingScenario(
            name="Aggressive Growth",
            bankroll_history=aggressive,
            expected_performance="Strong consistent growth with moderate volatility",
        )
    )

    # 3. High Volatility Growth (very high risk, potential high returns)
    volatile = initial_bankroll * (1 + 4 * t + 0.5 * np.random.randn(days))
    scenarios.append(
        BettingScenario(
            name="High Volatility Growth",
            bankroll_history=volatile,
            expected_performance="Excellent growth but with significant volatility",
        )
    )

    # 4. Seasonal Cyclic (periodic patterns)
    seasonal = initial_bankroll * (1 + 1.5 * t + 0.4 * np.sin(8 * np.pi * t) + 0.1 * np.random.randn(days))
    scenarios.append(
        BettingScenario(
            name="Seasonal Pattern",
            bankroll_history=seasonal,
            expected_performance="Growth with regular seasonal fluctuations",
        )
    )

    # 5. Recovery After Drawdown
    # recovery = np.zeros(days)
    # recovery[: days // 3] = initial_bankroll * (1 - 0.3 * t[: days // 3] + 0.05 * np.random.randn(days // 3))
    # recovery[days // 3 :] = recovery[days // 3 - 1] * (
    #     1 + 2 * t[: (2 * days // 3)] + 0.1 * np.random.randn(2 * days // 3)
    # )
    # scenarios.append(
    #     BettingScenario(
    #         name="Strong Recovery",
    #         bankroll_history=recovery,
    #         expected_performance="Initial losses followed by strong recovery",
    #     )
    # )

    # 6. Gradual Decline
    decline = initial_bankroll * (1 - 0.9 * t + 0.05 * np.random.randn(days))
    scenarios.append(
        BettingScenario(
            name="Gradual Decline", bankroll_history=decline, expected_performance="Steady decline with low volatility"
        )
    )

    # 7. Exponential Growth
    exponential = initial_bankroll * np.exp(1.5 * t + 0.1 * np.random.randn(days))
    scenarios.append(
        BettingScenario(
            name="Exponential Growth",
            bankroll_history=exponential,
            expected_performance="Accelerating returns with increasing volatility",
        )
    )

    # 8. Market Crash Recovery
    # crash = np.zeros(days)
    # crash_point = days // 4
    # crash[:crash_point] = initial_bankroll * (1 + 2 * t[:crash_point] + 0.1 * np.random.randn(crash_point))
    # crash[crash_point+1:] = crash[crash_point - 1] * (
    #     0.6 + 1.5 * t[: (3 * days // 4)] + 0.2 * np.random.randn(3 * days // 4)
    # )
    # scenarios.append(
    #     BettingScenario(
    #         name="Crash Recovery",
    #         bankroll_history=crash,
    #         expected_performance="Strong start, sudden crash, followed by recovery",
    #     )
    # )

    # 9. Stepped Growth (periods of stability followed by jumps)
    steps = np.zeros(days)
    for i in range(4):
        section = days // 4
        base = initial_bankroll * (1 + 0.5 * i)
        steps[i * section : (i + 1) * section] = base * (1 + 0.1 * t[:section] + 0.05 * np.random.randn(section))
    scenarios.append(
        BettingScenario(
            name="Stepped Growth",
            bankroll_history=steps,
            expected_performance="Periods of stability interrupted by sudden gains",
        )
    )

    # 10. High-Frequency Trading Pattern
    t_fine = np.linspace(0, 100, days)  # More cycles
    hft = initial_bankroll * (1 + 2 * t + 0.1 * np.sin(t_fine) + 0.2 * np.random.randn(days))
    scenarios.append(
        BettingScenario(
            name="High-Frequency Pattern",
            bankroll_history=hft,
            expected_performance="Rapid oscillations with underlying growth trend",
        )
    )

    return scenarios


def analyze_and_plot_scenarios():
    """
    Generates synthetic data, evaluates it, and creates visualizations.
    """
    # Generate scenarios
    scenarios = generate_synthetic_data()

    # Evaluate each scenario
    results = []
    for scenario in scenarios:
        print(f"Evaluating scenario: {scenario.name}")
        fitness_score = evaluate_strategy(scenario.bankroll_history)
        results.append(
            {
                "Scenario": scenario.name,
                "Final Bankroll": scenario.bankroll_history[-1],
                "Fitness Score": fitness_score,
                "Expected Performance": scenario.expected_performance,
            }
        )

    results_df = pd.DataFrame(results)

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot bankroll histories
    plt.subplot(2, 1, 1)
    for scenario in scenarios:
        plt.plot(scenario.bankroll_history, label=scenario.name)
    plt.axhline(y=300, color="r", linestyle="--", label="Critical Threshold")
    plt.legend()
    plt.title("Bankroll History by Scenario")
    plt.ylabel("Bankroll")

    # Plot fitness scores
    plt.subplot(2, 1, 2)
    plt.bar(results_df["Scenario"], results_df["Fitness Score"])
    plt.xticks(rotation=45)
    plt.title("Fitness Scores by Scenario")
    plt.ylabel("Fitness Score")

    plt.tight_layout()

    return results_df, plt


# Example usage:
if __name__ == "__main__":
    results_df, plot = analyze_and_plot_scenarios()
    print("\nEvaluation Results:")
    print(results_df[["Scenario", "Final Bankroll", "Fitness Score", "Expected Performance"]])
    plt.show()
