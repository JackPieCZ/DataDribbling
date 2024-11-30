# import pandas as pd


def kelly_criterion(probability, odds, bankroll, fraction):
    """
    Vypočítá optimální výši sázky pomocí Kellyho kritéria.

    :param probability: odhadovaná pravděpodobnost výhry (0 až 1).
    :param odds: kurz
    :param bankroll: dostupný kapitál
    :param fraction: frakční Kelly (např. 0.5 pro poloviční Kelly).
    :return: doporučená výše sázky.
    """
    q = 1 - probability
    b = odds - 1  # zisk
    optimal_fraction = probability - (q / b)
    optimal_bet = bankroll * optimal_fraction * fraction

    optimal_bet = max(0, optimal_bet)  # nevsázet, pokud je Kelly záporný
    return optimal_bet


bankroll = 10000
probability = 0.6
odds = 2.0
fraction = 1  # plný Kelly (napr. 0.5 pro poloviční Kelly)

optimal_bet = kelly_criterion(probability, odds, bankroll, fraction)
print(f"Doporučená sázka: {optimal_bet:.2f} Kč")
