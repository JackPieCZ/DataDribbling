import os
from collections import deque
import threading
from time import time
import matplotlib.pyplot as plt
from environment import Environment
from model_old import Model, Net
from random import choices, randint
from typing import Union
import numpy as np
from deap import creator, base, tools
import pandas as pd
import torch
import sys
sys.path.append("..")


class Evaluator:
    def __init__(self) -> None:
        self.games = pd.read_csv(r"C:\FEL\QQH2024\testing\data\games.csv", index_col=0)
        self.games["Date"] = pd.to_datetime(self.games["Date"])
        self.games["Open"] = pd.to_datetime(self.games["Open"])

        self.players = pd.read_csv(r"C:\FEL\QQH2024\testing\data\players.csv", index_col=0)
        self.players["Date"] = pd.to_datetime(self.players["Date"])

        self.season_starts = {
            1: "1975-11-07",
            2: "1976-11-12",
            3: "1977-11-11",
            4: "1978-11-10",
            5: "1979-11-09",
            6: "1980-11-07",
            7: "1981-11-13",
            8: "1982-11-12",
            9: "1983-11-11",
            10: "1984-11-09",
            11: "1985-11-08",
            12: "1986-11-07",
            13: "1988-02-12",
            14: "1988-11-08",
            15: "1989-11-07",
            16: "1990-11-06",
            17: "1991-11-05",
            18: "1992-11-03",
            19: "1993-11-09",
            20: "1994-11-08",
            21: "1995-11-07",
            22: "1996-11-05",
            23: "1997-11-04",
            24: "1998-11-03",
        }

    def evaluate(self, net, start_season=4, num_seasons=5) -> Union[float, pd.DataFrame]:
        env = Environment(
            self.games, self.players, Model(net), init_bankroll=1000, min_bet=5, max_bet=100,
            start_date=pd.Timestamp(self.season_starts.get(start_season, "1976-11-12")),
            end_date=pd.Timestamp(self.season_starts.get(start_season + num_seasons, "1983-11-11"))
        )

        evaluation = env.run()
        print(f"Final bankroll: {env.bankroll:.2f}")
        return env.bankroll, env.get_history()


def create_nn(individual):
    neural_network = Net()
    state_dict = neural_network.state_dict()
    param_shapes = [param.shape for param in state_dict.values()]

    reshaped_params = []
    start = 0
    for shape in param_shapes:
        size = torch.prod(torch.tensor(shape)).item()
        reshaped_tensor = torch.tensor(individual[start:start + size]).view(shape)
        reshaped_params.append(reshaped_tensor)
        start += size

    state_dict = {k: v for k, v in zip(state_dict.keys(), reshaped_params)}
    neural_network.load_state_dict(state_dict)
    return neural_network


def compute_fitness(individual, start_season, num_seasons, history_best_bankrolls) -> float:
    net = create_nn(individual)
    evaluator = Evaluator()
    final_bankroll, bankroll_history = evaluator.evaluate(net, start_season=start_season, num_seasons=num_seasons)

    bankrolls = np.array(bankroll_history["Bankroll"])
    history_best_bankrolls.append(bankroll_history["Bankroll"].tolist())
    initial_bankroll = 1000
    returns = np.diff(bankrolls) / bankrolls[:-1]  # Percentage returns

    # Early exit with severe penalty if bankroll drops below critical threshold
    # if np.min(bankrolls) < 300:
    #     return -100

    # 1. Basic Performance Metrics
    total_return = (final_bankroll - initial_bankroll) / initial_bankroll
    max_return = (np.max(bankrolls) - initial_bankroll) / initial_bankroll

    # 2. Risk Metrics
    peak = np.maximum.accumulate(bankrolls)
    drawdowns = (peak - bankrolls) / peak
    max_drawdown = np.max(drawdowns)

    # 3. Volatility (annualized, assuming daily data)
    volatility = np.std(returns) * np.sqrt(252)

    # 4. Growth Trajectory
    growth_coefficient = np.polyfit(np.arange(len(bankrolls)), bankrolls, 1)[0]

    # 6. Composite Scoring
    weights = {
        "final_bankroll": 2.0,
        "max_return": 2.0,
        "max_drawdown": 0.0,
        "growth_coef": 1.0,
        "volatility": 0,
    }

    normalized_metrics = {
        "final_bankroll": total_return,
        "max_return": max_return,
        "max_drawdown": max_drawdown,
        "growth_coef": growth_coefficient,
        "volatility": np.clip(volatility, 0, 10) / 5,
    }

    # Calculate final fitness score
    fitness_score = sum(weights[key] * normalized_metrics[key] for key in weights.keys())

    # Add survival bonus for maintaining healthy bankroll
    # min_bankroll_ratio = np.min(bankrolls) / initial_bankroll
    # if min_bankroll_ratio > 0.5:
    #     fitness_score *= 1.2

    return fitness_score,


def init_population(init_population_size):
    population = []
    for i in range(init_population_size):
        net = Net()
        weights = torch.cat([p.flatten() for p in net.parameters()]).tolist()
        individual = creator.Individual(weights)
        population.append(individual)
    return population


def mutation(ind, mutation_rate, strength=0.1):
    for i in range(len(ind)):
        if np.random.rand() < mutation_rate:
            ind[i] += np.random.normal(0, strength)


# def blx_alpha_crossover(parent1, parent2, alpha=0.5):
#     """
#     Performs BLX-alpha crossover between two parents.

#     Args:
#         parent1: First parent's weights
#         parent2: Second parent's weights
#         alpha: Blending range parameter, typically between 0.0-1.0

#     Returns:
#         Tuple of two offspring created by blending the parents
#     """
#     # Convert parents to numpy arrays for easier math
#     p1 = np.array(parent1)
#     p2 = np.array(parent2)

#     # Calculate range for each gene
#     min_vals = np.minimum(p1, p2)
#     max_vals = np.maximum(p1, p2)
#     range_vals = max_vals - min_vals

#     # Calculate extended bounds with alpha
#     lower_bounds = min_vals - alpha * range_vals
#     upper_bounds = max_vals + alpha * range_vals

#     # Generate random values within extended bounds
#     child1 = np.random.uniform(lower_bounds, upper_bounds)
#     child2 = np.random.uniform(lower_bounds, upper_bounds)

#     return creator.Individual(child1), creator.Individual(child2)


def uniform_crossover(parent1, parent2, rate=0.5):
    """
    Performs uniform crossover between two parents.

    Args:
        parent1: First parent's weights
        parent2: Second parent's weights
        rate: Probability of inheriting gene from parent1

    Returns:
        Tuple of two offspring created by uniform crossover
    """
    # Convert parents to numpy arrays for easier math
    p1 = np.array(parent1)
    p2 = np.array(parent2)

    # Initialize children as copies of parents
    child1 = p1.copy()
    child2 = p2.copy()

    # Randomly select genes to inherit from each parent
    for i in range(len(p1)):
        if np.random.rand() < rate:
            child1[i] = p2[i]
            child2[i] = p1[i]

    return creator.Individual(child1), creator.Individual(child2)
def plot_evolution(logbook, history_best_bankrolls):
    generations = logbook.select("generation")
    mean_fitness = logbook.select("mean")
    std_fitness = logbook.select("std")
    max_fitness = logbook.select("max")
    min_fitness = logbook.select("min")

    # Plot fitness metrics
    plt.figure(figsize=(12, 6))
    plt.plot(generations, mean_fitness, label="Mean Fitness")
    plt.fill_between(
        generations,
        np.array(mean_fitness) - np.array(std_fitness),
        np.array(mean_fitness) + np.array(std_fitness),
        alpha=0.2,
        label="Std Dev",
    )
    plt.plot(generations, max_fitness, label="Max Fitness", linestyle="--")
    plt.plot(generations, min_fitness, label="Min Fitness", linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Evolution Over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot bankroll history of the best individual per generation
    plt.figure(figsize=(12, 6))
    for gen_idx, bankroll_history in enumerate(history_best_bankrolls):
        plt.plot(
            range(len(bankroll_history)),
            bankroll_history,
            label=f"Gen {gen_idx + 1}" if gen_idx % 10 == 0 else "",
            alpha=0.6 if gen_idx % 10 == 0 else 0.2,
        )
    plt.xlabel("Days")
    plt.ylabel("Bankroll")
    plt.title("Bankroll Evolution of Best Individuals Across Generations")
    plt.legend()
    plt.grid(True)
    plt.show()


def thread_evaluate(deque_population, deque_lock, toolbox, thrad_id, start_season, seasons_to_evaluate, history_best_bankrolls):
    while True:
        with deque_lock:
            if len(deque_population) == 0:
                return

            individual = deque_population.popleft()
        individual.fitness.values = toolbox.evaluate(individual, start_season, seasons_to_evaluate, history_best_bankrolls)


if __name__ == "__main__":
    np.random.seed(int(time()))
    # np.random.seed(42)
    generations_num = 50
    population_size = 20
    selection_ratio = 0.4
    elite_ratio = 0.1
    mutation_prob = 0.2
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    population = init_population(population_size)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", compute_fitness)
    toolbox.register("mutate", mutation)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("mate", uniform_crossover)
    # toolbox.register("map", pool.map)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["generation", "mean", "std", "min", "max"]

    history_best_bankrolls = []  # Store bankroll history of best individuals
    good_start = False
    while not good_start:
        for gen_idx in range(generations_num):
            print(f"Generation {gen_idx+1}/{generations_num}")
            tournament_size = max(2, int(3 + 2 * gen_idx / generations_num))
            toolbox.register("select", tools.selTournament, tournsize=tournament_size)

            seasons_to_evaluate = (gen_idx // 10 + 1) * 2
            if gen_idx == 0:
                start_season = randint(3, 24-seasons_to_evaluate)
            # if gen_idx % 25 == 0:
            #     start_season = randint(3, 24-seasons_to_evaluate)
            # Step 1: Evaluate fitness of each individual

            for inv_id, individual in enumerate(population):
                individual.fitness.values = toolbox.evaluate(individual, start_season, seasons_to_evaluate, history_best_bankrolls)
                print(f"Individual {inv_id+1}, fitness: {individual.fitness.values}")

            # fitness_values = toolbox.map(lambda ind: toolbox.evaluate(ind, start_season, seasons_to_evaluate, history_best_bankrolls), population)
            # for ind, fit in zip(population, fitness_values):
            #     ind.fitness.values = fit

            # threads = []
            # threads_number = 1
            # deque_population = deque(population)

            # deque_lock = threading.Lock()
            # for thread_id in range(threads_number):
            #     thread = threading.Thread(target=thread_evaluate, args=(deque_population, deque_lock, toolbox,
            #                               thread_id, start_season, seasons_to_evaluate, history_best_bankrolls))
            #     threads.append(thread)
            #     thread.start()

            # for thread in threads:
            #     thread.join()

            # Record stats for this generation
            record = stats.compile(population)
            logbook.record(generation=gen_idx, **record)
            print(logbook.stream)

            # Save the best individuals from the current population
            number_best = 5
            best_individuals = tools.selBest(population, number_best)
            for i in range(len(best_individuals)):
                if best_individuals[i].fitness.values[0] <= 0:
                    continue
                best_nn = create_nn(best_individuals[i])
                current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                folder = f'./{current_time}_season{start_season}'
                os.makedirs(folder, exist_ok=True)
                torch.save(best_nn.state_dict(), f'{folder}/bestnet_gen{gen_idx}_top{i}_fit{int(best_individuals[i].fitness.values[0])}.pth')

            # best_individual = tools.selBest(population, 1)[0]
            # # if best_individual.fitness.values[0] <= 0:
            # # continue
            # best_nn = create_nn(best_individual)
            # torch.save(best_nn.state_dict(), f'./bestnet_gen{gen_idx}_fit{int(best_individual.fitness.values[0])}.pth')

            new_population = []
            # Step 2: Population Weights Selection
            preselected_indivs = toolbox.select(population, int(selection_ratio * population_size))
            selected_indivs = []
            for indiv in selected_indivs:
                print(f"Selected fitness: {indiv.fitness.values}")
                if indiv.fitness.values[0] > 0:
                    new_population.append(indiv)

            elites = tools.selBest(population, int(elite_ratio * population_size))
            for indiv in elites:
                print(f"Elite fitness: {indiv.fitness.values}")
                if indiv.fitness.values[0] > 0:
                    new_population.append(indiv)
            max_fitness = max([indiv.fitness.values[0] for indiv in population])
            if max_fitness <= 0 and gen_idx == 0:
                print("Bad start")
                break
            elif max_fitness > 0 and gen_idx == 0:
                print("Good start")
                good_start = True


            # Step 3: Generate offspring with mating and mutation
            offspring = []
            # # Get fitness values for selected individuals
            fitness_values = np.array([ind.fitness.values[0] for ind in selected_indivs])
            weights = np.where(fitness_values < 0, 1e-10, fitness_values)
            # Ensure non-negative weights by shifting up if needed

            # while len(offspring) + len(elites) < population_size:
            #     # Select pairs of parents from the best individuals for crossover
            #     parent1, parent2 = choices(selected_indivs, weights=weights, k=2)
            #     # Apply crossover and mutation
            #     child1, child2 = toolbox.mate(parent1, parent2)
            #     toolbox.mutate(child1, mutation_prob)
            #     toolbox.mutate(child2, mutation_prob)
            #     del child1.fitness.values, child2.fitness.values
            #     offspring.extend([child1, child2])

            # Generate offspring with mutation only
            if len(selected_indivs) > 0:
                while len(offspring) + len(elites) < population_size * 0.75:
                    parent = choices(selected_indivs, k=1, weights=weights)[0]
                    child = toolbox.clone(parent)
                    toolbox.mutate(child, mutation_prob)
                    del child.fitness.values
                    offspring.append(child)

                # Combine elites and offspring for the new population
                new_population.extend(offspring[:population_size - len(elites)])

            while len(new_population) < population_size:
                net = Net()
                weights = torch.cat([p.flatten() for p in net.parameters()]).tolist()
                individual = creator.Individual(weights)
                new_population.append(individual)
            population = new_population
            assert len(population) == population_size

    # for indiv in population:
    #     indiv.fitness.values = toolbox.evaluate(indiv)

    # Output the best individual from the final population
    best_individual = tools.selBest(population, 1)[0]
    print(f"Best individual fitness: {best_individual.fitness.values}")
    best_nn = create_nn(best_individual)
    # save nn to file
    torch.save(best_nn.state_dict(), 'best_nn.pth')
    plot_evolution(logbook, history_best_bankrolls)
