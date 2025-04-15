import os
import random
import csv
import time

from GA_for_k_domination import genetic_algorithm
from read_graph import read_graph
from unit import is_acceptable_solution

INSTANCE_FOLDER = "cities_small_instances"
NUM_RUNS = 3
TIME_LIMIT = 300

# Fiksirane vrijednosti
BASE_PARAMS = {
    "k": 2,
    "population_size": 100,
    "mutation_rate": 0.01,
    "penalty": 0.01,
    "crossover_rate": 0.85,
    "tournament_size": 5
}

# Parametri koje ćemo testirati
TEST_PARAM_VALUES = {
    "population_size": [50, 100, 150],
    "mutation_rate": [0.005, 0.01, 0.05],
    "penalty": [0.005, 0.01, 0.02],
    "k": [1, 2, 4]
}

OUTPUT_CSV = "sensitivity_results.csv"

# Učitaj sve instance
all_instances = [f for f in os.listdir(INSTANCE_FOLDER) if f.endswith(".txt")]
random.seed(42)
selected_instances = random.sample(all_instances, 3)


header = ["parametar", "vrijednost", "prosječan_fit", "prosječan_vrijeme", "validnost%"]

with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    for param_name, values in TEST_PARAM_VALUES.items():
        for val in values:
            fit_list = []
            time_list = []
            valid_count = 0

            print(f"Test parametra: {param_name} = {val}")
            for instance in selected_instances:
                print(f"Testing instance: {instance}")
                for run in range(NUM_RUNS):
                    print(f"Run {run + 1}/{NUM_RUNS}")
                    graph_path = os.path.join(INSTANCE_FOLDER, instance)
                    g = read_graph(graph_path)
                    params = BASE_PARAMS.copy()
                    params[param_name] = val

                    ga = genetic_algorithm(
                        instance_name=instance,
                        k=params["k"],
                        graph=g,
                        penalty=params["penalty"],
                        population_size=params["population_size"],
                        chromosome_length=g.number_of_nodes(),
                        mutation_rate=params["mutation_rate"],
                        crossover_rate=0.85,
                        tournament_size=5,
                        elitism=True,
                        time_limit=TIME_LIMIT,
                        generation_max=100,
                        max_no_improvment=10,
                        rseed=random.randint(0, 99999),
                    )

                    start = time.time()
                    best_chromosome, best_fitness = ga.run()
                    duration = time.time() - start

                    is_valid =is_acceptable_solution(g, ga.chromosone_to_set(best_chromosome), params["k"])

                    fit_list.append(sum(best_fitness))  # total fitness kao broj konflikata + veličina
                    time_list.append(duration)
                    if is_valid:
                        valid_count += 1
            print("Završeno testiranje instance: {instance}")
            total_runs = NUM_RUNS * len(selected_instances)
            avg_fit = round(sum(fit_list) / total_runs, 4)
            avg_time = round(sum(time_list) / total_runs, 2)
            valid_percent = round((valid_count / total_runs) * 100, 1)

            writer.writerow([param_name, val, avg_fit, avg_time, valid_percent])
            csvfile.flush()
