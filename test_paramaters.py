import os
import random
import csv
import time
from importlib import import_module

from GA_for_k_domination import genetic_algorithm
from read_graph import read_graph
from unit import is_acceptable_solution

# 📁 FOLDER sa instancama
INSTANCE_FOLDER = "cities_small_instances"

# 📋 PARAMETRI ZA GRID SEARCH
population_sizes = [50, 100, 150]
mutation_rates = [0.01, 0.05]
crossover_rates = [0.75,0.85, 0.9]
penalties = [0.005,0.01, 0.02]
k_values = [1, 2, 4]

# 🧪 Koliko instanci testiramo
percentage_instances = 0.2
time_limit = 300  # 5 minuta
max_generations = 100
max_no_improvment = 5
# 📄 CSV izlaz
OUTPUT_CSV = "experiment_results.csv"

# 📂 Učitaj sve instance
all_instances = [f for f in os.listdir(INSTANCE_FOLDER) if f.endswith(".txt")]
random.seed(42)
selected_instances = random.sample(all_instances, max(1, int(len(all_instances) * percentage_instances)))

print("Biće testirane instance:", selected_instances)

# 📝 CSV zaglavlje
header = [
    "instance", "k", "population", "mutation", "crossover", "penalty",
    "best_size", "time", "valid", "fit_tuple"
]

with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    for instance in selected_instances:
        for k in k_values:
            for pop_size in population_sizes:
                for mut in mutation_rates:
                    for crossover in crossover_rates:
                        for pen in penalties:
                            print(f"Pokreće se: {instance}, k={k}, pop={pop_size}, mut={mut}, crossover={crossover}, pen={pen}")
                            graph_path = os.path.join(INSTANCE_FOLDER, instance)
                            g = read_graph(graph_path)

                            ga = genetic_algorithm(
                                instance_name=instance,
                                k=k,
                                graph=g,
                                penalty=pen,
                                population_size=pop_size,
                                chromosome_length=g.number_of_nodes(),
                                mutation_rate=mut,
                                crossover_rate=crossover,
                                tournament_size=5,
                                elitism=True,
                                time_limit=time_limit,
                                generation_max=max_generations,
                                max_no_improvment=max_no_improvment,
                                rseed=random.randint(0, 99999)
                            )

                            start = time.time()
                            best_chromosome, best_fitness = ga.run()
                            duration = time.time() - start

                            # Validacija
                            valid = is_acceptable_solution(g, ga.chromosone_to_set(best_chromosome), k)

                            writer.writerow([
                                instance, k, pop_size, mut, crossover, pen,
                                ga.best_fitness[1], round(duration, 2), valid, best_fitness
                            ])
                            csvfile.flush()
