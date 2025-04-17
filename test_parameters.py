import os
import random
import csv

from GA_for_k_domination import genetic_algorithm
from read_graph import read_graph

# 📁 FOLDER sa instancama
INSTANCE_FOLDER = "cities_small_instances"

# 📋 PARAMETRI ZA GRID SEARCH
population_sizes = [100, 150, 200]
mutation_rates = [0.01, 0.05]
crossover_rates = [0.75, 0.85]
min_penalties = [0.005,0.01, 0.02]
penalty_reduction = [0.05, 0.1, 0.15]
k_values = [1, 2, 4]

# 🧪 Koliko instanci testiramo
percentage_instances = 0.3
time_limit = 600  # 10 minuta
max_generations = 100
max_no_improvment = 5
max_penalty = 1
# 📄 CSV izlaz
OUTPUT_CSV = "experiment_results.csv"

# 📂 Učitaj sve instance
all_instances = [f for f in os.listdir(INSTANCE_FOLDER) if f.endswith(".txt")]
random.seed(random.randint(0, 99999))
selected_instances = random.sample(all_instances, max(1, int(len(all_instances) * percentage_instances)))

print("Biće testirane instance:", selected_instances)

# 📝 CSV zaglavlje
header = [
    "instance", "k", "population", "mutation", "crossover", "min_penalty", 'penalty_reduction',
    "best_size", "alg_time", 'initialization_time', "valid", "best_chromosome"
]

progress_log_file = "progress_log_test_parametars.txt"
finished_keys = set()

if os.path.exists(progress_log_file):
    with open(progress_log_file, "r") as f:
        for line in f:
            finished_keys.add(line.strip())

mode = "a" if os.path.exists(OUTPUT_CSV) else "w"
# Ako je datoteka već postojala, dodaj zaglavlje
with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if mode == "w":
        # Ako je nova datoteka, dodaj zaglavlje
        writer.writerow(header)

    for instance in selected_instances:
        for k in k_values:
            for pop_size in population_sizes:
                for mut in mutation_rates:
                    for crossover in crossover_rates:
                        for min_pen in min_penalties:
                            for pen_red in penalty_reduction:
                                key = f"{instance},{k},{pop_size},{mut},{crossover},{min_pen},{pen_red}"
                                if key in finished_keys:
                                    print(f"Preskačem već završeno: {key}")
                                    continue
                                print(f"Pokreće se: {instance}, k={k}, pop={pop_size}, mut={mut}, crossover={crossover}, pen={min_pen}, pen_red={pen_red}")
                                graph_path = os.path.join(INSTANCE_FOLDER, instance)
                                g = read_graph(graph_path)

                                ga = genetic_algorithm(
                                    instance_name=instance,
                                    k=k,
                                    graph=g,
                                    max_penalty=max_penalty,
                                    min_penalty=min_pen,
                                    penalty_reduction=pen_red,
                                    population_size=pop_size,
                                    mutation_rate=mut,
                                    crossover_rate=crossover,
                                    tournament_size=5,
                                    elitism=True,
                                    time_limit=time_limit,
                                    generation_max=max_generations,
                                    max_no_improvment=max_no_improvment,
                                    rseed=random.randint(0, 99999)
                                )

                                initialization_time, alg_time, best_fitness, best_chromosome, valid = ga.run()

                                # Validacija

                                writer.writerow([
                                    instance, k, pop_size, mut, crossover, min_pen, pen_red,
                                    ga.best_fitness[1], round(alg_time, 2), round(initialization_time,2), "yes" if valid else "no", best_chromosome
                                ])
                                csvfile.flush()

                                with open(progress_log_file, "a") as logf:
                                    logf.write(f"{key}\n")
