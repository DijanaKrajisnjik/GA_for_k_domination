import os
import random
import csv
from multiprocessing import Pool
from GA_for_k_domination_final import genetic_algorithm
from read_graph import read_graph

# FOLDER sa instancama
INSTANCE_FOLDER = "cities_small_instances"

# PARAMETRI ZA GRID SEARCH
population_sizes = [50, 100, 150, 200]
mutation_rate = 0.15
crossover_rate = 0.8
min_penalty = 0.01
population_reduction = 0.05
k_values = [1, 2, 4]

# Ostali parametri
time_limit = 1800  # 30 minuta
max_generations = 100
max_no_improvment = 4
max_penalty = 2

# CSV i log fajlovi
OUTPUT_CSV = "experiment_results_final.csv"
progress_log_file = "progress_log_test_parameters_final.txt"

# Učitaj instance
all_instances = [f for f in os.listdir(INSTANCE_FOLDER) if f.endswith(".txt")]
selected_instances = all_instances  # testiramo sve instance
print("Biće testirane instance:", selected_instances)

# CSV zaglavlje
header = [
    "no", "instance", "k", "population", "best_size",
    "alg_time", "initialization_time", "valid", "best_chromosome"
]

# Učitaj već završene eksperimente iz loga
finished_keys = set()
if os.path.exists(progress_log_file):
    with open(progress_log_file, "r") as f:
        for line in f:
            finished_keys.add(line.strip())

def prepare_task():
    tasks = []
    for i in range(5):  # ponovi 5 puta za statistiku
        for instance in selected_instances:
            for k in k_values:
                for pop_size in population_sizes:
                    key = f"{i},{instance},{k},{pop_size}"
                    if key in finished_keys:
                        print(f"Preskačem već završeno: {key}")
                        continue
                    tasks.append((i, instance, k, pop_size, key))
    return tasks

def run_experiments(task):
    i, instance, k, pop_size, key = task
    graph_path = os.path.join(INSTANCE_FOLDER, instance)
    g = read_graph(graph_path)

    ga = genetic_algorithm(
        instance_name=instance,
        k=k,
        graph=g,
        population_reduction=population_reduction,
        population_size=pop_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=4,
        elitism=True,
        time_limit=time_limit,
        max_no_improvment=max_no_improvment,
        rseed=random.randint(0, 99999)
    )

    initialization_time, alg_time, best_fitness, best_chromosome, valid = ga.run()

    result = [
        i, instance, k, pop_size,
        ga.best_fitness[1], round(alg_time, 2),
        round(initialization_time, 2),
        "yes" if valid else "no", best_chromosome
    ]
    return result, key 

if __name__ == "__main__":
    tasks = prepare_task()
    mode = "a" if os.path.exists(OUTPUT_CSV) else "w"

    with open(OUTPUT_CSV, mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if mode == "w":
            writer.writerow(header)

        with Pool(8) as pool:
            for idx, (result, key) in enumerate(pool.imap_unordered(run_experiments, tasks), start=1):
                writer.writerow(result)
                csvfile.flush()
                with open(progress_log_file, "a") as logf:
                    logf.write(f"{key}\n")
                print(f"[{idx}/{len(tasks)}] Završeno: {key}")

    print("Završeno testiranje svih instanci!")
    print(f"Rezultati su sačuvani u: {OUTPUT_CSV}")
    print(f"Napredak sačuvan u: {progress_log_file}")
