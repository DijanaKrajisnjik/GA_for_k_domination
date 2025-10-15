import os
import random
import csv
from multiprocessing import Pool
from GA_for_k_domination_final import genetic_algorithm
from read_graph import read_graph

# 📁 FOLDER sa instancama
INSTANCE_FOLDER = "cities_small_instances"

# 📋 PARAMETRI ZA GRID SEARCH
population_sizes = [100, 150, 200]
#mutation_rates = [0.01, 0.05]
#crossover_rates = [0.75, 0.85]
#min_penalties = [0.005,0.01, 0.02]
mutation_rate = 0.15
crossover_rate = 0.8
min_penalty = 0.01

#population_reduction = [0.05, 0.1, 0.15]
population_reduction = 0.05
k_values = [1, 2, 4]

# 🧪 Koliko instanci testiramo
percentage_instances = 0.3
time_limit = 1800  # 30 minuta
max_generations = 100
max_no_improvment = 4
max_penalty = 2
# 📄 CSV izlaz
OUTPUT_CSV = "experiment_results_final.csv"

# 📂 Učitaj sve instance
all_instances = [f for f in os.listdir(INSTANCE_FOLDER) if f.endswith(".txt")]
random.seed(random.randint(0, 99999))
#selected_instances = random.sample(all_instances, max(1, int(len(all_instances) * percentage_instances)))
#selected_instances = ['glasgow.txt', 'exeter.txt', 'nottingham.txt', 'sunderland.txt']
selected_instances = all_instances
print("Biće testirane instance:", selected_instances)

# 📝 CSV zaglavlje
header = [
    "no","instance", "k", "population", "best_size", "alg_time", 'initialization_time', "valid", "best_chromosome"
]

progress_log_file = "progress_log_test_parameters_final.txt"
finished_keys = set()

if os.path.exists(progress_log_file):
    with open(progress_log_file, "r") as f:
        for line in f:
            finished_keys.add(line.strip())

def prepare_task():
    tasks = []
    for i in range(5):  # Ponovi cijeli set 5 puta za statistiku
        for instance in selected_instances:
            for k in k_values:
                for pop_size in population_sizes:
                    key = f"{i},{instance},{k},{pop_size}"
                    if key in finished_keys:
                        print(f"Preskačem već završeno: {key}")
                        continue
                    tasks.append((i, instance, k, pop_size, key))
    return tasks

mode = "a" if os.path.exists(OUTPUT_CSV) else "w"

def run_experiments(task):
    print(f"Započinjem: {task}")  
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
    print(f"Završeno:{i} {instance}, k={k}, pop={pop_size}")
    print(f"Najbolji fitness: {best_fitness}, vrijeme: {alg_time}, generacija: {ga.generation_max}")
    result = [
        i, instance, k, pop_size, 
        ga.best_fitness[1], round(alg_time, 2), round(initialization_time, 2), "yes" if valid else "no", best_chromosome
    ]
    # Upisivanje u CSV iz pojedinačnog procesa
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(result)
        f.flush()

    # 📒 Upis u log
    with open(progress_log_file, "a") as logf:
        logf.write(f"{key}\n")
        logf.flush()

    print(f"Završeno:{i} {instance}, k={k}, pop={pop_size}")
    return result 
'''
if __name__ == "__main__":
    tasks = prepare_task()
    mode = "a" if os.path.exists(OUTPUT_CSV) else "w"
    with open(OUTPUT_CSV, mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if mode == "w":
            writer.writerow(header)
        with Pool(8) as pool:
            results = pool.map(run_experiments, tasks)
            for result, key in results:
                writer.writerow(result)
                csvfile.flush()
                with open(progress_log_file, "a") as logf:
                    logf.write(f"{key}\n")
                print(f"Urađeno do sad: {len(finished_keys)}/{len(tasks)} %.")
    print("Završeno!")
    print(f"Rezultati su sačuvani u {OUTPUT_CSV}")
    print(f"Napredak je sačuvan u {progress_log_file}")
'''
if __name__ == "__main__":
    tasks = prepare_task()
    mode = "a" if os.path.exists(OUTPUT_CSV) else "w"
    with open(OUTPUT_CSV, mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if mode == "w":
            writer.writerow(header)
    with Pool(8) as pool:
        results = pool.map(run_experiments, tasks)
        for result in results:
            print(result)  # Ispis rezultata iz svakog zadatka
            print(f"Urađeno do sad: {len(finished_keys)}/{len(tasks)} %.")
            #pass  # već je zapisano u run_experiments
    print(f"Urađeno svih {len(tasks)} zadataka.")
    print("Završeno!")
    print(f"Rezultati su sačuvani u {OUTPUT_CSV}")
    print(f"Napredak je sačuvan u {progress_log_file}")
