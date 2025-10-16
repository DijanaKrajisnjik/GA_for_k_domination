import argparse
import csv
import random
import os
from GA_for_k_domination_final import genetic_algorithm
from read_graph import read_graph

def main():
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm for k-domination problem.")
    parser.add_argument('-i', '--instance', required=True, help="Path to graph instance file")
    parser.add_argument('-pop_size', '--population_size', type=int, required=True, help="Population size")
    parser.add_argument('-k', type=int, required=True, help="Parameter k")
    parser.add_argument('-seed', type=int, required=True, help="Random seed")
    parser.add_argument('-o', '--output', required=True, help="Output CSV file path")
    args = parser.parse_args()

    random.seed(args.seed)

    # --- Fiksni parametri ---
    mutation_rate = 0.15
    crossover_rate = 0.8
    population_reduction = 0.05
    time_limit = 1800  # 30 minuta
    max_no_improvment = 4

    # --- Učitavanje grafa ---
    g = read_graph(args.instance)

    # --- Pokretanje GA ---
    ga = genetic_algorithm(
        instance_name=os.path.basename(args.instance),
        k=args.k,
        graph=g,
        population_reduction=population_reduction,
        population_size=args.population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=4,
        elitism=True,
        time_limit=time_limit,
        max_no_improvment=max_no_improvment,
        rseed=args.seed
    )

    initialization_time, alg_time, best_fitness, best_chromosome, valid = ga.run()

    # --- Upis rezultata u CSV ---
    header = ["instance", "k", "population", "best_size", "alg_time", "initialization_time", "valid", "best_chromosome"]

    file_exists = os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            os.path.basename(args.instance),
            args.k,
            args.population_size,
            ga.best_fitness[1],
            round(alg_time, 2),
            round(initialization_time, 2),
            "yes" if valid else "no",
            best_chromosome
        ])

    print(f"Završeno: {args.instance} (k={args.k}, pop={args.population_size}, seed={args.seed})")
    print(f"Rezultati su upisani u {args.output}")

if __name__ == "__main__":
    main()
