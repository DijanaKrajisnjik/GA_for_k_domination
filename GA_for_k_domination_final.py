from time import time
from math import log, sqrt
import random
import math
from read_graph import read_graph
from networkx import DiGraph, Graph
from unit import fitness, fitness_rec_rem, fitness_rec_add, cache_rec_add, cache_rec_rem, is_acceptable_solution
class genetic_algorithm:
    def __init__(self, instance_name, k, graph: Graph, population_reduction, population_size, mutation_rate, crossover_rate, tournament_size, elitism, time_limit, max_no_improvment, rseed, loading=False):
        self.instance_name = instance_name
        self.k = k
        self.graph = graph
        n = graph.number_of_nodes()
        self.min_penalty = max(0.005, 0.05 / math.log(n + 5))
        self.max_penalty = min(2.0, 0.2 * math.log10(n + 10))

        self.population_reduction = population_reduction
        
        self.generation_max = int(10 * log(n) + 0.3 * sqrt(n))
        self.penalty = self.dynamic_penalty_v2(0)
        self.time_limit = time_limit
        self.max_no_improvment = max_no_improvment
        self.nodes = list(self.graph.nodes) 
        self.chromosome_length = n
        self.rseed = rseed
        self.loading = loading
        random.seed(self.rseed)
        
        

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.population = []
        self.fitness = []
        self.best_chromosome = []
        self.best_fitness = 0

        self.fitness_cache = {}
    
        self.neighbors = {}
        self.neighb_matrix = [[] for _ in range(len(self.graph.nodes))]
        for v in self.graph.nodes:
            self.neighbors[v] = set(self.graph.neighbors(v))
            self.neighb_matrix[v] = [False]*len(self.graph.nodes)
            for u in self.graph.neighbors(v):
                self.neighb_matrix[v][u] = True

    
    def initialize_population(self):
        if self.loading:
            self.load_population()
            return
        self.auto_tune_penalty_params()

        #print("Density: ", self.graph.number_of_edges()/self.graph.number_of_nodes())
        base_percentage = self.graph.number_of_nodes()/self.graph.number_of_edges()
        max_percentage = max(self.k*0.05, self.k*base_percentage * 1.25)
        min_percentage = self.k*base_percentage / 5

        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        density = num_edges / num_nodes  # prosječan stepen / 2

        # Ako je graf gust → idi prema manjem biasu (više raznolikosti)
        # Ako je graf rjeđi → idi prema većem biasu (više fokusiranosti)
        bias_strength = max(0.5, min(5.0, 2.5 * (1.0 / (density / 10 + 0.1))))

        max_percentage = max(self.k*0.05, self.k*base_percentage * 1.25)
        min_percentage = self.k*base_percentage / 5

        percentage = random.uniform(min_percentage, max_percentage)

        #percentage = max(0.002, min(0.1, 60 / num_nodes))
        
        #print("Base percentage ", base_percentage,"Min: ", min_percentage, "Max: ", max_percentage)
        #num_of_repaired = 0
        for _ in range(self.population_size):
            chromosome = self.generate_sparse_chromosome(percentage=percentage)
            #chromosome = self.generate_weighted_biased_chromosome(percentage=percentage, bias_strength=bias_strength)
            #print("Generated chromosome with num ones:", sum(chromosome))
            #print("Positions of ones:", [i for i in range(len(chromosome)) if chromosome[i]==1])
            self.population.append(self.local_search_best(chromosome))
            '''
            if is_acceptable_solution( self.graph, self.chromosone_to_set(chromosome), self.k) and random.random()<0.5:
                chromosome = self.population.append(chromosome)
            else:
                self.population.append(self.local_search_best(chromosome))
                num_of_repaired += 1
            '''
            number_of_ones = sum(self.population[-1])
            percentage=number_of_ones / self.chromosome_length
            #print("After local search num ones:", number_of_ones, "Percentage:", round(percentage,4))
            #print("Positions of ones:", [i for i in range(len(self.population[-1])) if self.population[-1][i]==1])
        #print("Populacija inicijalizovana. Broj hromozoma:", len(self.population))
        #print("Number of repaired chromosomes:", num_of_repaired)

        #self.save_population()    
       
    def auto_tune_penalty_params(self):
        n = self.graph.number_of_nodes()
        m = self.graph.number_of_edges()
        density = m / (n * (n - 1) / 2)
        
        self.min_penalty = 0.0005 + (1 - density) * 0.002
        self.max_penalty = min(0.01, 0.001 * (self.k + 1) * (1000 / n))
        self.generation_max = int(200 + 0.02 * n)

    def dynamic_penalty_v2(self, generation):
        ratio = min(1, max(0, (generation - 2) / self.generation_max))
        return (1 - ratio) * self.min_penalty + ratio * self.max_penalty
    
    ##############################

    def generate_biased_chromosome_v2(self, top_percent=0.65, degree_bias=0.3, percentage=0.1):
        chromosome = [0] * self.chromosome_length
        num_ones = int(self.chromosome_length * percentage)

        # 1. Odredi top čvorove po stepenu (širi skup npr. top 60%)
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(
            [(self.nodes.index(v), deg) for v, deg in degrees.items() if v in self.nodes],
            key=lambda x: x[1],
            reverse=True
        )
        top_k = int(self.chromosome_length * top_percent)
        top_indices = [i for i, _ in sorted_nodes[:top_k]]

        # 2. Nasumično uzmi dio iz top skupa (biased)
        num_biased = int(num_ones * degree_bias)
        biased_indices = random.sample(top_indices, min(num_biased, len(top_indices)))

        # 3. Ostatak nasumično iz svih koji nisu u biased
        remaining_needed = num_ones - len(biased_indices)
        available_indices = list(set(range(self.chromosome_length)) - set(biased_indices))
        random.shuffle(available_indices)
        additional_indices = available_indices[:remaining_needed]

        # 4. Postavi gene
        for idx in biased_indices + additional_indices:
            chromosome[idx] = 1

        return chromosome

    def generate_weighted_biased_chromosome(self, percentage=0.1, bias_strength=3.0):
        """
        - percentage: ukupan procenat gena sa vrijednošću 1.
        - bias_strength: veće vrijednosti favorizuju visoko stepenovane čvorove (npr. 2.0 do 5.0).
        """
        chromosome = [0] * self.chromosome_length
        num_ones = int(self.chromosome_length * percentage)

        # Izračunaj stepen svakog čvora
        degrees = dict(self.graph.degree())
        node_indices = [self.nodes.index(v) for v in self.nodes if v in degrees]

        # Uskladi redoslijed stepene sa indeksima
        weights = [degrees[self.nodes[i]] ** bias_strength for i in node_indices]

        # Normalizacija (opcionalno, ali može pomoći)
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Ponderisani izbor indeksa
        chosen_indices = random.choices(node_indices, weights=weights, k=num_ones)

        for idx in chosen_indices:
            chromosome[idx] = 1

        return chromosome

    def generate_biased_chromosome(self, degree_bias=0.3, percentage = 0.1):
        chromosome = [0] * self.chromosome_length
        num_ones = int(self.chromosome_length * percentage)
        num_biased = int(num_ones * degree_bias)

        # 1. Top čvorovi po stepenu
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted([(self.nodes.index(v), deg) for v, deg in degrees.items() if v in self.nodes], key=lambda x: x[1], reverse=True)
        biased_indices = [i for i, _ in sorted_nodes[:num_biased]]

        # 2. Ostale pozicije nasumično (bez dupliranja)
        remaining_indices = list(set(range(self.chromosome_length)) - set(biased_indices))
        random.shuffle(remaining_indices)
        additional_indices = remaining_indices[:num_ones - len(biased_indices)]

        # 3. Postavi jedinice
        for i in biased_indices + additional_indices:
            chromosome[i] = 1

        return chromosome

    def generate_sparse_chromosome(self, percentage=0.1):
        # Generate a sparse chromosome with a certain percentage of ones
        length = self.chromosome_length
        chromosome = [0] * length
        num_ones = int(length * percentage)
        indices = random.sample(range(length), num_ones)
        for idx in indices:
            chromosome[idx] = 1
        return chromosome

    def save_population(self):
        file_init_name="initial_population" + self.instance_name + ".txt"
        with open(file_init_name, 'w') as f:
            for chromosome in self.population:
                f.write(' '.join(map(str, chromosome)) + '\n')
    
    def load_population(self):
        file_init_name="initial_population" + self.instance_name + ".txt"

        with open(file_init_name, 'r') as f:
            self.population = [list(map(int, line.strip().split())) for line in f.readlines()]
        self.chromosome_length = len(self.population[0])
        self.population_size = len(self.population)

    def evaluate_population(self):
        
        self.fitness = []
        self.best_fitness = self.fitness_function(self.population[0])
        for chromosome in self.population:
            fitness = self.fitness_function(chromosome)
            self.fitness.append(fitness)
            if self.first_fitness_better(fitness, self.best_fitness):
                self.best_fitness = fitness
                self.best_chromosome = chromosome
        return self.best_fitness, self.best_chromosome
    
    def fitness_function(self, chromosome):
        chromosome_tuple = tuple(chromosome)
        if chromosome_tuple in self.fitness_cache:
            return self.fitness_cache[chromosome_tuple]
        fitness_value = fitness(set(i for i, gene in enumerate(chromosome) if gene == 1), self.graph, self.k, {})
        self.fitness_cache[chromosome_tuple] = fitness_value
        return fitness_value
    
    def tournament_selection(self):
        tournament = []
        for i in range(self.tournament_size):
            tournament.append(random.randint(0, self.population_size - 1))
        best_chromosome = tournament[0]
        for i in tournament:
            if self.first_fitness_better(self.fitness[i], self.fitness[best_chromosome]):
                best_chromosome = i
        return self.population[best_chromosome]
    
    def roullette_selection(self):
        total_fitness = sum((1 + f[0]) * (1 + f[1] * self.penalty) for f in self.fitness)
        selection_probs = [(1 + f[0]) * (1 + f[1] * self.penalty) / total_fitness for f in self.fitness]
        cumulative_probs = [sum(selection_probs[:i + 1]) for i in range(len(selection_probs))]
        random_value = random.random()
        for i, cumulative_prob in enumerate(cumulative_probs):
            if random_value <= cumulative_prob:
                return self.population[i]
        return self.population[-1]  # Fallback in case of rounding errors
    
    def elitism_selection(self):
        if not self.elitism:
            return []
        # Select the best chromosomes based on fitness
        sorted_population = sorted(zip(self.population, self.fitness), key=lambda x: x[1], reverse=True)
        # Select the top 1% of chromosomes
        elite_chromosomes = [chromosome for chromosome, _ in sorted_population[:self.population_size // 100]]
        return elite_chromosomes
    def elitism_selection_with_diversity(self):
        if not self.elitism:
            return []

        # Sortiraj populaciju po fitnessu (bolji je manji)
        sorted_population = sorted(zip(self.population, self.fitness), key=lambda x: (1 + x[1][0]) * (1 + x[1][1] * self.penalty))

        elite_chromosomes = [sorted_population[0][0]]
        elite_sets = [set(i for i, bit in enumerate(sorted_population[0][0]) if bit == 1)]

        for chromosome, _ in sorted_population[1:]:
            s = set(i for i, bit in enumerate(chromosome) if bit == 1)
            # Računaj Jaccard razliku (1 - sličnost)
            jaccard_dist = 1 - len(elite_sets[0] & s) / max(1, len(elite_sets[0] | s))

            if jaccard_dist > 0.5:
                elite_chromosomes.append(chromosome)
                elite_sets.append(s)

            if len(elite_chromosomes) >= max(2, self.population_size // 50):
                break

        return elite_chromosomes
    
    def one_position_crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(0, self.chromosome_length - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        return parent1, parent2

    def two_position_crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point1 = random.randint(0, self.chromosome_length - 1)
            point2 = random.randint(point1, self.chromosome_length - 1)
            child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
            return child1, child2
        return parent1, parent2
    def uniform_crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            child1 = []
            child2 = []
            for i in range(self.chromosome_length):
                if random.random() < 0.5:
                    child1.append(parent1[i])
                    child2.append(parent2[i])
                else:
                    child1.append(parent2[i])
                    child2.append(parent1[i])
            return child1, child2
        return parent1, parent2
        
    def mutation_whole(self, chromosome):
        # Perform mutation on the entire chromosome
        # with a certain probability
        # This is a simple mutation where each gene has a chance to be flipped
        mutated_chromosome = []
        for gene in chromosome:
            if random.random() < self.mutation_rate:
                mutated_chromosome.append(1 - gene)
            else:
                mutated_chromosome.append(gene)
        return mutated_chromosome

    def mutation(self, chromosome, change_size=1):
        # Perform mutation on a single gene with a certain probability
        # This is a simple mutation where each gene has a chance to be flipped
        # with a certain probability
        chromosome = chromosome.copy()
        if random.random() < self.mutation_rate:
            for _ in range(change_size):
                mutation_point = random.randint(0, self.chromosome_length - 1)
                chromosome[mutation_point] = 1 - chromosome[mutation_point]
        return chromosome
    def reinject_refresh(self, percentage=0.1):
        sorted_pop_fit = sorted(zip(self.population, self.fitness), key=lambda x: (1 + x[1][0]) * (1 + x[1][1] * self.penalty), reverse=True)
        #num_replace = random.randint(int(percentage * self.population_size), int(0.1 * self.population_size))
        num_replace = int(percentage * self.population_size)
        for i in range(num_replace):
            new_chrom = self.generate_biased_chromosome(percent_ones=percentage)  # ili biased
            #improved_chrom = self.local_search_best(new_chrom)
            worst_index = self.population.index(sorted_pop_fit[i][0])
            #self.population[worst_index] = improved_chrom
            self.population[worst_index] = new_chrom
        # Reinject a portion of the population with new random chromosomes

    def evolve(self):
        # Reduce the population size dynamically if conditions are met
        #reduction_fraction = 0.1
        #print("Populacija ima hromozoma:", len(self.population))

        if  self.population_size > 50:
            # Sort population by fitness to keep the best individuals
            sorted_population = sorted(zip(self.population, self.fitness), key=lambda x: x[1])
            new_size = int(len(self.population) * (1 - self.population_reduction))
            self.population, self.fitness = zip(*sorted_population[:new_size])
            self.population = list(self.population)
            self.fitness = list(self.fitness)
            self.population_size = len(self.population)

        new_population = []
        if self.elitism:
            elite_chromosomes = self.elitism_selection_with_diversity()
            new_population.extend(elite_chromosomes)
        #num_repair = 0
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child1, child2 = self.uniform_crossover(parent1, parent2)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            '''
            if is_acceptable_solution( self.graph, self.chromosone_to_set(child1), self.k) and random.random()<0.5:
                new_population.append(child1)
            else:
                new_population.append(self.local_search_best(child1))
                num_repair += 1
            if is_acceptable_solution( self.graph, self.chromosone_to_set(child2), self.k) and random.random()<0.5:
                new_population.append(child2)
            else:
                new_population.append(self.local_search_best(child2))
                num_repair += 1
            '''
            new_population.append(self.local_search_best(child1))
            new_population.append(self.local_search_best(child2))
        #print("Number of repaired in this generation :", num_repair)
        self.population = new_population    

    
    def first_fitness_better(self, fit1, fit2):
        fit1Tot = (1+fit1[0])*(1+fit1[1]*self.penalty)
        fit2Tot = (1+fit2[0])*(1+fit2[1]*self.penalty)
        return fit1Tot<fit2Tot
    
    def fitness_equal(self, fit1, fit2):
        return not self.first_fitness_better(fit1, fit2) and not self.first_fitness_better(fit2, fit1)
    
    def chromosone_to_set(self, c):    
        s= set()
        for i in range(len(c)):
            if c[i] == 1:
                s.add(i)
        return s
    
    def local_search_best(self, c):
        s= set()
        for i in range(len(c)):
            if c[i] == 1:
                s.add(i)
        improved = True
        cache = {}
        curr_fit = fitness(s, self.graph, self.k, cache)
        # adding nodes to achieve feasibility
        while improved:
            improved = False
            best_fit = curr_fit
            best_v = None

            for v in self.nodes:
                if v not in s:
                    new_fit = fitness_rec_add(s, v, curr_fit, self.graph, self.neighbors, self.neighb_matrix, self.k, cache)
                    if self.first_fitness_better(new_fit, best_fit):
                        best_fit = new_fit
                        best_v = v
                        improved = True
            
            if improved:
                cache_rec_add(s, best_v, curr_fit, self.graph, self.neighbors, self.neighb_matrix, self.k, cache)
                s.add(best_v)
                curr_fit = best_fit

        # now simple removal
        improved = True
        while improved:
            improved = False
            best_fit = curr_fit
            best_v = None

            for v in self.nodes:
                if v in s:
                    new_fit = fitness_rec_rem(s, v, curr_fit, self.graph, self.neighbors, self.neighb_matrix, self.k, cache)
                    if self.first_fitness_better(new_fit, best_fit):
                        best_fit = new_fit
                        best_v = v
                        improved = True
            
            if improved:
                cache_rec_rem(s, best_v, curr_fit, self.graph, self.neighbors, self.neighb_matrix, self.k, cache)
                s.remove(best_v)
                curr_fit = best_fit
        result = [0] * self.chromosome_length
        for i in s:
            result[i] = 1

        return result
    
    def inject_diversity(self, percentage=0.1, variance=0.15):
        num_to_inject = int(self.population_size * percentage)
        new_chromosomes = []

        best_size = self.best_fitness[1]  # trenutna najbolja kardinalnost

        for _ in range(num_to_inject):
            # +- varijacija oko najboljeg broja čvorova
            variation = int(best_size * random.uniform(-variance, variance))
            target_size = max(1, best_size + variation)

            new_chrom = self.generate_biased_chromosome(degree_bias=0.4, percentage=target_size / self.chromosome_length)

            new_chrom = self.local_search_best(new_chrom)
            new_chromosomes.append(new_chrom)

        # Zamijeni najgore jedinke (ili dodaj ako želiš privremeno veću populaciju)
        self.population.extend(new_chromosomes)
        self.fitness.extend([self.fitness_function(c) for c in new_chromosomes])


    def dynamic_penalty(self, generation):
        #print("Generation: ", generation, "Max gen: ", self.generation_max)
        ratio = (generation-2) / self.generation_max
        return (1 - ratio) * self.min_penalty + ratio * self.max_penalty
    
    def change_duplicates(self):    
        seen = set()
        new_population = []
        duplicates = []

        for chrom in self.population:
            t = tuple(chrom)
            if t not in seen:
                seen.add(t)
                new_population.append(chrom)
            else:
                duplicates.append(chrom)

        unique_ratio = len(new_population) / len(self.population)
        #print("Unique ratio:", round(unique_ratio, 2))

        # Ako ima previše duplikata – zamijeni ih
        if unique_ratio < 0.60:
            start_time = time()
            #print(f"⚠️  Diverzitet nizak ({round(unique_ratio,2)}), zamjena {len(duplicates)} duplikata.")
            for _ in range(len(duplicates)):
                # Brza random inicijalizacija
                percentage = random.uniform(0.05, 0.25)
                new_chrom = self.generate_sparse_chromosome(percentage=percentage)
                # Ako hoćeš da koristiš pametnu verziju:generate_sparse_chromosome
                # new_chrom = self.generate_biased_chromosome()

                # Ako želiš i LS, koristi ovo:
                # new_chrom = self.local_search_best(new_chrom)

                new_population.append(self.local_search_best(new_chrom))

            self.population = new_population[:self.population_size]  # da se ne prekorači veličina
            end_time = time() - start_time
            #print(f"✅  Zamjena duplikata završena, trajalo {end_time:.2f} sekundi.")

        
    def run(self):
        start_time = time()
        best_time = 0
        generation = 1
        no_improvment = 0
        self.initialize_population()
        initialization_time = time() - start_time
        print("Initial population created, Time:", initialization_time)
        start_time = time()
        self.evaluate_population()
        print("Initial population evaluated, Time:", time() - start_time)
        while time() - start_time < self.time_limit and generation < self.generation_max and no_improvment < self.max_no_improvment:
            print("Current generation:", generation, "Time:", time() - start_time)
            oldBestFitness = self.best_fitness
            generation += 1
            self.evolve()
            self.penalty= self.dynamic_penalty_v2(generation)
            #print("Penalty: ", self.penalty)
            self.evaluate_population()
            if self.first_fitness_better(self.best_fitness, oldBestFitness):
                no_improvment = 0
                best_time = time() - start_time
                print("Best fitness :", self.best_fitness, ", fitness calculated: ", (1+self.best_fitness[0])*(1+self.best_fitness[1]*self.penalty), "Time:", best_time)
            else:
                no_improvment += 1
                print("No improvement, no_improvment:", no_improvment)

        alg_time = time() - start_time
        print("Algorithm finished, Time:", alg_time)
    
        print("Chromosome acceptable: ", is_acceptable_solution(self.graph, self.chromosone_to_set(self.best_chromosome), self.k))
        print("Best fitness:", self.best_fitness, "Time:", time() - start_time, "Generation:", generation)
        return initialization_time, alg_time, self.best_fitness[1], self.best_chromosome, is_acceptable_solution(self.graph, self.chromosone_to_set(self.best_chromosome), self.k)
    

if __name__ == '__main__':
    arguments={'instance_dir': "cities_small_instances",'instance':"liverpool.txt", 'k': 2, 'time_limit':1800, 'generation_max':100, 'max_no_improvment': 4,'rseed': random.randint(1,1000), 'population_size': 100, 'mutation_rate': 0.15, 'crossover_rate': 0.8, 'tournament_size': 4, 'elitism': True, 'max_penalty': 2, 'min_penalty': 0.01, 'population_reduction': 0.05}
    
    graph_open = arguments["instance_dir"] + '/' + arguments["instance"]
    #print("Reading graph!")
    g = read_graph(graph_open)
    #print("Graph loaded: ", graph_open)

    ga = genetic_algorithm(arguments['instance'], arguments['k'], g, arguments['population_reduction'], arguments['population_size'],  arguments['mutation_rate'], arguments['crossover_rate'], arguments['tournament_size'], arguments['elitism'], arguments['time_limit'], arguments['max_no_improvment'], arguments['rseed'], False)

    start_time = time()
    initialization_time, alg_time, best_fitness, best_chromosome, valid = ga.run()
    #print("Initialization time:", initialization_time, "Algorithm time:", alg_time, "Best fitness:", best_fitness, "Valid:", valid)
   