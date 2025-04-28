from time import time
from math import log, sqrt
import random
from read_graph import read_graph
from networkx import DiGraph, Graph
from unit import fitness, fitness_rec_rem, fitness_rec_add, cache_rec_add, cache_rec_rem, is_acceptable_solution
class genetic_algorithm:
    def __init__(self, instance_name, k, graph: Graph, max_penalty, min_penalty, population_reduction, population_size, mutation_rate, crossover_rate, tournament_size, elitism, time_limit, generation_max, max_no_improvment, rseed, loading=False):
        self.instance_name = instance_name
        self.k = k
        self.graph = graph
        self.max_penalty = max_penalty
        self.min_penalty = min_penalty
        self.population_reduction = population_reduction
        self.penalty = min_penalty
        self.time_limit = time_limit
        self.generation_max = generation_max
        self.max_no_improvment = max_no_improvment
        self.nodes = list(self.graph.nodes) 
        self.chromosome_length = self.graph.number_of_nodes()
        self.rseed = rseed
        self.loading = loading
        random.seed(self.rseed)
        
        n = graph.number_of_nodes()
        self.generation_max = int(10 * log(n) + 0.3 * sqrt(n))

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
        #if self.loading:
            #self.load_population()
            #return

        base_percentage = self.graph.number_of_nodes()/self.graph.number_of_edges()
        #min_percentage = pow((self.k / sqrt(base_percentage)) * 0.01, 2)
        #max_percentage = min(1, base_percentage + 0.05) + (self.k / sqrt(base_percentage)) * 0.01
        #print("Base percentage ", base_percentage,"Min: ", min_percentage, "Max: ", max_percentage)
        
        # Testirano: Best fitness: (0, 83, 478, 10, 893) Time: 172.0357894897461 Generation: 13
        #Initialization time: 137.6408486366272 Algorithm time: 172.02130460739136 Best fitness: 83 Valid: True
        #max_percentage = max(k*0.35, self.k*base_percentage * 1.25)
        min_percentage = self.k*base_percentage / 5
        max_percentage = base_percentage * self.k * 1.25

        #Za testiranje: Best fitness: (0, 81, 459, 10, 895) Time: 174.45230174064636 Generation: 15
        #Initialization time: 173.80391430854797 Algorithm time: 174.43782377243042 Best fitness: 81 Valid: True

        #min_percentage = base_percentage / self.k
        #max_percentage = base_percentage * self.k

        # Za testiranje:Best fitness: (0, 81, 497, 11, 895) Time: 240.30986905097961 Generation: 21
        #Initialization time: 109.5612564086914 Algorithm time: 240.29135298728943 Best fitness: 81 Valid: True
        #min_percentage = 0.0125 * self.k
        #max_percentage = 0.065 * self.k

        # Za testiranje:
        #Best fitness: (0, 83, 480, 11, 893) Time: 234.86414074897766 Generation: 20
        #Initialization time: 165.0601305961609 Algorithm time: 234.84715056419373 Best fitness: 83 Valid: True
        #min_percentage = base_percentage / sqrt(self.k)
        #max_percentage = base_percentage * pow(self.k, 2)

        # Za testiranje: Best fitness: (0, 81, 445, 11, 895) Time: 243.8924651145935 Generation: 25
        #Initialization time: 120.5368230342865 Algorithm time: 243.87470841407776 Best fitness: 81 Valid: True

        #min_percentage = 0.2
        #max_percentage = 0.3
        
        # Testiran: Best fitness: (0, 79, 392, 11, 897) Time: 203.0644257068634 Generation: 21
        ##Initialization time: 105.87152314186096 Algorithm time: 203.05018830299377 Best fitness: 79 Valid: True
        #min_percentage = 0.02
        #max_percentage = 0.15
        
        #Formula: ---- Radilo najbolje ----
        #min_percentage = 0.015 * self.k
        #max_percentage = 0.065 * self.k

        #min_percentage = 0.03
        #max_percentage = 0.2

        #Testiran: Best fitness: (0, 81, 505, 10, 895) Time: 182.24197125434875 Generation: 16
        #Initialization time: 243.1526973247528 Algorithm time: 182.22357869148254 Best fitness: 81 Valid: True
        #min_percentage = 0
        #max_percentage = 0.05
        #base_percentage = self.graph.number_of_nodes() / self.graph.number_of_edges()
        #min_percentage = max(0.01, base_percentage * (self.k / 2))  # donja granica (1% minimum)
        #max_percentage = min(0.9, base_percentage * (self.k * 2))   # gornja granica (90% max)

        print("Base percentage ", base_percentage,"Min: ", min_percentage, "Max: ", max_percentage)
        for _ in range(self.population_size//2):
            #chromosome = [random.randint(0, 1) for j in range(self.chromosome_length)]
            percentage = random.uniform(min_percentage, max_percentage)
            #chromosome = self.generate_sparse_chromosome(self.chromosome_length, percentage)
            chromosome = self.generate_biased_chromosome(degree_bias=0.4, percentage=percentage)
        
        for _ in range(self.population_size-self.population_size//2):
            percentage = random.uniform(min_percentage, max_percentage)
            #chromosome = self.generate_sparse_chromosome(percentage)
            chromosome = self.generate_sparse_chromosome(percentage=percentage)

            self.population.append(self.local_search_best(chromosome))
        #self.save_population()    
        #self.population = [self.local_search_best(chromosome) for chromosome in self.population]
        #for i in range(self.population_size // 2):
            #self.population[i] = self.local_search_best(self.population[i])
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
        #number_of_accepted= 0
        for chromosome in self.population:
            fitness = self.fitness_function(chromosome)
            self.fitness.append(fitness)
            if self.first_fitness_better(fitness, self.best_fitness):
                self.best_fitness = fitness
                self.best_chromosome = chromosome
            #if is_acceptable_solution(self.graph, self.chromosone_to_set(chromosome), self.k):
                #number_of_accepted += 1
        return self.best_fitness, self.best_chromosome
    
    def fitness_function(self, chromosome):
        chromosome_tuple = tuple(chromosome)
        if chromosome_tuple in self.fitness_cache:
            return self.fitness_cache[chromosome_tuple]
        fitness_value = fitness(set(i for i, gene in enumerate(chromosome) if gene == 1), self.graph, self.k, {})
        self.fitness_cache[chromosome_tuple] = fitness_value
        return fitness_value
        #s = set([i for i in range(len(chromosome)) if chromosome[i] == 1])
        #return fitness(s, self.graph, self.k, {})
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
    #def mutation(self, chromosome, change_size=1):
        #mutation_points = random.sample(range(self.chromosome_length), change_size)
        #for point in mutation_points:
            #chromosome[point] = 1 - chromosome[point]
        #return chromosome
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
        #num_replace = int(0.1 * self.population_size)
        #for i in range(num_replace):
            #percentage = random.uniform(0.05, 0.25)
            #chromosome = self.generate_sparse_chromosome(self.chromosome_length, percentage)
            #self.population.append(self.local_search_best(chromosome))

            
            #new_chrom = [0] * self.chromosome_length
            #ones = random.sample(range(self.chromosome_length), random.randint(5, int(0.2 * self.chromosome_length)))
            #for i in ones:
                #new_chrom[i] = 1
            #self.population[-1 * (1 + i)] = self.local_search_best(new_chrom)

    def evolve(self):
        # Reduce the population size dynamically if conditions are met
        #reduction_fraction = 0.1
        if  self.population_size > 30:
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
        #average_fitness = sum(f[0] for f in self.fitness) / len(self.fitness)
        #print("Average fitness: ", average_fitness)

        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child1, child2 = self.uniform_crossover(parent1, parent2)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            # Evaluacija djece odmah nakon mutacije
            #fit1 = self.fitness_function(child1)
            #fit2 = self.fitness_function(child2)

            # Prag - LS se koristi samo za dovoljno dobra rješenja
            # (npr. broj konflikata manji od 10)
            #conflict_threshold = 10
            #Average fitness[0]
            
            #if fit1[0] < average_fitness*3:
                #child1 = self.local_search_best(child1)
            #new_population.append(child1)

            #if fit2[0] < average_fitness*3:
                #child2 = self.local_search_best(child2)
            #new_population.append(child2)

            new_population.append(self.local_search_best(child1))
            new_population.append(self.local_search_best(child2))
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
            #new_chrom = [0] * self.chromosome_length
            #ones_indices = random.sample(range(self.chromosome_length), min(target_size, self.chromosome_length))
            #for idx in ones_indices:
                #new_chrom[idx] = 1

            new_chrom = self.local_search_best(new_chrom)
            new_chromosomes.append(new_chrom)

        # Zamijeni najgore jedinke (ili dodaj ako želiš privremeno veću populaciju)
        #self.population[-num_to_inject:] = new_chromosomes
        self.population.extend(new_chromosomes)
        self.fitness.extend([self.fitness_function(c) for c in new_chromosomes])


    def dynamic_penalty(self, generation):
        print("Generation: ", generation, "Max gen: ", self.generation_max)
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
        print("Unique ratio:", round(unique_ratio, 2))

        # Ako ima previše duplikata – zamijeni ih
        if unique_ratio < 0.70:
            start_time = time()
            print(f"⚠️  Diverzitet nizak ({round(unique_ratio,2)}), zamjena {len(duplicates)} duplikata.")
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
            print(f"✅  Zamjena duplikata završena, trajalo {end_time:.2f} sekundi.")

        
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
            #if no_improvment > 0:
                #self.inject_diversity(percentage=no_improvment/10)
            oldBestFitness = self.best_fitness
            generation += 1
            self.evolve()
            #self.change_duplicates()

            # Testirano za leicester, bolje bez njega jer usporava
            #if generation % 5 == 0 or no_improvment > 0:
                #self.reinject_refresh(percentage= no_improvment/10 if no_improvment>0 else 0.1)
            #if no_improvment > 0:
                #self.reinject_refresh(percentage= no_improvment/10)
            self.penalty= self.dynamic_penalty(generation)
            print("Penalty: ", self.penalty)
            self.evaluate_population()
            if self.first_fitness_better(self.best_fitness, oldBestFitness):
                no_improvment = 0
                best_time = time() - start_time
                print("Best fitness :", self.best_fitness, ", fitness calculated: ", (1+self.best_fitness[0])*(1+self.best_fitness[1]*self.penalty), "Time:", best_time)
            else:
                no_improvment += 1
                print("No improvement, no_improvment:", no_improvment)
                #self.inject_diversity(percentage=no_improvment/10)

        alg_time = time() - start_time
        print("Algorithm finished, Time:", alg_time)
    
        print("Chromosome acceptable: ", is_acceptable_solution(self.graph, self.chromosone_to_set(self.best_chromosome), self.k))
        print("Best fitness:", self.best_fitness, "Time:", time() - start_time, "Generation:", generation)
        return initialization_time, alg_time, self.best_fitness[1], self.best_chromosome, is_acceptable_solution(self.graph, self.chromosone_to_set(self.best_chromosome), self.k)
        #return self.best_chromosome, self.best_fitness
    

if __name__ == '__main__':
    arguments={'instance_dir': "cities_small_instances",'instance':"newcastle.txt", 'k':2, 'time_limit':600, 'generation_max':100, 'max_no_improvment': 5,'rseed': random.randint(1,1000), 'population_size': 100, 'mutation_rate': 0.15, 'crossover_rate': 0.80, 'tournament_size': 4, 'elitism': True, 'max_penalty': 2, 'min_penalty': 0.01, 'penalty_reduction': 0.1}
    
    graph_open = arguments["instance_dir"] + '/' + arguments["instance"]
    print("Reading graph!")
    g = read_graph(graph_open)
    print("Graph loaded: ", graph_open)

    ga = genetic_algorithm(arguments['instance'], arguments['k'], g, arguments['max_penalty'], arguments['min_penalty'], arguments['penalty_reduction'], arguments['population_size'],  arguments['mutation_rate'], arguments['crossover_rate'], arguments['tournament_size'], arguments['elitism'], arguments['time_limit'], arguments['generation_max'], arguments['max_no_improvment'], arguments['rseed'], False)

    start_time = time()
    initialization_time, alg_time, best_fitness, best_chromosome, valid = ga.run()
    print("Initialization time:", initialization_time, "Algorithm time:", alg_time, "Best fitness:", best_fitness, "Valid:", valid)
    ## best fitness i result
## Tournament selection:
    ##two_position_crossover: 28, 77.75
    ##one_position_crossover: 26, 94.80
    ##uniform_crossover: 25, 96.37   

    ##uniform_crossover: 
        ##penality 0.01, mutation: 25, 96.37   ########### BEST ############# 
            ##penalty: 0.05: 25, 91.82
            ## veći penalty: ubrzava malo
        ##mutation whole: 26, 146.36
        
    ## Roullette selection, mutation, uniform crossover: 28, 260.35
    ## without elitism: 26, 85.91
    #### 90, 124.77, no_gen 22
    #### 47, 106.19 gen 13 ---- classic
#### Brington with LS reduction for children
    #### brington time 611-524=87, 45, k=2
    #### brington time 157, 44, k=2
    #### brington time 163, 44, k=2
#### Brington without LS reduction for children: 279, 42, k=2
    #### with fitness cache: 269, 42, k=2
    #### with reduction fraction: 76, 44, k=2; 134, 43, k=2

#### Oxford res: 48, time: 111-78 = 33s
#### with child LS reduction (min population 30): res 52, time 19s
#### with bigger min population (50): 50, 28s

#### 1. Best fitness: (0, 42, 303, 7, 934) Time: 242.9368588924408 Generation: 16
#####Initialization time: 97.45764994621277 Algorithm time: 242.9216022491455 Best fitness: 42 Valid: True
#### 2.Best fitness: (0, 42, 293, 7, 934) Time: 207.05100440979004 Generation: 17
##### Initialization time: 96.163827419281 Algorithm time: 207.03664565086365 Best fitness: 42 Valid: True
#### 3. Best fitness: (0, 42, 316, 6, 934) Time: 143.37209010124207 Generation: 19
#####Initialization time: 52.08418321609497 Algorithm time: 143.35619688034058 Best fitness: 42 Valid: True
#### 4. Best fitness: (0, 42, 316, 6, 934) Time: 143.49504041671753 Generation: 19
#####Initialization time: 47.984344482421875 Algorithm time: 143.48119974136353 Best fitness: 42 Valid: True

### coventry.txt,2,100,76,258.6,822.85,yes
####Best fitness: (0, 75, 296, 6, 1100) Time: 174.4880576133728 Generation: 24
#####Initialization time: 69.58992743492126 Algorithm time: 174.47254586219788 Best fitness: 75 Valid: True

### leicester.txt,2,100,79,490.44,1793.34,yes
####Best fitness: (0, 79, 599, 8, 1452) Time: 283.5680708885193 Generation: 17
#####Initialization time: 135.6845829486847 Algorithm time: 283.5420699119568 Best fitness: 79 Valid: True
####Best fitness: (0, 78, 571, 9, 1453) Time: 275.5397217273712 Generation: 20
#####Initialization time: 138.87649869918823 Algorithm time: 275.51872062683105 Best fitness: 78 Valid: True
#### Best fitness: (0, 77, 522, 7, 1454) Time: 370.42776370048523 Generation: 21
##### Initialization time: 0.028728485107421875 Algorithm time: 370.4028089046478 Best fitness: 77 Valid: True
#### Best fitness: (0, 77, 504, 7, 1454) Time: 256.5586450099945 Generation: 17
#####Initialization time: 0.023563861846923828 Algorithm time: 256.5335817337036 Best fitness: 77 Valid: True
#### Best fitness: (0, 77, 503, 8, 1454) Time: 302.5390508174896 Generation: 19
#####Initialization time: 0.021300315856933594 Algorithm time: 302.51588916778564 Best fitness: 77 Valid: True
#### Best fitness: (0, 78, 594, 8, 1453) Time: 485.5801510810852 Generation: 22
#####Initialization time: 0.02158522605895996 Algorithm time: 485.55440068244934 Best fitness: 78 Valid: True
### Best fitness: (0, 78, 443, 7, 1453) Time: 231.00704073905945 Generation: 17
#### Initialization time: 0.023653030395507812 Algorithm time: 230.98172211647034 Best fitness: 78 Valid: True
###Best fitness: (0, 76, 486, 9, 1455) Time: 269.9954090118408 Generation: 18
####Initialization time: 167.7390730381012 Algorithm time: 269.9696650505066 Best fitness: 76 Valid: True
### pop=150 Best fitness: (0, 75, 471, 7, 1456) Time: 292.03903102874756 Generation: 17
####Initialization time: 262.54358100891113 Algorithm time: 292.0097711086273 Best fitness: 75 Valid: True
