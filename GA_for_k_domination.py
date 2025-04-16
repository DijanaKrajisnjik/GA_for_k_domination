from time import time
from math import log, sqrt
import random
from read_graph import read_graph
from networkx import DiGraph, Graph
from unit import fitness, fitness_rec_rem, fitness_rec_add, cache_rec_add, cache_rec_rem, is_acceptable_solution
class genetic_algorithm:
    def __init__(self, instance_name, k, graph: Graph, max_penalty, min_penalty, population_size, chromosome_length, mutation_rate, crossover_rate, tournament_size, elitism, time_limit, generation_max, max_no_improvment, rseed, loading=False):
        self.instance_name = instance_name
        self.k = k
        self.graph = graph
        self.penalty=min_penalty
        self.max_penalty = max_penalty
        self.min_penalty = min_penalty
        self.time_limit = time_limit
        self.generation_max = generation_max
        self.max_no_improvment = max_no_improvment
        self.chromosome_length = chromosome_length
        self.nodes = list(self.graph.nodes) 
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
        if self.loading:
            self.load_population()
            return
        for i in range(self.population_size):
            chromosome = [random.randint(0, 1) for j in range(self.chromosome_length)]
            self.population.append(self.local_search_best(chromosome))
        self.save_population()    
        #self.population = [self.local_search_best(chromosome) for chromosome in self.population]
        #for i in range(self.population_size // 2):
            #self.population[i] = self.local_search_best(self.population[i])
    
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
    
    def evolve(self):
        # Reduce the population size dynamically if conditions are met
        reduction_fraction = 0.1
        if  self.population_size > 30:
            # Sort population by fitness to keep the best individuals
            sorted_population = sorted(zip(self.population, self.fitness), key=lambda x: x[1])
            new_size = int(len(self.population) * (1 - reduction_fraction))
            self.population, self.fitness = zip(*sorted_population[:new_size])
            self.population = list(self.population)
            self.fitness = list(self.fitness)
            self.population_size = len(self.population)

        new_population = []
        if self.elitism:
            elite_chromosomes = self.elitism_selection()
            new_population.extend(elite_chromosomes)
        average_fitness = sum(f[0] for f in self.fitness) / len(self.fitness)
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
    
    def dynamic_penalty(self, generation):
        print("Generation: ", generation, "Max gen: ", self.generation_max)
        ratio = (generation-2) / self.generation_max
        return (1 - ratio) * self.min_penalty + ratio * self.max_penalty
    
    
    def run(self):
        start_time = time()
        best_time = 0
        generation = 1
        no_improvment = 0
        self.initialize_population()
        print("Initial population created, Time:", time() - start_time)
        self.evaluate_population()
        print("Initial population evaluated, Time:", time() - start_time)
        while time() - start_time < self.time_limit and generation < self.generation_max and no_improvment < self.max_no_improvment:
            print("Current generation:", generation, "Time:", time() - start_time)
            oldBestFitness = self.best_fitness
            generation += 1
            self.evolve()
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
            
    
        print("Chromosome acceptable: ", is_acceptable_solution(self.graph, self.chromosone_to_set(self.best_chromosome), self.k))
        print("Best fitness:", self.best_fitness, "Time:", time() - start_time, "Generation:", generation)
        return self.best_chromosome, self.best_fitness
    

if __name__ == '__main__':
    arguments={'instance_dir': "cities_small_instances",'instance':"oxford.txt", 'k':2, 'time_limit':600, 'generation_max':150, 'max_no_improvment': 5,'rseed': 78, 'population_size': 200, 'mutation_rate': 0.05, 'crossover_rate': 0.85, 'tournament_size': 4, 'elitism': True, 'max_penalty': 1, 'min_penalty': 0.01}
    
    graph_open = arguments["instance_dir"] + '/' + arguments["instance"]
    print("Reading graph!")
    g = read_graph(graph_open)
    print("Graph loaded: ", graph_open)

    ga = genetic_algorithm(arguments['instance'], arguments['k'], g, arguments['max_penalty'], arguments['min_penalty'], arguments['population_size'], g.number_of_nodes(), arguments['mutation_rate'], arguments['crossover_rate'], arguments['tournament_size'], arguments['elitism'], arguments['time_limit'], arguments['generation_max'], arguments['max_no_improvment'], arguments['rseed'], False)

    start_time = time()
    best_chromosome, best_fitness = ga.run()
    print("Final best fitness:", best_fitness)
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