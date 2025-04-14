from time import time
import random
from read_graph import read_graph
from networkx import DiGraph, Graph
from unit import fitness, fitness_rec_rem, fitness_rec_add, cache_rec_add, cache_rec_rem, is_acceptable_solution
class genetic_algorithm:
    def __init__(self, instance_name, k, graph: Graph, penalty, population_size, chromosome_length, mutation_rate, crossover_rate, tournament_size, elitism, time_limit, generation_max, max_no_improvment, rseed):
        self.instance_name = instance_name
        self.k = k
        self.graph = graph
        self.penalty = penalty
        self.time_limit = time_limit
        self.generation_max = generation_max
        self.max_no_improvment = max_no_improvment
        self.nodes = list(self.graph.nodes) # kopiram cvorove zbog MJESANJA - necu da mjesam original
        self.rseed = rseed
        random.seed(self.rseed)

        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.population = []
        self.fitness = []
        self.best_chromosome = []
        self.best_fitness = 0

        self.neighbors = {}
        self.neighb_matrix = [[] for _ in range(len(self.graph.nodes))]
        for v in self.graph.nodes:
            self.neighbors[v] = set(self.graph.neighbors(v))
            self.neighb_matrix[v] = [False]*len(self.graph.nodes)
            for u in self.graph.neighbors(v):
                self.neighb_matrix[v][u] = True

            
    def initialize_population(self):
        for i in range(self.population_size):
            chromosome = [random.randint(0, 1) for j in range(self.chromosome_length)]
            self.population.append(chromosome)
        for i in range(self.population_size // 2):
            self.population[i] = self.local_search_best(self.population[i])

    def evaluate_population(self):
        self.fitness = []
        self.best_fitness = self.fitness_function(self.population[0])
        number_of_accepted= 0
        for chromosome in self.population:
            fitness = self.fitness_function(chromosome)
            self.fitness.append(fitness)
            if self.first_fitness_better(fitness, self.best_fitness):
                self.best_fitness = fitness
                self.best_chromosome = chromosome
            if is_acceptable_solution(self.graph, self.chromosone_to_set(chromosome), self.k):
                number_of_accepted += 1
        print("Percentage of accepted solutions: ", number_of_accepted / self.population_size * 100)
        return self.best_fitness, self.best_chromosome
    
    def fitness_function(self, chromosome):
        s = set([i for i in range(len(chromosome)) if chromosome[i] == 1])
        return fitness(s, self.graph, self.k, {})
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
        new_population = []
        if self.elitism:
            elite_chromosomes = self.elitism_selection()
            new_population.extend(elite_chromosomes)
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child1, child2 = self.uniform_crossover(parent1, parent2)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
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
    arguments={'instance_dir': "cities_small_instances",'instance':"oxford.txt", 'k':2, 'time_limit':600, 'generation_max':100, 'max_no_improvment': 4,'rseed': 77, 'population_size': 400, 'chromosome_length': 200, 'mutation_rate': 0.05, 'crossover_rate': 0.85, 'tournament_size': 4, 'elitism': True, 'penalty': 0.005}
    
    graph_open = arguments["instance_dir"] + '/' + arguments["instance"]
    print("Reading graph!")
    g = read_graph(graph_open)
    print("Graph loaded: ", graph_open)

    ga = genetic_algorithm(arguments['instance'], arguments['k'], g, arguments['penalty'], arguments['population_size'], g.number_of_nodes(), arguments['mutation_rate'], arguments['crossover_rate'], arguments['tournament_size'], arguments['elitism'], arguments['time_limit'], arguments['generation_max'], arguments['max_no_improvment'], arguments['rseed'])

    start_time = time()
    best_chromosome, best_fitness = ga.run()
    print("Final best fitness:", best_fitness)