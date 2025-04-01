from time import time
from random import shuffle, random, seed
from read_graph import read_graph
from networkx import DiGraph, Graph
from unit import fitness, fitness_rec_rem, fitness_rec_add, cache_rec_add, cache_rec_rem
import sys
class genetic_algorithm:
    def __init__(self, instance_name, graph: Graph, population_size, chromosome_length, mutation_rate, crossover_rate, tournament_size, elitism, time_limit, iteration_max, rseed):
        self.instance_name = instance_name
        self.graph = graph
        self.time_limit = time_limit
        self.iteration_max = iteration_max
        self.nodes = list(self.graph.nodes) # kopiram cvorove zbog MJESANJA - necu da mjesam original
        self.rseed = rseed
        seed(self.rseed)

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
            
    def initialize_population(self):
        for i in range(self.population_size):
            chromosome = [random.randint(0, 1) for j in range(self.chromosome_length)]
            self.population.append(chromosome)
        

    def evaluate_population(self):
        self.fitness = []
        for chromosome in self.population:
            fitness = self.fitness_function(chromosome)
            self.fitness.append(fitness)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_chromosome = chromosome

    def fitness_function(self, chromosome):
        return sum(chromosome)

    def tournament_selection(self):
        tournament = []
        for i in range(self.tournament_size):
            tournament.append(random.randint(0, self.population_size - 1))
        best_chromosome = tournament[0]
        for i in tournament:
            if self.fitness[i] > self.fitness[best_chromosome]:
                best_chromosome = i
        return self.population[best_chromosome]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(0, self.chromosome_length - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        return parent1, parent2

    def mutate(self, chromosome):
        mutated_chromosome = []
        for gene in chromosome:
            if random.random() < self.mutation_rate:
                mutated_chromosome.append(1 - gene)
            else:
                mutated_chromosome.append(gene)
        return mutated_chromosome

    def evolve(self):
        new_population = []
        if self.elitism:
            new_population.append(self.best_chromosome)
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        self.population = new_population    

    
    def first_fitness_better(self, fit1, fit2):
        fit1Tot = (1+fit1[0])*(1+fit1[1]*self.penalty)
        fit2Tot = (1+fit2[0])*(1+fit2[1]*self.penalty)
        return fit1Tot<fit2Tot
    
    def fitness_equal(self, fit1, fit2):
        return not self.first_fitness_better(fit1, fit2) and not self.first_fitness_better(fit2, fit1)

    def local_search_best(self, s: set):
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

        return curr_fit

    def run(self, generations):
        self.initialize_population()
        self.evaluate_population()
        for i in range(generations):
            self.evolve()
            self.evaluate_population()
        return self.best_chromosome, self.best_fitness
if __name__ == '__main__':
    arguments={'instance_dir': "cities_small_instances",'instance':"oxford.txt", 'time_limit':600, 'iteration_max':10000,'rseed': 42, 'population_size': 100, 'chromosome_length': 200, 'mutation_rate': 0.01, 'crossover_rate': 0.85, 'tournament_size': 5, 'elitism': True}
    
    graph_open = arguments["instance_dir"] + '/' + arguments["instance"]
    print("Reading graph!")
    g = read_graph(graph_open)
    print("Graph loaded: ", graph_open)
    #g = read_graph("random_instances/NEW-V200-P0.2-G0.txt")
    ga = genetic_algorithm(arguments['instance'], g, arguments['population_size'], arguments['chromosome_length'], arguments['mutation_rate'], arguments['crossover_rate'], arguments['tournament_size'], arguments['elitism'], arguments['time_limit'], arguments['iteration_max'], arguments['rseed'])
    start_time = time()
    best_chromosome, best_fitness = ga.run(100)
    print("Best chromosome:", best_chromosome)
    print("Best fitness:", best_fitness)