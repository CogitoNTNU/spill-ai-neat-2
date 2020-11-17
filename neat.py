# NeuroEvolution of Augmented Topologies (NEAT)
# Paper: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

#import nn

import configparser
import copy
import random
from random import uniform

# Configurations
config = configparser.ConfigParser()
config.read('config.ini')

n_input = int(config['Genome']['n_input'])
n_output = int(config['Genome']['n_output'])
w_init_min = float(config['Genome']['weight_init_max'])
w_init_max = float(config['Genome']['weight_init_min'])

max_generations = float(config['Goals']['max_generation'])
goal_fitness = float(config['Goals']['goal_fitness'])

comp_thres = float(config['Species']['compatibility_threshold'])

prob_mut_w = float(config['Mutation']['weight_probability'])
mut_w_min = float(config['Mutation']['weight_min_change'])
mut_w_max = float(config['Mutation']['weight_max_change'])

prob_mut_n = float(config['Mutation']['node_probability'])
prob_mut_c = float(config['Mutation']['connection_probability'])
prob_mut_a = float(config['Mutation']['activate_probability'])
prob_mut_d = float(config['Mutation']['deactivate_probability'])


e = 2.71828182845904523536

# Activation functions
def sigmoid(x):
    return 1 / (1 + e**(-x))


def tanh(x):
    return (e**x - e**-x) / (e**x + e**-x)


def relu(x):
    return max(0, x)

# Genome component classes
class Node:
    def __init__(self, network_type, id, activation_f=relu):
        self.type = network_type
        self.id = id
        self.activation_f = activation_f


class Connection:
    def __init__(self, i, o, w, innov):
        self.i = i
        self.o = o
        self.w = w
        self.active = 1

        self.innov = innov


class Genome:
    def __init__(self, nodes, connections):
        self.nodes = nodes
        self.connections = connections
        self.fitness = 0

    def create_nn(self):
        return FFNN(self)


# Genome presentation
def print_genome(genome):
    i = list([n.id for n in genome.nodes if n.type == 'input'])
    h = list([n.id for n in genome.nodes if n.type == 'hidden'])
    o = list([n.id for n in genome.nodes if n.type == 'output'])
    print(f'{len(genome.nodes)} Nodes:')
    print(len(i), 'input:')
    print(i)
    print(len(h), 'hidden:')
    print(h)
    print(len(o), 'output:')
    print(o)
    print(f'{len(genome.connections)} Connections')
    for c in genome.connections:
        print(f"{c.i} -> {c.o} | w: {c.w}, active: {c.active}")


# Compatibility distance
# Used for speciation. Compares two genomes and returns their similarity based on structure.
# Full explanation in paper
def comp_distance(genome1, genome2):
    # Find values for constants
    matching_c = [c for c in genome1.connections if c.innov in [c.innov for c in genome2.connections]]
    matching_c_innov = [c.innov for c in matching_c]
    disjoint = []
    excess = []
    # Runs only if disjoint or excess connections exist
    if not len(matching_c_innov) == len(genome1.connections) == len(genome2.connections):
        not_matching_c1_innov = [c.innov for c in genome1.connections if c.innov not in matching_c_innov]
        not_matching_c2_innov = [c.innov for c in genome2.connections if c.innov not in matching_c_innov]
        not_matching_c1_innov = not_matching_c1_innov if len(not_matching_c1_innov) else [-1]
        not_matching_c2_innov = not_matching_c2_innov if len(not_matching_c2_innov) else [-1]
        disjoint_max = min(max(not_matching_c1_innov), max(not_matching_c2_innov))

        disjoint = [innov for innov in not_matching_c1_innov + not_matching_c2_innov if innov <= disjoint_max]
        excess = [innov for innov in not_matching_c1_innov + not_matching_c2_innov if innov > disjoint_max]

    c_1 = 0.1
    c_2 = 0.1
    c_3 = 0.1

    E = len(excess) # Excess connections
    D = len(disjoint) # Disjoint connections
    N = max(len(genome1.nodes), len(genome2.nodes)) # Number of nodes
    N = N if N >= 20 else 1
    W = sum([c.w for c in matching_c])/len(matching_c) if len(matching_c) else 0 # Average weight between matching connections

    # Speciation formulae from paper
    return c_1 * E / N + c_2 * D / N + c_3 * W


# Store genomes and common properties
class Population:
    def __init__(self, size):

        self.genomes = []
        self.species = []
        self.innov_number = 0
        self.node_number = 1
        self.size = size
        self.generation = 0

        # Generate first population
        for i in range(size):

            nodes = []
            connections = []

            for _ in range(n_input):
                nodes.append(Node('input', self.node_number))
                self.node_number += 1
            for _ in range(n_output):
                outp = Node('output', self.node_number)
                nodes.append(outp)
                self.node_number += 1
                for inp in [node for node in nodes if node.type == 'input']:
                    connections.append(Connection(inp, outp, uniform(w_init_min, w_init_max), self.innov_number))
                    self.innov_number += 1

            self.genomes.append(Genome(nodes, connections))

            # First Speciation
            if i == 0:
                self.species = [[self.genomes[0]]]
            else:
                for specie in self.species:
                    if comp_distance(self.genomes[-1], specie[0]) < comp_thres:
                        specie.append(self.genomes[-1])
                        break
                    else:
                        self.species.append([self.genomes[-1]])


def adjusted_fitness(genome, species):
    # Every genome will get a fitness score after testing. This function will adjust this score based on
    # its population size, such that smaller species do not instantly eradicate.
    return genome.fitness / len(species)


def evaluate(population):
    population.genomes.sort(key=lambda g: g.fitness)
    # Has a genome reached the goal fitness?
    if population.genomes[0] >= goal_fitness:
        return population.genomes, True
    # Have the simulaiton ran out of generations?
    if population.generation >= max_generations:
        return population.genomes, True
    return False


def crossover(genome1, genome2):
    if genome1.fitness > genome2.fitness:
        parent1, parent2 = genome1, genome2
    else:
        parent1, parent2 = genome2, genome1

    # implement the new genome
    new_gene = copy.deepcopy(parent1)

    # Inherit connection genes
    new_gene.connections = []
    for i, c1 in enumerate(parent1.connections):
        print(c1)
        c2 = list(filter(lambda c: c.innov == c1.innov, parent2.connections))
        if not len(c2):
            # Inferior genome does not have connection, append it from superior
            new_gene.connections.append(copy.copy(c1))
        else:
            # Both genomes have the connection. Choose randomly which properties are inherited from which genomes
            c2 = c2[0]

            new_w = c1.w if random.random() > 0.5 else c2.w
            new_active = c1.active if random.random() > 0.5 else c2.active
            new_c = Connection(c1.i, c1.o, new_w, c1.innov)
            new_c.active = new_active
            new_gene.connections.append(new_c)

    # Inherit node properties
    # Not necessary since we don't change node properties (all are inherited from superior).
    """ 
    # Inherit node genes
    old_n = []
    for i, n1 in enumerate(parent1.nodes):
        n2 = filter(lambda n: n.id == n1.id, parent2.nodes)
        if not len(n2):
            new_gene.nodes.append(copy.copy(n1))
        else:
            old_n.append(i)
            new_gene.nodes.append(neat.Node(n1.id, n1.type))

    for i in old_c[::-1]:
        del new_gene.nodes[i]
    """

    return new_gene


def renew_population(population):
    # Adjust fitness based on the size of their population
    for s in population.species:
        for g in s:
            g.fitness = adjusted_fitness(g, s)

    # Remove worst genomes
    population.genomes.sort(key=lambda g: g.fitness)
    dead = population.genomes[population.size//2:]
    population.genomes = population.genomes[:population.size//2]

    # Generate new genomes using crossover of two remaining genomes
    for n in range(len(dead)):
        population.genomes.append(crossover(random.choice(population.genomes), random.choice(population.genomes)))

    # Speciation
    population.species = [[population.genomes[0]]]
    for genome in population.genomes[1:]:
        for specie in population.species:
            if comp_distance(genome, specie[0]) < comp_thres:
                specie.append(genome)
                break
        else:
            population.species.append([genome])

    # Delete from memory
    for d in dead:
        del d

    population.generation += 1

# Used to ensure no new connections lead to loops
def find_yourself(to_find, current_node, connections):
    connect_to = [c.o for c in connections if c.i == current_node]
    for next_node in connect_to:
        if next_node == to_find:
            return True
        if find_yourself(to_find, next_node, connections):
            return True
    return False


def mutate(genome, population):

    # Weight mutation
    for c in genome.connections:
        if random.random() < prob_mut_w:
            c.w += random.uniform(-1, 1)
        if random.random() < prob_mut_a:
            c.active = 1
        if random.random() < prob_mut_d:
            c.active = 0

    # Node mutation
    if random.random() < prob_mut_n:
        c_chosen = random.choice(genome.connections)  # choose which nodes will get a new node between them
        genome.nodes.append(Node('hidden', population.node_number))  # create new node in genome
        population.node_number += 1
        # add connection from i-node in chosen connection to new node,
        genome.connections.append(Connection(c_chosen.i, genome.nodes[-1].id, c_chosen.w, population.innov_number))
        population.innov_number += 1
        # add connection from new node to i-node in chosen connection,
        genome.connections.append(Connection(genome.nodes[-1].id, c_chosen.o, c_chosen.w, population.innov_number))
        population.innov_number += 1
        # deactivate chosen connection
        c_chosen.active = 0

    # Connection mutation
    if random.random() < prob_mut_c:
        # Find potential input nodes
        i_nodes = [n for n in genome.nodes if n.type != 'output']
        random.shuffle(i_nodes)
        for i_node in i_nodes:
            # Find potential output nodes
            o_nodes = [n for n in genome.nodes if n.id != i_node.id
                       and n.type != 'input'
                       # Check if connection already exists (optimize plox)
                       and not len(list(filter(lambda c: c.i == i_node.id and c.o == n.id, genome.connections)))
                       # Check if connection leads to loop
                       and not find_yourself(i_node, n.id, genome.connections)]
            # If any exist, choose one of them
            if len(o_nodes):
                o_node = random.choice(o_nodes)
                genome.connections.append(
                    Connection(i_node.id, o_node.id, random.uniform(mut_w_min, mut_w_max), population.innov_number))
                population.innov_number += 1
                break


# NEURAL NETWORK (Phenome)


def find_parents(node, nodes, connections):
    parents_c = [c for c in connections if c.o == node.id and c.active]
    ids = [c.i for c in parents_c]
    return [n for n in nodes if n.id in ids], parents_c


# Recursive function for neural network. Probably one of the worst offenders optimally in this code:3
def get_value(node, nodes, connections, observation):
    if node.type == "input":
        return observation[node.id]

    parents, parents_c = find_parents(node, nodes, connections)
    parents_v = [get_value(parent, nodes, connections, observation) for parent in parents]
    parents_v_sum = sum([parents_v[i] * parents_c[i].w for i in range(len(parents_v))])
    return node.activation_f(parents_v_sum)


class FFNN:
    def __init__(self, genome):
        self.input_nodes = [node for node in genome.nodes if node.type == 'input']
        self.input_nodes.sort(key=lambda node: node.id)
        self.output_nodes = [node for node in genome.nodes if node.type == 'output']
        self.input_nodes.sort(key=lambda node: node.id)
        self.nodes = [node for node in genome.nodes]
        self.connections = genome.connections

    def activate(self, observation):
        values = []
        for output in self.output_nodes:
            values.append(get_value(output, self.nodes, self.connections, observation))
        return values

# Testing area
"""
xorgenome = Genome([Node("input", 0), Node("input", 1), Node("hidden", 2), Node("hidden", 3), Node("output", 4, sigmoid)],
                   [Connection(0, 2, 1, 0), Connection(0, 3, -1, 1), Connection(1, 2, 1, 2), Connection(1, 3, -1, 3),
                    Connection(2, 4, 1, 4), Connection(3, 4, 1, 5)])

xorgenome_2 = copy.deepcopy(xorgenome)

xorffnn = FFNN(xorgenome)
print(xorffnn.activate([1, 0]))

pop = Population(0)
pop.genomes.append(xorgenome)
pop.genomes.append(xorgenome_2)
pop.innov_number = 6
pop.node_number = 5

print_genome(xorgenome)
print_genome(xorgenome_2)
mutate(xorgenome_2, pop)
xorgenome_2.fitness = 0
print_genome(xorgenome_2)
new_xorgenome = crossover(xorgenome, xorgenome_2)
print_genome(new_xorgenome)

print(comp_distance(xorgenome, new_xorgenome))
"""