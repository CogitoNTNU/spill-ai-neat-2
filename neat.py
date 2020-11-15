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

comp_thres = float(config['Species']['compatibility_threshold'])

prob_mut_w = float(config['Mutation']['weight_probability'])
mut_w_min = float(config['Mutation']['weight_min_change'])
mut_w_max = float(config['Mutation']['weight_max_change'])

prob_mut_n = float(config['Mutation']['node_probability'])
prob_mut_c = float(config['Mutation']['connection_probability'])
prob_mut_a = float(config['Mutation']['activate_probability'])
prob_mut_d = float(config['Mutation']['deactivate_probability'])


e = 2.71828182845904523536


def sigmoid(x):
    return 1 / (1 + e**(-x))


def tanh(x):
    return (e**x - e**-x) / (e**x + e**-x)


def relu(x):
    return max(0, x)


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
        fitness = 0

    def create_nn(self):
        return FFNN(self)


# Store genomes and common properties
class Population:
    def __init__(self, size):

        self.genomes = []
        self.species = []
        self.innov_number = 0
        self.node_number = 1

        # Generate first population
        for _ in range(size):

            nodes = []
            connections = []

            for _ in range(n_input):
                nodes.append(Node('input', self.node_number))
                self.node_number += 1
            for _ in range(n_output):
                outp = Node('output')
                nodes.append(outp)
                self.node_number += 1
                for inp in [node for node in nodes if node.type == 'input']:
                    connections.append(Connection(inp, outp, uniform(w_init_min, w_init_max)))
                    self.innov_number += 1

            self.species.append(self.genomes[0])
            self.genomes.append(Genome(nodes, connections))

        # First Speciation
        self.species = []


def comp_distance(genome_1, genome_2):  # Compatibility distance
    # Used for speciation. Compares two genomes and returns their similarity based on structure.
    """
    :param float E: Excess genes
    :param float D: Disjoint genes
    :param float N: Number of genes in larger genomes
    :param float W: Average weight of matching genes (including disabled)
    :return float: Compatibility distance
    """

    c_1 = 0.1
    c_2 = 0.1
    c_3 = 0.1
    N = N if N >= 20 else 1

    return c_1 * E / N + c_2 * D / N + c_3 * W


def adjusted_fitness(G):
    # Every genome will get a fitness score after testing. This function will adjust this score based on
    # its population size, such that smaller species do not instantly eradicate.
    """
    :param Genotype G: Genotype
    :return float:
    """
    return G.fitness / (sum(0 if cd > comp_thres else 1 for cd in [comp_distance(_, _, _, _) for G_j in Genes]))


def find_yourself(to_find, current_node, connections):
    connect_to = [c.o for c in connections if c.i == current_node]
    for next_node in connect_to:
        if next_node == to_find:
            return True
        if find_yourself(to_find, next_node, connections):
            return True
    return False

def crossover(genome1, genome2):
    if genome1.fitness > genome2.fitness:
        parent1, parent2 = genome1, genome2
    else:
        parent1, parent2 = genome2, genome1

    # implement the new genome
    new_gene = copy.deepcopy(parent1)

    # Inherit connection genes
    old_c = []
    for i, c1 in enumerate(parent1.connections):
        c2 = filter(lambda c: c.innov == c1.innov, parent2.connections)
        if not len(c2):
            new_gene.connections.append(copy.copy(c1))
        else:
            old_c.append(i)

            new_w = c1.w if random().random() > 0.5 else c2.w
            new_active = c1.active if random().random() > 0.5 else c2.active

            new_gene.connections.append(neat.Connection(c1.i, c1.o, new_w, c1.innov))
            new_gene.connections[-1].active = new_active

    for i in old_c[::-1]:
        del new_gene.connections[i]

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
        i_nodes = [n for n in genome.nodes if n.type != 'output']
        random.shuffle(i_nodes)
        for i_node in i_nodes:

            o_nodes = [n for n in genome.nodes if n.id != i_node.id and
                      n.type != 'input' and
                      filter(lambda c: c.i == i_node.id and c.o == n.id, genome.connections) and
                      not find_yourself(i_node, n.id, genome.connections)]

            if len(o_nodes):
                o_node = random.choice(o_nodes)
                genome.connections.append(
                    Connection(i_node.id, o_node.id, random.uniform(mut_w_min, mut_w_max), population.innov_number))
                population.innov_number += 1
                break


def find_parents(node, nodes, connections):
    parents_c = [c for c in connections if c.o == node.id and c.active]
    ids = [c.i for c in parents_c]
    return [n for n in nodes if n.id in ids], parents_c


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

xorgenome = Genome([Node("input", 0), Node("input", 1), Node("hidden", 2), Node("hidden", 3), Node("output", 4, sigmoid)],
                   [Connection(0, 2, 1, 0), Connection(0, 3, -1, 1), Connection(1, 2, 1, 2), Connection(1, 3, -1, 3),
                    Connection(2, 4, 1, 4), Connection(3, 4, 1, 5)])

xorffnn = FFNN(xorgenome)

pop = Population(0)
pop.genomes.append(xorgenome)
pop.innov_number = 6
pop.node_number = 5
print_genome(xorgenome)
mutate(xorgenome, pop)
print_genome(xorgenome)
print(xorffnn.activate([1, 0]))
