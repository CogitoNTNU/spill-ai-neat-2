# NeuroEvolution of Augmented Topologies (NEAT)
# Paper: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

#import nn

import configparser
from random import uniform

# Configurations
config = configparser.ConfigParser()
config.read('config.ini')

n_input = int(config['Genome']['n_input'])
n_output = int(config['Genome']['n_output'])
w_init_min = float(config['Genome']['weight_init_max'])
w_init_max = float(config['Genome']['weight_init_min'])

comp_thres = float(config['Species']['compatibility_threshold'])


class Node:
    def __init__(self, network_type, id):
        self.type = network_type
        self.id = id


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
        return nn.create(self)


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


e = 2.71828182845904523536


def sigmoid(x):
    return 1 / (1 + e**(-x))


def tanh(x):
    return (e**x - e**-x) / (e**x + e**-x)


def relu(x):
    return max(0, x)


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


"""
xorgenome = Genome([Node("input", 0), Node("input", 1), Node("hidden", 2), Node("hidden", 3), Node("output", 4, sigmoid)],
                   [Connection(0, 2, 1), Connection(0, 3, -1), Connection(1, 2, 1), Connection(1, 3, -1),
                    Connection(2, 4, 1), Connection(3, 4, 1)])

xorffnn = FFNN(xorgenome)

print(xorffnn.activate([1, 0]))
"""