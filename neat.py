# NeuroEvolution of Augmented Topologies (NEAT)
# Paper: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

import nn

import configparser
from random import uniform

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


class Population:
    def __init__(self, size):

        self.genomes = []
        self.species = []
        self.innov_number = 0
        self.node_number = 1
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

        self.species = []

def comp_distance(genome_1, genome_2):  # Compatibility distance
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
    """
    :param Genotype G: Genotype
    :return float:
    """
    return G.fitness / (sum(0 if cd > comp_thres else 1 for cd in [comp_distance(_, _, _, _) for G_j in Genes]))

