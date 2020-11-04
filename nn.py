import torch


def sigmoid(x):
    return 1/(1+torch.exp(-x))

class Node:

class FFNN:
    def __init__(self, population):
        self.input_nodes = [node for node in population.nodes if node.type == 'input']
        self.input_nodes.sort(key=lambda node: node.id)
        self.nodes = [node for node in population.nodes if node.type != 'input']
        self.connects = population.connections
    def activate(self, observation):


def create(population):
    return FFNN(population)