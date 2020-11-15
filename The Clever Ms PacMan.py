from random import random

import gym
import nn
import neat

import time
import copy

import configparser
import random

# Configurations
config = configparser.ConfigParser()
config.read('config.ini')

pop_size = int(config['Environment']['population_size'])
render = bool(config['Environment']['render'])

env = None


# Initialize atari environment
def start():
    global env
    env = gym.make('MsPacMan-ram-v0')
    env.reset()


# Give every genome a fitness score
def selection():
    for genome in genomes:
        # Create NN from genome
        pheno = nn.generate(genome)

        # Start new instance of atari environment
        observation = env.reset()

        # run environment for certain timeframe
        for t in range(1000):

            if render:
                env.render()
                time.sleep(.01)

            # Run NN based on ram from environment
            action = pheno.run(observation)
            # input action
            observation, reward, done, info = env.step(action)
            # calculate fitness (can be changed)
            genome.fitness += reward / 10 + 2
            # ends test if game over
            if done:
                break

        else:
            # runs only if agent did not die within timeframe
            genome.fitness += 30

        env.close()


# adjust fitness and determine if desired fitness is reached
def evaluate():
    pass


# exterminate worst genomes (reduce population)
def exterminate():
    pass


# create new genomes based on two living genomes (restore population)
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

    # Not necessary since we don't change node properties.
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


if __name__ == '__main__':
    # Start environment
    start()

    # Create new population
    population = neat.Population(pop_size)

    # Start evolution
    finished = False
    while not finished:
        # Give genomes fitness score
        genomes = selection()

        # Adjust fitness, determine whether we reached desired score
        genomes, finished = evaluate()
        if finished:
            break

        # Exterminate worst genomes
        genomes = exterminate()

        # Replace exterminated with offspring from surviving genomes
        genomes = crossover()

        # mutations: adjust weights, add new nodes and connections
        genomes = neat.mutate()
