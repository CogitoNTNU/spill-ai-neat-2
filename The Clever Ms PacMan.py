from random import random

import gym
import nn
import neat

import time

import configparser

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
    new_gene = neat.Genome.__class__(parent1.key)

    # Inherit connection genes
    for key, cg1 in parent1.connections.items():
        cg2 = parent2.connections.get(key)
        if cg2 is None:
            new_gene.connections[key] = cg1.copy()
        else:
            new_gene.connections[key] = crossover(cg1, cg2)

    # Inherit node genes
    parent1_set = parent1.nodes
    parent2_set = parent2.nodes

    for key, ng1 in parent1_set.items():
        ng2 = parent2_set.get(key)
        assert key not in new_gene.nodes
        if ng2 is None:
            new_gene.nodes[key] = ng1.copy()
        else:
            # Homologous gene: combine genes from both parents.
            new_gene.nodes[key] = crossover(ng1, ng2)
    return new_gene


# mutations to weights, add new nodes and connections
def mutate():
    pass


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
        genomes = mutate()
