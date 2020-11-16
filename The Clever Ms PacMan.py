from random import random

import gym
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
    env = gym.make('MsPacman-ram-v0')
    env.reset()


# Give every genome a fitness score
def selection(pop):
    for genome in pop.genomes:
        # Create NN from genome
        pheno = neat.FFNN(genome)

        # Start new instance of atari environment
        observation = env.reset()

        # run environment for certain timeframe
        for t in range(1000):

            if render:
                env.render()
                time.sleep(.01)

            # Run NN based on ram from environment
            action = pheno.activate(observation)
            print(action)
            # input action
            observation, reward, done, info = env.step(action[0]*10)
            # calculate fitness (can be changed)
            genome.fitness += reward / 10 + 2
            # ends test if game over
            if done:
                break

        else:
            # runs only if agent did not die within timeframe
            genome.fitness += 30

        env.close()


if __name__ == '__main__':
    # Start environment<
    start()

    # Create new population
    population = neat.Population(pop_size)

    # Start evolution
    finished = False
    while not finished:
        # Give genomes fitness score
        selection(population)

        # Adjust fitness, determine whether we reached desired score
        genomes, finished = neat.evaluate(population)
        if finished:
            break

        # Exterminate worst genomes
        # Replace exterminated with offspring from surviving genomes
        neat.renew_population(population)

        # mutations: adjust weights, add new nodes and connections
        for genome in population.genomes:
            neat.mutate(genome, population)
