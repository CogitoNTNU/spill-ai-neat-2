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
parallel_sims = int(config['Environment']['parallel_simulations'])

env = None


# Initialize atari environment
def start():
    global env
    env = gym.make('MsPacman-ram-v0')
    env.reset()


# Give every genome a fitness score
def selection(pop):
    for genome in pop.genomes:

        genome.fitness = 0

        # Create NN from genome
        pheno = neat.FFNN(genome)

        # Start new instance of atari environment
        observation = env.reset()

        # run environment for certain timeframe
        #for t in range(1000):
        while True:

            if render:
                env.render()
                time.sleep(.001)

            # Run NN based on ram from environment
            # input action
            action = pheno.activate(observation)[0]
            #print(action)
            action = neat.sigmoid(action) - 0.00001
            action = int(action*8)

            #if action == prev_action:
            #    genome.fitness -= 2
            observation, reward, done, info = env.step(action)
            # calculate fitness (can be changed)
            #if reward:
            #    print(reward)
            genome.fitness += reward # + 1
            # ends test if game over
            if done:
                break

        else:
            # runs only if agent did not die within timeframe
            genome.fitness += 10

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

        population.print_status()
        population.print_species()

        # Adjust fitness, determine whether we have reached a desired score
        genomes, finished = neat.evaluate(population)
        if finished:
            neat.print_genome(genomes[0])
            print(genomes[0].fitness)
            break

        # Exterminate worst genomes
        # Replace exterminated with offspring from surviving genomes
        neat.renew_population(population)

        # mutations: adjust weights, add new nodes and connections
        for genome in population.genomes:
            neat.mutate(genome, population)

    input("Finished:")
    selection(population)