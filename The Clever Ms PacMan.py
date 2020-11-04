import gym
import nn
import neat

import time

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

pop_size = int(config['Environment']['population_size'])
render = bool(config['Environment']['render'])


env = None
def start():
    global env
    env = gym.make('MsPacMan-ram-v0')
    env.reset()


def selection():
    for genome in genomes:
        pheno = nn.generate(genome)

        observation = env.reset()

        for t in range(1000):

            if render:
                env.render()
                time.sleep(.01)

            action = pheno.run(observation)

            observation, reward, done, info = env.step(action)

            genome.fitness += reward/10 + 2

            if done:
                break

        else:
            genome.fitness += 30

        env.close()


def evaluate():
    pass


def exterminate():
    pass


def crossover():
    pass


def mutate():
    pass


if __name__ == '__main__':

    start()

    neat.initialize(pop_size)

    finished = False
    while not finished:

        genomes = selection()

        if evaluate():
            finished = True
            #break?

        genomes = exterminate()

        genomes = crossover()

        genomes = mutate()
