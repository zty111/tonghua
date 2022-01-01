from experience import load_experience, combine_experience
from multiprocessing import Process
from play import train
from tiaocan import learning_rate
exp = combine_experience([load_experience('exp.h5')])
train(exp, learning_rate, 2048)