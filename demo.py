from experience import load_experience, combine_experience
from multiprocessing import Process
from play import get_exp, train
from tiaocan import ker_num, learning_rate
import os

if __name__ == '__main__':
    gener = 0
    while True:
        gener += 1
        print(f"=======gener:{gener}===========")

        
        hand_pool = []
        for i in range(ker_num):
            p = Process(target = get_exp, args=(i,))
            p.start()
            hand_pool.append(p)
        for p in hand_pool:
            p.join()
        
        exp = []
        for i in range(ker_num):
            exp.append(load_experience(f"exp{i}.h5"))
        if os.path.exists('exp.h5'):
            exp.append(load_experience('exp.h5'))
        exp = combine_experience(exp)
        exp.serialize('exp.h5')
        p = Process(target = train, args = (exp, learning_rate, 2048))
        p.start()
        p.join()