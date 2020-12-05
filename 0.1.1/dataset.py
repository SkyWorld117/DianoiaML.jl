import numpy as np
import random

class Datagenerator:
    
    def __init__(self, runtimes = 100):
        self.runtimes = runtimes
        self.x = np.zeros((runtimes, ))
        self.y = np.zeros((runtimes, ))

    def run(self):
        for i in range(self.runtimes):
            self.x[i] = random.uniform(0, 2*np.pi)
            self.y[i] = np.sin(self.x[i])
        self.x = np.reshape(self.x, (self.runtimes, 1))
        self.y = np.reshape(self.y, (self.runtimes, 1))

if __name__ == '__main__':
    dg = Datagenerator()
    dg.run()
    print(dg.x)
    print(dg.y)