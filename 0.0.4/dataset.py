import numpy as np
import random
from counter import s

class Datagenerator:

    def __init__(self, runtimes = 100):
        self.runtimes = runtimes
        self.x = np.zeros((runtimes, 3))
        self.y = np.zeros((runtimes, len(s)))

    def run(self):
        for i in range(self.runtimes):
            self.x[i][0] = random.randint(0,9)
            self.x[i][1] = random.randint(0,3)
            self.x[i][2] = random.randint(0,9)
            if self.x[i][1] == 0:
                self.y[i][s.index(self.x[i][0]+self.x[i][2])] = 1
            elif self.x[i][1] == 1:
                self.y[i][s.index(self.x[i][0]-self.x[i][2])] = 1
            elif self.x[i][1] == 2:
                self.y[i][s.index(self.x[i][0]*self.x[i][2])] = 1
            elif self.x[i][1] == 3:
                if self.x[i][2] != 0:
                    self.y[i][s.index(self.x[i][0]//self.x[i][2])] = 1
                else:
                    self.y[i][s.index(None)] = 1
