# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:12:04 2020

@author: Zhou
"""

class Lunzi(object):
    def __init__(self, name):
        self.name = name
        
    def whoami(self):
        print(self.name)


class Che(object):
    def __init__(self, name):
        self.name = name
        self.lunzimen = [Lunzi("lunzi{}".format(i)) for i in range(self.get_n_lunzi())]
        
    def whoami(self):
        print(self.name)
        
    def get_n_lunzi(self):
        return 4
    
    def print_lunzi(self):
        for l in self.lunzimen:
            l.whoami()
        
        
class Jiaoche(Che):
    def __init__(self, name):
        super().__init__(name)
        

if __name__ == "__main__":
    zhen_jiaoche = Jiaoche("hahaha")
    zhen_jiaoche.whoami()
    zhen_jiaoche.print_lunzi()

