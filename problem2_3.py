import sys
import math
import csv
# import matplotlib.pyplot as plt


class linear_regression:
    def __init__(self):
        self.inputcsv = sys.argv[1]
        self.outputcsv = sys.argv[2]

        self.alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.05]
        self.iters = [100,100,100,100,100,100,100,100,100, 200]

        self.age = []
        self.weight = []
        self.height = []

        self.read()

    def read(self):
        with open(self.inputcsv, newline='') as input_f:
            freader = csv.reader(input_f)
            for row in freader:
                row = list(map(float, row))
                self.age.append(row[0])
                self.weight.append(row[1])
                self.height.append(row[2])

        self.normalize(self.age)
        self.normalize(self.weight)
    
    def normalize(self, l):
        # norm = (x - mean) / stdev
        suml = 0
        for i in l:
            suml += i
        mean = suml/len(l)
        # print ('m', mean)
        sumsq = 0
        for i in l:
            sumsq += (i - mean)**2
        stdev = math.sqrt(sumsq / (len(l)))
        for i in range(len(l)): 
            l[i] = (l[i] - mean) / stdev

        # print ('d', stdev)

    def run(self, test_num):

        iterations = self.iters[test_num]
        alpha = self.alphas[test_num]
        age = self.age
        weight = self.weight
        height = self.height

        b_0 = 0
        b_age = 0
        b_weight = 0

        for a in range(iterations):
            
            sum_b0 = 0
            sum_bage = 0
            sum_bweight = 0

            for i in range(len(age)):
                fxi = b_0 + b_age * age[i] + b_weight * weight[i] - height[i]
                sum_b0 += fxi
                sum_bage += fxi * age[i]
                sum_bweight += fxi * weight[i]
            
            grad_k = alpha * 1 / len(age)

            b_0 = b_0 - grad_k * sum_b0
            b_age = b_age - grad_k * sum_bage
            b_weight = b_weight - grad_k * sum_bweight 

        outrow = list(map(str, [alpha, self.iters[test_num], b_0, b_age, b_weight]))
        
        return outrow


init = linear_regression()

with open (init.outputcsv, 'w', newline='') as csvfile:
    outwriter = csv.writer(csvfile)

    for i in range(10):
        res = init.run(i)
        print (res)
        outwriter.writerow(init.run(i))
