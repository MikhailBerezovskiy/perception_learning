import sys
import csv
# import matplotlib.pyplot as plt


class perception_learning:
    def __init__(self):
        self.inputcsv = sys.argv[1]
        self.outputcsv = sys.argv[2]
        self.D = []
        self.x = []
        self.y = []
        self.col = []
        self.w_1 = 0
        self.w_2 = 0
        self.b = 0


        with open(self.inputcsv, newline='') as input_f:
            freader = csv.reader(input_f)
            for row in freader:
                x_j = [int(row[0]), int(row[1])]
                d_j = int(row[2])
                if d_j == -1:
                    d_j = 0
                self.D.append([x_j, d_j])
                self.x.append(int(row[0]))
                self.y.append(int(row[1]))
                if row[2] == '1':
                    self.col.append('r')
                else:
                    self.col.append('b')
        
# print (D)

    def learn(self):
        convergence = False
        with open(self.outputcsv, 'w', newline='') as csvfile:
            outwriter = csv.writer(csvfile)
            while not convergence:
                convergence = True
                for j in range(len(self.D)):
                    x_j = self.D[j][0]
                    d_j = self.D[j][1]
                    wt_x = self.b + self.w_1 * x_j[0] + self.w_2 * x_j[1] 
                    if wt_x > 0:
                        y_j = 1
                    else:
                        y_j = 0
                    # print ('j=',j, 'xj=',x_j, 'dj=', d_j, 'wtx=', wt_x, 'yj=', y_j)
                    if y_j != d_j:
                        convergence = False
                        err = (d_j - y_j)
                        self.b = self.b + err
                        self.w_1 = self.w_1 + err * x_j[0]
                        self.w_2 = self.w_2 + err * x_j[1]
                        # print ('b=',b, 'w1=', self.w_1, 'w2=', self.w_2)
                    # print ('______________________')
                # print ('w1=', self.w_1, 'w2=', self.w_2, 'b=', self.b)
                outrow = map(str, [self.w_1, self.w_2, self.b])
                # print (outrow)
                outwriter.writerow(outrow)
       # self.plotresults()


    # def line (self, x):
    #     if self.w_2 == 0:
    #         self.w_2 = 1
    #     return - (self.w_1*x + self.b) / (self.w_2) 

    # def plotresults(self):
    #     plt.plot([1,15], [self.line(1), self.line(15)])
    #     plt.scatter(self.x, self.y, s=None, c=self.col)
    #     plt.show()


perception_learning_algo = perception_learning()
perception_learning_algo.learn()