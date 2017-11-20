import sys
import math
import csv
# import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class classification:
    def __init__(self):
        if len(sys.argv) == 1:
            self.inputcsv = 'input3.csv'
            self.outputcsv = 'output3.csv'
        else:
            self.inputcsv = sys.argv[1]
            self.outputcsv = sys.argv[2]
            
        self.sourceData = []
        self.X = []
        self.y = []
        self.read()
        
    def read(self):
        with open(self.inputcsv, newline='') as input_f:
            freader = csv.reader(input_f)
            for row in freader:
                if row[0] != 'A':
                    row = list(map(float, row))
                    self.sourceData.append(row)
                    self.X.append([row[0], row[1]])
                    self.y.append(row[2])

    def plot(self, X, y):
        lx,ly,c, s = [], [], y, X
        for i in range(len(s)):
            lx.append(s[i][0])
            ly.append(s[i][1])
            if c[i] == 1:
                c[i] = 'r'
            else:
                c[i] = 'b'
        plt.scatter(lx,ly, s=None, c=c)
        plt.show()

    def run(self):

        linear_params = [{ 
            'kernel': ['linear'], 
            'C': [0.1, 0.5, 1, 5, 10, 50, 100]
        }]

        poly_params = [{ 
            'kernel': ['poly'], 
            'C': [0.1, 1, 3], 
            'degree': [4,5,6], 
            'gamma': [0.1, 0.5]
        }]

        rbf_params = [{ 
            'kernel': ['rbf'], 
            'C': [0.1, 0.5, 1, 5, 10, 50, 100],  
            'gamma': [0.1, 0.5, 1, 3, 6, 10] 
        }]
        
        log_reg_params = [{
            'C': [0.1, 0.5, 1, 5, 10, 50, 100]
        }]

        kn_params = [{
            'n_neighbors': list(range(1,51)),
            'leaf_size': list(range(5,61,5))
        }]

        dt_params = [{
            'max_depth': list(range(1,51)),
            'min_samples_split': list(range(2,11))
        }]

        rf_params = [{
            'max_depth': list(range(1,51)),
            'min_samples_split': list(range(2,11))
        }]
        
        with open(self.outputcsv, 'w', newline='') as csvfile:
            outwriter = csv.writer(csvfile)
            
            # linear
            res_linear = self.estimator('svc_l', linear_params)
            print_linear = ['svm_linear', res_linear[0], res_linear[1]]
            outwriter.writerow(print_linear)
            print (print_linear)

            # poly
            res_poly = self.estimator('svc', poly_params)
            print_poly = ['svm_polynomial', res_poly[0], res_poly[1]]
            outwriter.writerow(print_poly)
            print (print_poly)

            # rbf
            res_rbf = self.estimator('svc', rbf_params)
            print_rbf = ['svm_rbf', res_rbf[0], res_rbf[1]]
            outwriter.writerow(print_rbf)
            print (print_rbf)

            # log_reg
            res_log_reg = self.estimator('lr', log_reg_params)
            print_log_reg = ['logistic', res_log_reg[0], res_log_reg[1]]
            outwriter.writerow(print_log_reg)
            print (print_log_reg)

            # knn
            res_kn = self.estimator('kn', kn_params)
            print_kn = ['knn', res_kn[0], res_kn[1]]
            outwriter.writerow(print_kn)
            print (print_kn)

            # decision tree
            res_dt = self.estimator('dt', dt_params)
            print_dt = ['decision_tree', res_dt[0], res_dt[1]]
            outwriter.writerow(print_dt)
            print (print_dt)
            
            # random forest
            res_rf = self.estimator('rf', rf_params)
            print_rf = ['random_forest', res_rf[0], res_rf[1]]
            outwriter.writerow(print_rf)
            print (print_rf)


    def estimator(self, est, params):
        X = np.array(self.X)
        y = np.array(self.y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=0)

        scores = ['recall']
        for score in scores:
            
            if est == 'svc_l':
                score = 'f1'
                clf = GridSearchCV(SVC(), params, cv=5)
                                # scoring='%s_macro' % score)
            elif est == 'svc':
                clf = GridSearchCV(SVC(), params, cv=5)
                                # scoring='%s_macro' % score)
            elif est == 'lr':
                score = 'f1'
                clf = GridSearchCV(LogisticRegression(), params, cv=5)
                                # scoring='%s_macro' % score)
            elif est == 'kn':
                clf = GridSearchCV(KNeighborsClassifier(), params, cv=5)
                                # scoring='%s_macro' % score)
            elif est == 'dt':
                clf = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
                                # scoring='%s_macro' % score)
            elif est == 'rf':
                clf = GridSearchCV(RandomForestClassifier(), params, cv=5)
                                # scoring='%s_macro' % score)

            clf.fit(X_train, y_train)

            best_score = clf.best_score_
            test_score = clf.score(X_test, y_test)

        return [str(best_score), str(test_score)]
 

myclass = classification()
# myclass.plot()
myclass.run()