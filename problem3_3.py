import sys
import math
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC



class classification:
    def __init__(self):
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
        X = np.array(self.X)
        y = np.array(self.y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=0)

        # Set the parameters by cross-validation
        # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
        #                     'C': [1, 10, 100, 1000]},
        #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        tuned_parameters = [ { 
            'kernel': ['rbf'], 
            'C': [0.1, 0.5, 1, 5, 10, 50, 100], 
            # 'degree':  [4, 5, 6], 
            'gamma': [0.1, 0.5, 1, 3, 6, 10]
             }]

        scores = ['recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                            scoring='%s_macro' % score)
            clf.fit(X_train, y_train)

            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)

            print ('train_score', train_score)
            print ('test_score', test_score)


            l_params = [ { 
                'kernel': ['linear'], 
                'C': [0.1, 0.5, 1, 5, 10, 50, 100] 
                # 'degree':  [4, 5, 6], 
                # 'gamma': [0.1, 0.5, 1, 3, 6, 10]
                }]
            clf_l = GridSearchCV(SVC(), l_params, cv=5,
                            scoring='%s_macro' % score)
            clf_l.fit(X_train, y_train)

            train_score = clf_l.score(X_train, y_train)
            test_score = clf_l.score(X_test, y_test)

            print ('train_score', train_score)
            print ('test_score', test_score)

            # print("Best parameters set found on development set:")
            # print()
            # print(clf.best_params_)
            # print()
            # print("Grid scores on development set:")
            # print()
            # means = clf.cv_results_['mean_test_score']
            # stds = clf.cv_results_['std_test_score']
            # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            #     print("%0.3f (+/-%0.03f) for %r"
            #         % (mean, std * 2, params))
            # print()

            # print("Detailed classification report:")
            # print()
            # print("The model is trained on the full development set.")
            # print("The scores are computed on the full evaluation set.")
            # print()
            # y_true, y_pred = y_test, clf.predict(X_test)
            # print(classification_report(y_true, y_pred))
            # print()


myclass = classification()
# myclass.plot()
myclass.run()