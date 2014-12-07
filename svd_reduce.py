import json
import numpy
import math
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix

num_singular_vals = [439, 817, 1214, 1590, 1989, 2320]

for i in range(6):
    size = 500 * (i + 1)
    x_file_name = './data/task2_tfidf2d_list' + str(size) + '.json'
    x_file_data = open(x_file_name)
    x = json.load(x_file_data)
    x_file_data.close()

    X1 = csc_matrix(x)
    U,S,V = svds(X1, num_singular_vals[i])
    X_mod = []
    X_mod = numpy.dot(U,numpy.diag(S))

    file_name = './data/reduced_tfidf' + str(size) + '.json'
    with open(file_name, 'w') as outfile:
        json.dump(X_mod.tolist(), outfile)
    outfile.close()