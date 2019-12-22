import csv
import os
import sys
import cudf
from cuml.neighbors import NearestNeighbors
import numpy
from sklearn.neighbors import NearestNeighbors as skNearestNeighbors


qr = int(sys.argv[1])
sr = int(sys.argv[2])
k = int(sys.argv[3])
point = int(sys.argv[4])

np_float = None
with open(os.path.join(os.path.dirname(__file__),'3D_spatial_network.txt'), 'r+') as f:
    tmp_np_float = list(csv.reader(f, delimiter=','))
    np_float = numpy.array(tmp_np_float).astype('float32')

nn_float = NearestNeighbors(algorithm="sweet", q_landmarks=qr, t_landmarks=sr)
nn_float.fit(np_float)
s_distances, s_indices = nn_float.kneighbors(np_float, k)
nb_float = NearestNeighbors(algorithm="brute")
nb_float.fit(np_float)
b_distances, b_indices = nb_float.kneighbors(np_float, k)

diffs = numpy.equal(s_distances, b_distances)
count = 0

diff_points = []
for i, res in enumerate(diffs):
    if False in res:
        diff_points.append(i)
        count += 1

print("{} percent of the points ({}/{}) have conflicting nearest neighbors".format(count/len(diffs)*100, count, len(diffs)))
