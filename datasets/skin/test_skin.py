import csv
import os
import sys
import cudf
from cuml.neighbors import NearestNeighbors
import numpy
from sklearn.neighbors import NearestNeighbors as skNearestNeighbors


ql = int(sys.argv[1])
tl = int(sys.argv[2])
k = int(sys.argv[3])
point = int(sys.argv[4])


np_float = None
with open(os.path.join(os.path.dirname(__file__),'Skin_NonSkin.txt'), 'r+') as f:
    tmp_np_float = list(csv.reader(f, delimiter='\t'))
    gdf_float = numpy.array(tmp_np_float).astype('float32')

nn_float = NearestNeighbors(algorithm="sweet", q_landmarks=ql, t_landmarks=tl)
nn_float.fit(gdf_float)
s_distances, s_indices = nn_float.kneighbors(gdf_float, k)
nb_float = NearestNeighbors(algorithm="brute")
nb_float.fit(gdf_float)
b_distances, b_indices = nb_float.kneighbors(gdf_float, k)

diffs = numpy.equal(s_distances, b_distances)
count = 0

diff_points = []
for i, res in enumerate(diffs):
    if False in res:
        diff_points.append(i)
        count += 1

print("{} percent of the points ({}/{}) have conflicting nearest neighbors".format(count/len(diffs)*100, count, len(diffs)))

# # for point in diff_points:
# print("KNN results for point {} (neighbor, sweet_index, sweet_distance, brute_index, brute_distance)".format(point))
# sd = list(s_distances.iloc[point])
# si = list(s_indices.iloc[point])
# bd = list(b_distances.iloc[point])
# bi = list(b_indices.iloc[point])

#for i in range(0, k):
#    print("{}: \t{}\t{:0.6f}\t{}\t{:0.6f}".format(i, s_indices[point][i], s_distances[point][i], b_indices[point][i], b_distances[point][i]))
