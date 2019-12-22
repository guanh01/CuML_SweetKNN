import csv
import os
import sys
import time
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
    np_float = numpy.array(tmp_np_float).astype('float32')

t_start = time.time()
gdf_float = cudf.DataFrame()
gdf_float['dim_0'] = numpy.ascontiguousarray(np_float[:,0])
gdf_float['dim_1'] = numpy.ascontiguousarray(np_float[:,1])
gdf_float['dim_2'] = numpy.ascontiguousarray(np_float[:,2])
gdf_float['dim_3'] = numpy.ascontiguousarray(np_float[:,3])

cudf_float = NearestNeighbors(algorithm="sweet", q_landmarks=ql, t_landmarks=tl)
cudf_float.fit(gdf_float)
s_distances, s_indices = cudf_float.kneighbors(gdf_float, k)
print("cuDF Sweet KNN total time: {}".format(time.time() - t_start))

t_start = time.time()
numpy_float = NearestNeighbors(algorithm="sweet", q_landmarks=ql, t_landmarks=tl)
numpy_float.fit(np_float)
b_distances, b_indices = numpy_float.kneighbors(np_float, k)
print("ndarray Sweet KNN total time:{}".format(time.time() - t_start))