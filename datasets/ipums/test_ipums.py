import csv
import os
import sys
import cudf
from cuml.neighbors import NearestNeighbors
import numpy
import cuml


ql = int(sys.argv[1])
tl = int(sys.argv[2])
k = int(sys.argv[3])
point = int(sys.argv[4])


np_float = None
with open(os.path.join(os.path.dirname(__file__),'s.ipums.la.99'), 'r+') as f:
    tmp_np_float = list(csv.reader(f, delimiter=','))
    np_float = numpy.array(tmp_np_float)

gdf_float = cudf.DataFrame()
gdf_float['dim_0'] = numpy.ascontiguousarray(np_float[:,0]).astype('float32')
gdf_float['dim_1'] = numpy.ascontiguousarray(np_float[:,1]).astype('float32')
gdf_float['dim_2'] = numpy.ascontiguousarray(np_float[:,2]).astype('float32')
gdf_float['dim_3'] = numpy.ascontiguousarray(np_float[:,3]).astype('float32')
gdf_float['dim_4'] = numpy.ascontiguousarray(np_float[:,4]).astype('float32')
gdf_float['dim_5'] = numpy.ascontiguousarray(np_float[:,5]).astype('float32')
gdf_float['dim_6'] = numpy.ascontiguousarray(np_float[:,6]).astype('float32')
gdf_float['dim_7'] = numpy.ascontiguousarray(np_float[:,7]).astype('float32')
gdf_float['dim_8'] = numpy.ascontiguousarray(np_float[:,8]).astype('float32')
gdf_float['dim_9'] = numpy.ascontiguousarray(np_float[:,9]).astype('float32')
gdf_float['dim_10'] = numpy.ascontiguousarray(np_float[:,10]).astype('float32')
gdf_float['dim_11'] = numpy.ascontiguousarray(np_float[:,11]).astype('float32')
gdf_float['dim_12'] = numpy.ascontiguousarray(np_float[:,12]).astype('float32')
gdf_float['dim_13'] = numpy.ascontiguousarray(np_float[:,13]).astype('float32')
gdf_float['dim_14'] = numpy.ascontiguousarray(np_float[:,14]).astype('float32')
gdf_float['dim_15'] = numpy.ascontiguousarray(np_float[:,15]).astype('float32')
gdf_float['dim_16'] = numpy.ascontiguousarray(np_float[:,16]).astype('float32')
gdf_float['dim_17'] = numpy.ascontiguousarray(np_float[:,17]).astype('float32')
gdf_float['dim_18'] = numpy.ascontiguousarray(np_float[:,18]).astype('float32')
gdf_float['dim_19'] = numpy.ascontiguousarray(np_float[:,19]).astype('float32')
gdf_float['dim_20'] = numpy.ascontiguousarray(np_float[:,20]).astype('float32')
gdf_float['dim_21'] = numpy.ascontiguousarray(np_float[:,21]).astype('float32')
gdf_float['dim_22'] = numpy.ascontiguousarray(np_float[:,22]).astype('float32')
gdf_float['dim_23'] = numpy.ascontiguousarray(np_float[:,23]).astype('float32')
gdf_float['dim_24'] = numpy.ascontiguousarray(np_float[:,14]).astype('float32')
gdf_float['dim_25'] = numpy.ascontiguousarray(np_float[:,15]).astype('float32')
gdf_float['dim_26'] = numpy.ascontiguousarray(np_float[:,16]).astype('float32')
gdf_float['dim_27'] = numpy.ascontiguousarray(np_float[:,17]).astype('float32')
gdf_float['dim_28'] = numpy.ascontiguousarray(np_float[:,18]).astype('float32')
gdf_float['dim_29'] = numpy.ascontiguousarray(np_float[:,19]).astype('float32')
gdf_float['dim_30'] = numpy.ascontiguousarray(np_float[:,10]).astype('float32')
gdf_float['dim_31'] = numpy.ascontiguousarray(np_float[:,11]).astype('float32')
gdf_float['dim_32'] = numpy.ascontiguousarray(np_float[:,12]).astype('float32')
gdf_float['dim_33'] = numpy.ascontiguousarray(np_float[:,13]).astype('float32')
gdf_float['dim_34'] = numpy.ascontiguousarray(np_float[:,14]).astype('float32')
gdf_float['dim_35'] = numpy.ascontiguousarray(np_float[:,15]).astype('float32')
gdf_float['dim_36'] = numpy.ascontiguousarray(np_float[:,16]).astype('float32')
gdf_float['dim_37'] = numpy.ascontiguousarray(np_float[:,17]).astype('float32')
gdf_float['dim_38'] = numpy.ascontiguousarray(np_float[:,18]).astype('float32')
gdf_float['dim_39'] = numpy.ascontiguousarray(np_float[:,19]).astype('float32')
gdf_float['dim_40'] = numpy.ascontiguousarray(np_float[:,10]).astype('float32')
gdf_float['dim_41'] = numpy.ascontiguousarray(np_float[:,11]).astype('float32')
gdf_float['dim_42'] = numpy.ascontiguousarray(np_float[:,12]).astype('float32')
gdf_float['dim_43'] = numpy.ascontiguousarray(np_float[:,13]).astype('float32')
gdf_float['dim_44'] = numpy.ascontiguousarray(np_float[:,14]).astype('float32')
gdf_float['dim_45'] = numpy.ascontiguousarray(np_float[:,15]).astype('float32')
gdf_float['dim_46'] = numpy.ascontiguousarray(np_float[:,16]).astype('float32')
gdf_float['dim_47'] = numpy.ascontiguousarray(np_float[:,17]).astype('float32')
gdf_float['dim_48'] = numpy.ascontiguousarray(np_float[:,18]).astype('float32')
gdf_float['dim_49'] = numpy.ascontiguousarray(np_float[:,19]).astype('float32')
gdf_float['dim_50'] = numpy.ascontiguousarray(np_float[:,10]).astype('float32')
gdf_float['dim_51'] = numpy.ascontiguousarray(np_float[:,11]).astype('float32')
gdf_float['dim_52'] = numpy.ascontiguousarray(np_float[:,12]).astype('float32')
gdf_float['dim_53'] = numpy.ascontiguousarray(np_float[:,13]).astype('float32')
gdf_float['dim_54'] = numpy.ascontiguousarray(np_float[:,14]).astype('float32')
gdf_float['dim_55'] = numpy.ascontiguousarray(np_float[:,15]).astype('float32')
gdf_float['dim_56'] = numpy.ascontiguousarray(np_float[:,16]).astype('float32')
gdf_float['dim_57'] = numpy.ascontiguousarray(np_float[:,17]).astype('float32')
gdf_float['dim_58'] = numpy.ascontiguousarray(np_float[:,18]).astype('float32')
gdf_float['dim_59'] = numpy.ascontiguousarray(np_float[:,19]).astype('float32')
gdf_float['dim_60'] = numpy.ascontiguousarray(np_float[:,19]).astype('float32')

# le = cuml.preprocessing.LabelEncoder()
# gdf_float['dim_0'] = le.fit_transform(gdf_float['dim_0']).astype('float32')

nn_float = NearestNeighbors(algorithm="sweet", q_landmarks=ql, t_landmarks=tl)
nn_float.fit(gdf_float)
s_distances, s_indices = nn_float.kneighbors(gdf_float, k)
nb_float = NearestNeighbors(algorithm="brute")
nb_float.fit(gdf_float)
b_distances, b_indices = nb_float.kneighbors(gdf_float, k)

#TODO: compare indices

diffs = numpy.equal(s_distances, b_distances, )
count = 0

diff_points = []
for i, res in enumerate(diffs):
    if False in res:
        diff_points.append(i)
        count += 1

print("{} percent of the points ({}/{}) have conflicting nearest neighbors".format(count/len(diffs)*100, count, len(diffs)))


for i in range(0, k):
    print("{}: \t{}\t{:0.6f}\t{}\t{:0.6f}".format(i, s_indices[point][i], s_distances[point][i], b_indices[point][i], b_distances[point][i]))
