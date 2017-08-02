import numpy as np
import h5py
import nmslib
#a = np.random.random(size=(10000,100))
#h5f = h5py.File('data.h5', 'w')
#h5f.create_dataset('dataset_1', data=a)
#h5f.close()



with h5py.File('q2v.h5','r') as h5f:
    data = h5f['q2v'][:]
index = nmslib.init(method='hnsw', space='cosinesimil', data_type=nmslib.DataType.DENSE_VECTOR, dtype=nmslib.DistType.FLOAT)
index.loadIndex('q2v.index')

ids, distances = index.knnQuery(data[0], k=10)
print(ids)