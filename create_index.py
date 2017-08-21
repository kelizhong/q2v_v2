import argparse
import h5py
import numpy as np
import nmslib


def parse_args():
    parser = argparse.ArgumentParser(description='Create nms lib index')

    # vocabulary parameter
    parser.add_argument('h5_file', type=str, default='q2v.h5', help='h5 file for query vector')
    parser.add_argument('-d', '--dataset', type=str, default='q2v', help='dataset name')
    parser.add_argument('-v', '--vector-text-path', type=str, default='q2v.txt', help='query vector text')
    parser.add_argument('-i', '--index-path', type=str, default='q2v.index', help='nms index name')

    return parser.parse_args()


def create_nms_index(h5_file, dataset, vector_text_path, index_path):
    with h5py.File(h5_file, 'r') as h5f:
        data = h5f[dataset][:]

    with open(vector_text_path, 'wb') as f:
        np.savetxt(f, data, delimiter="\t")

    index = nmslib.init(method='hnsw', space='cosinesimil', data_type=nmslib.DataType.DENSE_VECTOR,
                        dtype=nmslib.DistType.FLOAT)
    index.addDataPointBatch(data)
    index.createIndex({'post': 2}, print_progress=True)
    # created and saved index
    index.saveIndex(index_path)
    nmslib.freeIndex(index)


if __name__ == "__main__":
    args = parse_args()
    create_nms_index(args.h5_file, args.dataset, args.vector_text_path, args.index_path)
