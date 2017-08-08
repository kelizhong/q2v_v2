import h5py
import nmslib
import argparse
from multiprocessing import cpu_count


def parse_args():
    parser = argparse.ArgumentParser(description='Create nms lib index')

    # vocabulary parameter
    parser.add_argument('-hf', '--h5-file', type=str, default='q2v.h5', help='h5 file for query vector')
    parser.add_argument('-d', '--dataset', type=str, default='q2v', help='dataset name')
    parser.add_argument('-i', '--index-name', type=str, default='q2v.index', help='nms index name')

    return parser.parse_args()


def create_nms_index(h5_file, dataset, index_name):
    with h5py.File(h5_file, 'r') as h5f:
        data = h5f[dataset][:]
    index = nmslib.init(method='hnsw', space='cosinesimil', data_type=nmslib.DataType.DENSE_VECTOR,
                        dtype=nmslib.DistType.FLOAT)
    index.addDataPointBatch(data)
    index.createIndex({'post': 2, 'indexThreadQty': cpu_count}, print_progress=True)
    'created and saved index'
    index.saveIndex(index_name)
    nmslib.freeIndex(index)


if __name__ == "__main__":
    args = parse_args()
    create_nms_index(args.h5_file, args.dataset, args.index_name)
