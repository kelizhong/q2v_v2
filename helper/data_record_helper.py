import glob
import tensorflow as tf


class DataRecordHelper(object):
    @staticmethod
    def _int64_feature(value):
        """Wrapper for inserting an int64 Feature into a Example proto."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _float32_feature(value):
        """Wrapper for inserting an float32 Feature into a Example proto."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def create_sequence_example(source, targets, label):
        assert len(targets) == len(
            label), "targets length must equal to label length, targets length:%d, label length:%d" % (
            len(targets), len(label))
        feature = dict()
        feature['source'] = DataRecordHelper._int64_feature(source)
        feature['source_length'] = DataRecordHelper._int64_feature([len(source)])
        for i, target in enumerate(targets):
            feature['target_%d' % i] = DataRecordHelper._int64_feature(target)
            feature['target_%d_length' % i] = DataRecordHelper._int64_feature([len(target)])
        feature['label'] = DataRecordHelper._float32_feature(label)

        features = tf.train.Features(feature=feature)
        ex = tf.train.Example(features=features)
        return ex

    def get_padded_batch(self, file_list, batch_size, label_size, queue_capacity=2048, num_enqueuing_threads=4):
        file_queue = tf.train.string_input_producer(file_list)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)

        features = dict()
        features['source'] = tf.VarLenFeature(dtype=tf.int64)
        features['source_length'] = tf.FixedLenFeature(shape=[], dtype=tf.int64)
        for i in range(label_size):
            features['target_%d' % i] = tf.VarLenFeature(dtype=tf.int64)
            features['target_%d_length' % i] = tf.FixedLenFeature(shape=[], dtype=tf.int64)
        features['label'] = tf.VarLenFeature(dtype=tf.float32)

        parsed = tf.parse_single_example(serialized_example, features=features)

        queue = tf.PaddingFIFOQueue(
            capacity=queue_capacity,
            dtypes=[tf.int64] * ((label_size + 1) * 2) + [tf.float32],
            shapes=[(None,), ()] * (label_size + 1) + [(None,)])

        vals = list()
        vals.append(tf.sparse_tensor_to_dense(parsed['source']))
        vals.append(parsed['source_length'])
        for i in range(label_size):
            vals.append(tf.sparse_tensor_to_dense(parsed['target_%d' % i]))
            vals.append(parsed['target_%d_length' % i])
        vals.append(tf.sparse_tensor_to_dense(parsed['label']))
        enqueue_ops = [queue.enqueue(vals)] * num_enqueuing_threads
        tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
        data = queue.dequeue_many(batch_size)
        data = self.process_batch_data(data, label_size)
        return data

    @staticmethod
    def process_batch_data(data, label_size):
        source_batch_data = data[0]
        source_batch_length = data[1]
        # data[i] -> [batch_size, padding_dim]
        targets_batch_data = tf.concat([data[i] for i in range(2, 2 * label_size + 1, 2)], axis=1)
        # targets_batch_data = [batch_size, padding_dim * label_size]
        # data[i] -> [1, batch_size]
        targets_batch_length = [data[i] for i in range(3, 2 * label_size + 2, 2)]
        # targets_batch_length ->[label_size, batch_size]
        targets_batch_length = tf.transpose(targets_batch_length)
        # targets_batch_length ->[batch_size, label_size]
        label_batch = data[-1]

        return source_batch_data, source_batch_length, targets_batch_data, targets_batch_length, label_batch

    def create_sequence(self, data_iterator, record_path='train.tfrecords'):
        writer = tf.python_io.TFRecordWriter(record_path)
        for i, (source, targets, label) in enumerate(data_iterator):
            # sequence_example = create_sequence_example([12, 3, 33], [[1, 3, i], [2, 32, 2]], [2, 2])
            sequence_example = DataRecordHelper.create_sequence_example(source, targets, label)

            writer.write(sequence_example.SerializeToString())
        writer.close()


if __name__ == '__main__':
    file_list = glob.glob('./train.tfrecords')
    d = DataRecordHelper()
    d.create_sequence([([12, 3, 33], [[1, 3], [2, 32, 2, 100]], [2, 2]), ([12, 3], [[1, 3, 200], [2, 32, 2]], [2, 2])])
    # d.create_sequence([([12, 3, 33], [[1, 3], [2, 32, 2,100]], [2, 3]), ([12, 3], [[1, 3,200], [2, 32, 2]], [2, 2])])
    # 初始化所有的op
    init = tf.global_variables_initializer()

    source_batch_data, source_batch_length, targets_batch_data, targets_batch_length, label_batch = d.get_padded_batch(
        ['./train.tfrecords'], batch_size=1, label_size=2)

    # sources = tf.squeeze(data[0])
    # labels = tf.squeeze(data[-1])
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        # 启动队列
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                data = sess.run(
                    [source_batch_data, source_batch_length, targets_batch_data, targets_batch_length, label_batch])
                print(data[2])

        except tf.errors.OutOfRangeError:
            print('Finished extracting.')
        finally:
            coord.request_stop()
            coord.join(threads)
