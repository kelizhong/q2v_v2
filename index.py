import json
import os
import numpy
import tensorflow as tf
from flask import Flask, jsonify
from flask_restful import Resource, Api

from google.protobuf.json_format import MessageToJson

from utils.config_decouple import config
from external.tf_serving.protocol import predict_pb2
from helper.tensorflow_serving_client_helper import TFServingClientHelper
from helper.thrift_client_helper import ThriftClientHelper
from helper.tokenizer_helper import TextBlobTokenizerHelper
from helper.tokens_helper import TokensHelper
from helper.vocabulary_helper import VocabularyHelper

FLAGS = tf.app.flags.FLAGS

app = Flask(__name__)
api = Api(app)

vocabulary = VocabularyHelper().load_vocabulary()
tokenizer = TextBlobTokenizerHelper()
tokens_helper = TokensHelper(tokenize_fn=tokenizer.tokenize, vocabulary=vocabulary,
                             unk_token=config('_unk_', section='vocabulary_symbol'))

tf_serving_host = config('host', section='tf_serving')
tf_serving_port = config('port', section='tf_serving')
model_name = config('model_name', section='tf_serving')
model_version = config('model_version', section='tf_serving')
request_timeout = config('timeout', section='tf_serving')

client = ThriftClientHelper(host=config('host', section='nmslib_thrift'), port=config('port', section='nmslib_thrift'))

tf_serving_client = TFServingClientHelper(tf_serving_host, tf_serving_port)

meta = dict((line.strip().split('\t') for line in open(os.path.join(config('rawdata_dir'), 'q2v.tsv'))))


@app.before_first_request
def setup_database(*args, **kwargs):
    print("sdfdsfdsfds")


def make_inference_request(inputs):
    lengths = [len(s) for s in inputs]
    # Generate inference data
    inputs = numpy.asarray(inputs)

    inputs_tensor_proto = tf.contrib.util.make_tensor_proto(inputs,
                                                            dtype=tf.int32)

    inputs_len = numpy.asarray(lengths)

    inputs_len_tensor_proto = tf.contrib.util.make_tensor_proto(inputs_len,
                                                                dtype=tf.int32)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    if model_version > 0:
        request.model_spec.version.value = model_version
    request.inputs['inputs'].CopyFrom(inputs_tensor_proto)
    request.inputs['inputs_length'].CopyFrom(inputs_len_tensor_proto)

    return request


class HelloWorld(Resource):
    def get(self, query):
        input = tokens_helper.tokens(query, return_data=False)
        print(input)

        request = make_inference_request([input])
        # Send request
        result = tf_serving_client.predict(request, request_timeout)
        result = json.loads(MessageToJson(result))
        query = result['outputs']['src_last_output']['floatVal']
        print(" ".join(map(str, query)))
        res = client.find_k_nearest(100, " ".join(map(str, query)))
        res = [(meta.get(str(ele[0]), 'unknown word for %d' % ele[0]), ele[1]) for ele in res]

        return jsonify(res)


api.add_resource(HelloWorld, '/<query>')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
