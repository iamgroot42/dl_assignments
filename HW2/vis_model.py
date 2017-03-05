# Tensorflow bug fix while importing keras
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import load_model

import pydot
import sys

def plot(model, to_file):

    graph = pydot.Dot(graph_type='digraph')

    previous_node = None
    written_nodes = []
    n = 1
    for node in model.get_config()['layers']:
        if (node['name'] + str(n)) in written_nodes:
            n += 1
        current_node = pydot.Node(node['name'] + str(n))
        written_nodes.append(node['name'] + str(n))
        graph.add_node(current_node)
        if previous_node:
            graph.add_edge(pydot.Edge(previous_node, current_node))
        previous_node = current_node
    graph.write_png(to_file)


if __name__ == "__main__":
	model = load_model(sys.argv[1])
	plot(model, sys.argv[2])
