import tensorflow as tf
from graph import RandomGraph


# reference, Thank you.
# https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py
# Reporting 1,
# I don't know which one is better, between 'bias=False' and 'bias=True'
class SeparableConv2d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, stride=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.conv = tf.keras.layers.Conv2D(in_channels, 3, stride, padding='same', groups=in_channels, use_bias=bias)
        self.pointwise = tf.keras.layers.Conv2D(out_channels, 1, stride, padding='valid', use_bias=bias)

    def call(self, x):
        out = self.conv(x)
        out = self.pointwise(out)
        return out


# ReLU-convolution-BN triplet
class Unit(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Unit, self).__init__()
        self.dropout_rate = 0.2
        self.SeparableConv2d = SeparableConv2d(in_channels, out_channels, stride=stride)
        self.BatchNormalization = tf.keras.layers.BatchNormalization()
        self.ReLU = tf.keras.layers.ReLU()
        self.Dropout_rate = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, x):
        out = self.SeparableConv2d(x)
        out = self.BatchNormalization(out)
        out = self.ReLU(out)
        out = self.Dropout_rate(out)
        return out


# Reporting 2,
# In the paper, they said "The aggregation is done by weighted sum with learnable positive weights".
class Node(tf.keras.layers.Layer):
    def __init__(self, in_degree, in_channels, out_channels, stride=1):
        super(Node, self).__init__()
        self.in_degree = in_degree
        if len(self.in_degree) > 1:
            # self.weights = nn.Parameter(torch.zeros(len(self.in_degree), requires_grad=True))
            self.w = tf.Variable(tf.ones(len(self.in_degree)))
        self.unit = Unit(in_channels, out_channels, stride=stride)

    def call(self, *input):
        if len(self.in_degree) > 1:
            x = (input[0] * tf.sigmoid(self.w[0]))
            for index in range(1, len(input)):
                x += (input[index] * tf.sigmoid(self.w[index]))
            out = self.unit(x)
        else:
            out = self.unit(input[0])
        return out
        
class RandWire(tf.keras.layers.Layer):

    def __init__(self, node_num, p, in_channels, out_channels, graph_mode, is_train, name):
        super(RandWire, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.is_train = is_train
        self.gname = name
        #get graph nodes and in edges
        graph_node = RandomGraph(self.node_num, self.p, graph_mode=graph_mode)
        if self.is_train is True:
            print("is_train: True")
            graph = graph_node.make_graph()
            self.nodes, self.in_edges = graph_node.get_graph_info(graph)
            graph_node.save_random_graph(graph, self.gname)
        else:
            graph = graph_node.load_random_graph(self.gname)
            self.nodes, self.in_edges = graph_node.get_graph_info(graph)
        
        #define input Node
        self.module_list = [Node(self.in_edges[0], self.in_channels, self.out_channels, stride=2)]
        # define the rest Node
        self.module_list.extend([Node(self.in_edges[node], self.out_channels, self.out_channels) for node in self.nodes if node > 0])

    def call(self, x):
        memory = {}
        # start vertex
        out = self.module_list[0](x)
        memory[0] = out

        # the rest vertex
        for node in range(1, len(self.nodes) - 1):
            # print(node, self.in_edges[node][0], self.in_edges[node])
            if len(self.in_edges[node]) > 1:
                
                out = self.module_list[node](*[memory[in_vertex] for in_vertex in self.in_edges[node]])
            else:          
                out = self.module_list[node](memory[self.in_edges[node][0]])
            memory[node] = out
 
        for node in range(1, len(self.nodes) - 1):
            # print(node, self.in_edges[node][0], self.in_edges[node])
            if len(self.in_edges[node]) > 1:
                out = self.module_list[node](*[memory[in_vertex] for in_vertex in self.in_edges[node]])
            else:
                out = self.module_list[node](memory[self.in_edges[node][0]])
                memory[node] = out

        # Reporting 3,
        # How do I handle the last part?
        # It has two kinds of methods.
        # first, Think of the last module as a Node and collect the data by proceeding in the same way as the previous operation.
        # second, simply sum the data and export the output.

        # My Opinion
        # out = self.module_list[self.node_num + 1].forward(*[memory[in_vertex] for in_vertex in self.in_edges[self.node_num + 1]])

        # In paper
        # print("self.in_edges: ", self.in_edges[self.node_num + 1], self.in_edges[self.node_num + 1][0])
        out = memory[self.in_edges[self.node_num + 1][0]]
        for in_vertex_index in range(1, len(self.in_edges[self.node_num + 1])):
            out += memory[self.in_edges[self.node_num + 1][in_vertex_index]]
        out = out / len(self.in_edges[self.node_num + 1])
        return out
