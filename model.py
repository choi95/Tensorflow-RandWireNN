import tensorflow as tf
from randwire import RandWire
from tensorflow.keras.models import Model

class Custom_Model(tf.keras.models.Model):
    def __init__(self, node_num, p, filters, kernel_size, graph_mode, model_mode, dataset_mode, is_train):
        super(Custom_Model, self).__init__()
        self.node_num = node_num
        self.p = p
        self.filters = filters
        self.kernel_size = kernel_size
        self.graph_mode = graph_mode
        self.model_mode = model_mode
        self.is_train = is_train
        self.dataset_mode = dataset_mode

        self.num_classes = 1000
        self.dropout_rate = 0.2
       
        if self.dataset_mode is "CIFAR10":
            self.num_classes = 10
        elif self.dataset_mode is "CIFAR100":
            self.num_classes = 100
        elif self.dataset_mode is "IMAGENET":
            self.num_classes = 1000
        elif self.dataset_mode is "MNIST":
            self.num_classes = 10

        if self.model_mode is "CIFAR10":
            self.input_layer = tf.keras.layers.InputLayer(input_shape=(32,32,3), name="CIFAR10_input")       
            self.CIFAR_conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.relu, name="CIFAR10_conv1")        
            self.CIFAR_conv2 = tf.keras.layers.Conv2D(self.filters, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.relu, name="CIFAR10_conv2")                   
            self.CIFAR_conv3 = RandWire(self.node_num, self.p, self.filters, self.kernel_size, self.graph_mode, self.is_train, name="CIFAR10_conv3")
            self.CIFAR_conv4 = RandWire(self.node_num, self.p, self.filters, self.kernel_size, self.graph_mode, self.is_train, name="CIFAR10_conv4")
            self.CIFAR_classifier = tf.keras.layers.Conv2D(self.filters, kernel_size=1, strides=(1, 1), padding='same', name="CIFAR10_classifier")       
            self.BatchNormalization = tf.keras.layers.BatchNormalization(name="CIFAR10_batch")     
    
        elif self.model_mode is "CIFAR100":
            self.input_layer = tf.keras.layers.InputLayer(input_shape=(32,32,3), name="CIFAR100_input")
            self.CIFAR100_conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.relu, name="CIFAR100_conv1")     
            self.CIFAR100_conv2 = RandWire(self.node_num, self.p, self.filters, self.kernel_size * 2, self.graph_mode, self.is_train, name="CIFAR100_conv2")           
            self.CIFAR100_conv3 = RandWire(self.node_num, self.p, self.filters, self.kernel_size * 4, self.graph_mode, self.is_train, name="CIFAR100_conv3")
            self.CIFAR100_conv4 = RandWire(self.node_num, self.p, self.filters, self.kernel_size * 8, self.graph_mode, self.is_train, name="CIFAR100_conv4")            
            self.CIFAR100_classifier = tf.keras.layers.Conv2D(self.filters, kernel_size=1, strides=(1, 1), padding='same', name="CIFAR100_classifier")
    
        elif self.model_mode is "SMALL_REGIME":
            self.input_layer = tf.keras.layers.InputLayer(input_shape=(32,32,3), name="SMALL_input")
            self.SMALL_conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.relu, name="SMALL_conv1")     
            self.SMALL_conv2 = tf.keras.layers.Conv2D(self.filters, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.relu, name="SMALL_conv2")   
            self.SMALL_conv3 = RandWire(self.node_num, self.p, self.filters, self.kernel_size, self.graph_mode, self.is_train, name="SMALL_conv3")
            self.SMALL_conv4 = RandWire(self.node_num, self.p, self.filters, self.kernel_size * 2, self.graph_mode, self.is_train, name="SMALL_conv4")
            self.SMALL_conv5 = RandWire(self.node_num, self.p, self.filters, self.kernel_size * 4, self.graph_mode, self.is_train, name="SMALL_conv5")            
            self.SMALL_classifier = tf.keras.layers.Conv2D(self.filters, kernel_size=1, strides=(1, 1), padding='same', name="SMALL_classifier")
    
        elif self.model_mode is "REGULAR_REGIME":
            self.input_layer = tf.keras.layers.InputLayer(input_shape=(32,32,3), name="REGULAR_input")
            self.REGULAR_conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.relu, name="REGULAR_conv1")   
            self.REGULAR_conv2 = RandWire(self.node_num // 2, self.p, self.filters, self.kernel_size, self.graph_mode, self.is_train, name="REGULAR_conv2")
            self.REGULAR_conv3 = RandWire(self.node_num, self.p, self.filters, self.kernel_size * 2, self.graph_mode, self.is_train, name="REGULAR_conv3")           
            self.REGULAR_conv4 = RandWire(self.node_num, self.p, self.filters, self.kernel_size * 4, self.graph_mode, self.is_train, name="REGULAR_conv4")         
            self.REGULAR_conv5 = RandWire(self.node_num, self.p, self.filters, self.kernel_size * 8, self.graph_mode, self.is_train, name="REGULAR_conv5")
            self.REGULAR_classifier = tf.keras.layers.Conv2D(self.filters, kernel_size=1, strides=(1, 1), padding='same', name="REGULAR_classifier")
    
        self.Dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.Dense = tf.keras.layers.Dense(10)
            

    def call(self, x):
        if self.model_mode is "CIFAR10":
            out = self.CIFAR_conv1(x)     
            out = self.BatchNormalization(out)         
            out = self.CIFAR_conv2(out) 
            out = self.BatchNormalization(out)              
            out = self.CIFAR_conv3(out)         
            out = self.CIFAR_conv4(out)         
            out = self.CIFAR_classifier(out)
            out = self.BatchNormalization(out)
            
        elif self.model_mode is "CIFAR100":
            out = self.CIFAR100_conv1(x)
            out = self.CIFAR100_conv2(out)
            out = self.CIFAR100_conv3(out)
            out = self.CIFAR100_conv4(out)
            out = self.CIFAR100_classifier(out)
        elif self.model_mode is "SMALL_REGIME":
            out = self.SMALL_conv1(x)
            out = self.SMALL_conv2(out)
            out = self.SMALL_conv3(out)
            out = self.SMALL_conv4(out)
            out = self.SMALL_conv5(out)
            out = self.SMALL_classifier(out)
        elif self.model_mode is "REGULAR_REGIME":
            out = self.REGULAR_conv1(x)
            out = self.REGULAR_conv2(out)
            out = self.REGULAR_conv3(out)
            out = self.REGULAR_conv4(out)
            out = self.REGULAR_conv5(out)
            out = self.REGULAR_classifier(out)

        # global average pooling
        batch_size, height, width, channels = tuple(out.shape)
        out = tf.nn.avg_pool2d(out, ksize=[height, width], strides=[height,width], padding='VALID')
        out = tf.squeeze(out, axis=[1,2])
        out = self.Dropout(out)
        out = self.Dense(out)
        return out

    