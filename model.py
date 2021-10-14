import tensorflow as tf
from randwire import RandWire
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

class Model(tf.keras.models.Model):
    def __init__(self, node_num, p, filters, kernel_size, graph_mode, model_mode, dataset_mode, is_train, batch_size):
        super(Model, self).__init__()
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
           
            self.CIFAR_conv1 = Sequential([
                tf.keras.layers.InputLayer(input_shape=(32,32,3), batch_size=batch_size),
                tf.keras.layers.Conv2D(self.filters, kernel_size=3, activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization()
            ])
            self.CIFAR_conv2 = Sequential([
                tf.keras.layers.Conv2D(self.filters, kernel_size=3, activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization()
            ])
            #self.CIFAR_conv3 = tf.keras.Sequential([             
            #    RandWire(self.node_num, self.p, self.filters, self.kernel_size, self.graph_mode, self.is_train, name="CIFAR10_conv2")
            #])
            self.CIFAR_conv3 = Sequential([
                RandWire(self.node_num, self.p, self.filters, self.kernel_size, self.graph_mode, self.is_train, name="CIFAR10_conv3")
            ])
            self.CIFAR_conv4 = Sequential([
                RandWire(self.node_num, self.p, self.filters, self.kernel_size, self.graph_mode, self.is_train, name="CIFAR10_conv4")
            ])    
            self.CIFAR_classifier = Sequential([
                tf.keras.layers.Conv2D(self.filters, kernel_size=1, activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization()
            ])
        elif self.model_mode is "CIFAR100":
            self.CIFAR100_conv1 = tf.keras.Sequential([
                tf.keras.Input(shape=(32,32,3)),
                tf.keras.layers.Conv2D(self.filters, kernel_size=3, activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization()
            ])
            self.CIFAR100_conv2 = tf.keras.Sequential([
                RandWire(self.node_num, self.p, self.filters, self.kernel_size * 2, self.graph_mode, self.is_train, name="CIFAR100_conv2")
            ])
            self.CIFAR100_conv3 = tf.keras.Sequential([
                RandWire(self.node_num, self.p, self.filters, self.kernel_size * 4, self.graph_mode, self.is_train, name="CIFAR100_conv3")
            ])
            self.CIFAR100_conv4 = tf.keras.Sequential([
                RandWire(self.node_num, self.p, self.filters, self.kernel_size * 8, self.graph_mode, self.is_train, name="CIFAR100_conv4")
            ])
            self.CIFAR100_classifier = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.filters, kernel_size=1, activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization()
            ])
        elif self.model_mode is "SMALL_REGIME":
            self.SMALL_conv1 = tf.keras.Sequential([
                tf.keras.Input(shape=(32,32,3)),
                tf.keras.layers.Conv2D(self.filters, kernel_size=3, activation='relu', padding='same'), #check
                tf.keras.layers.BatchNormalization()
            ])
            self.SMALL_conv2 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.filters, kernel_size=3, activation='relu',  padding='same'),
                tf.keras.layers.BatchNormalization()
            ])
            self.SMALL_conv3 = tf.keras.Sequential([
                RandWire(self.node_num, self.p, self.filters, self.kernel_size, self.graph_mode, self.is_train, name="SMALL_conv3")
            ])
            self.SMALL_conv4 = tf.keras.Sequential([
                RandWire(self.node_num, self.p, self.filters, self.kernel_size * 2, self.graph_mode, self.is_train, name="SMALL_conv4")
            ])
            self.SMALL_conv5 = tf.keras.Sequential([
                RandWire(self.node_num, self.p, self.filters, self.kernel_size * 4, self.graph_mode, self.is_train, name="SMALL_conv5")
            ])
            self.SMALL_classifier = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.filters, kernel_size=3, activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization()
            ])
        elif self.model_mode is "REGULAR_REGIME":
            self.REGULAR_conv1 = tf.keras.Sequential([
                tf.keras.Input(shape=(32,32,3)),
                tf.keras.layers.Conv2D(self.filters, kernel_size=3, activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization()
            ])
            self.REGULAR_conv2 = tf.keras.Sequential([
                RandWire(self.node_num // 2, self.p, self.filters, self.kernel_size, self.graph_mode, self.is_train, name="REGULAR_conv2")
            ])
            self.REGULAR_conv3 = tf.keras.Sequential([
                RandWire(self.node_num, self.p, self.filters, self.kernel_size * 2, self.graph_mode, self.is_train, name="REGULAR_conv3")
            ])
            self.REGULAR_conv4 = tf.keras.Sequential([
                RandWire(self.node_num, self.p, self.filters, self.kernel_size * 4, self.graph_mode, self.is_train, name="REGULAR_conv4")
            ])
            self.REGULAR_conv5 = tf.keras.Sequential([
                RandWire(self.node_num, self.p, self.filters, self.kernel_size * 8, self.graph_mode, self.is_train, name="REGULAR_conv5")
            ])
            self.REGULAR_classifier = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.filters,  kernel_size=3, activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization()
            ])

        self.output_layer = tf.keras.Sequential([
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(1)
        ])

    def call(self, x):
        if self.model_mode is "CIFAR10":
            out = self.CIFAR_conv1(x)     
            print('***', out.shape)
            out = self.CIFAR_conv2(out)    
            print('***', out.shape)     
            out = self.CIFAR_conv3(out)   
            print('***', out.shape)  
            out = self.CIFAR_conv4(out)   
            print('***', out.shape)
            out = self.CIFAR_classifier(out)
            print('***', out.shape)
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
        print('11111')
        out = self.output_layer(out)
        print('22222')
        return out

    