import tensorflow as tf
from randwire import RandWire


class Custom_Model(tf.keras.models.Model):
    def __init__(self, node_num, p, filters, kernel_size, graph_mode, model_mode, dataset_mode, is_train):
        super(Custom_Model, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = filters
        self.out_channels = filters
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
            self.CIFAR_conv1 = tf.keras.layers.Conv2D(
                    self.out_channels, self.kernel_size, strides=(1, 1), padding='same', activation=tf.nn.relu, name="CIFAR10_conv1")
            self.BatchNorm2D_conv1 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)
            self.CIFAR_conv2 = tf.keras.layers.Conv2D(
                    self.out_channels, self.kernel_size, strides=(1, 1), padding='same', name="CIFAR10_conv2")
            self.BatchNorm2D_conv2 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)
            self.CIFAR_conv3 = RandWire(
                    self.node_num, self.p, self.in_channels, self.out_channels *2, self.graph_mode, self.is_train, name="CIFAR10_conv3")
            self.CIFAR_conv4 = RandWire(
                    self.node_num, self.p, self.in_channels *2, self.out_channels *4, self.graph_mode, self.is_train, name="CIFAR10_conv4")
            self.CIFAR_classifier = tf.keras.layers.Conv2D(
                    self.out_channels *4, kernel_size=1, strides=(1, 1), padding='same', name="CIFAR10_classifier")
            self.BatchNorm2D_classifier = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)
    
        elif self.model_mode is "CIFAR100":
            self.CIFAR100_conv1 = tf.keras.layers.Conv2D(
                    self.out_channels, kernel_size=3, strides=(1, 1), padding='same', name="CIFAR100_conv1")     
            self.CIFAR100_conv2 = RandWire(
                    self.node_num, self.p, self.in_channels, self.out_channels *2, self.graph_mode, self.is_train, name="CIFAR100_conv2")
            self.CIFAR100_conv3 = RandWire(
                    self.node_num, self.p, self.in_channels *2, self.out_channels *4, self.graph_mode, self.is_train, name="CIFAR100_conv3")
            self.CIFAR100_conv4 = RandWire(
                    self.node_num, self.p, self.in_channels *4, self.out_channels *8, self.graph_mode, self.is_train, name="CIFAR100_conv4")
            self.CIFAR100_classifier = tf.keras.layers.Conv2D(
                    self.out_channels *8, kernel_size=1, strides=(1, 1), padding='same', name="CIFAR100_classifier")
    
        elif self.model_mode is "SMALL_REGIME":
            self.SMALL_conv1 = tf.keras.layers.Conv2D(
                    self.out_channels // 2, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.relu, name="SMALL_conv1")     
            self.SMALL_conv2 = tf.keras.layers.Conv2D(
                    self.out_channels, kernel_size=3, strides=(1, 1), padding='same', name="SMALL_conv2")
            self.SMALL_conv3 = RandWire(
                    self.node_num, self.p, self.in_channels, self.out_channels, self.graph_mode, self.is_train, name="SMALL_conv3")
            self.SMALL_conv4 = RandWire(
                    self.node_num, self.p, self.in_channels, self.out_channels *2, self.graph_mode, self.is_train, name="SMALL_conv4")
            self.SMALL_conv5 = RandWire(
                    self.node_num, self.p, self.in_channels *2, self.out_channels *4, self.graph_mode, self.is_train, name="SMALL_conv5")
            self.SMALL_classifier = tf.keras.layers.Conv2D(
                    self.out_channels *4, kernel_size=1, strides=(1, 1), padding='same', name="SMALL_classifier")
    
        elif self.model_mode is "REGULAR_REGIME":
            self.REGULAR_conv1 = tf.keras.layers.Conv2D(
                    self.in_channels, kernel_size=3, strides=(1, 1), padding='same', name="REGULAR_conv1")   
            self.REGULAR_conv2 = RandWire(
                    self.node_num, self.p, self.in_channels // 2, self.out_channels, self.graph_mode, self.is_train, name="REGULAR_conv2")
            self.REGULAR_conv3 = RandWire(
                    self.node_num, self.p, self.in_channels, self.out_channels *2, self.graph_mode, self.is_train, name="REGULAR_conv3")
            self.REGULAR_conv4 = RandWire(
                    self.node_num, self.p, self.in_channels *2, self.out_channels *4, self.graph_mode, self.is_train, name="REGULAR_conv4")
            self.REGULAR_conv5 = RandWire(
                    self.node_num, self.p, self.in_channels *4, self.out_channels *8, self.graph_mode, self.is_train, name="REGULAR_conv5")
            self.REGULAR_classifier = tf.keras.layers.Conv2D(
                    self.out_channels *8, kernel_size=1, strides=(1, 1), padding='same', name="REGULAR_classifier")
    
        self.Dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.Dense = tf.keras.layers.Dense(1280)
            

    def call(self, x):
        if self.model_mode is "CIFAR10":
            out = self.CIFAR_conv1(x)
            out = self.BatchNorm2D_conv1(out)
            out = self.CIFAR_conv2(out)
            out = self.BatchNorm2D_conv2(out)
            out = self.CIFAR_conv3(out)  
            out = self.CIFAR_conv4(out)
            out = self.CIFAR_classifier(out)
            out = self.BatchNorm2D_classifier(out)
        elif self.model_mode is "CIFAR100":
            out = self.CIFAR100_conv1(x)
            out = self.BatchNormalization(out)
            out = self.CIFAR100_conv2(out)
            out = self.CIFAR100_conv3(out)
            out = self.CIFAR100_conv4(out)
            out = self.CIFAR100_classifier(out)
            out = self.BatchNormalization(out)
        elif self.model_mode is "SMALL_REGIME":
            out = self.SMALL_conv1(x)
            out = self.BatchNormalization(out)
            out = self.SMALL_conv2(out)
            out = self.BatchNormalization(out)
            out = self.SMALL_conv3(out)
            out = self.SMALL_conv4(out)
            out = self.SMALL_conv5(out)
            out = self.SMALL_classifier(out)
            out = self.BatchNormalization(out)
        elif self.model_mode is "REGULAR_REGIME":
            out = self.REGULAR_conv1(x)
            out = self.BatchNormalization(out)
            out = self.REGULAR_conv2(out)
            out = self.REGULAR_conv3(out)
            out = self.REGULAR_conv4(out)
            out = self.REGULAR_conv5(out)
            out = self.REGULAR_classifier(out)
            out = self.BatchNormalization(out)

        # global average pooling
        batch_size, height, width, channels = tuple(out.shape)
        out = tf.nn.avg_pool2d(out, ksize=[height, width], strides=[height,width], padding='VALID')
        out = tf.squeeze(out, axis=[1,2])
        out = self.Dropout(out)
        out = self.Dense(out)
        return out

    
