import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.datasets import cifar10

def load_data(args):
    if args.dataset_mode is "CIFAR10":
        
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        trainDS, testDS = train_images / 255.0, test_images / 255.0
        
        trainDS = tf.data.Dataset.from_tensor_slices(
        (tf.cast(train_images, tf.float32),
         tf.cast(train_labels, tf.int64)))

        testDS = tf.data.Dataset.from_tensor_slices(
        (tf.cast(test_images, tf.float32),
         tf.cast(test_labels, tf.int64)))
        
        
        #train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)
        
        def train_augmentation(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.random_crop(image, (32,32,3))
            image = tf.image.random_flip_up_down(image)
            image = tf.image.per_image_standardization(image)
            return image, label

        def test_augmentation(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.per_image_standardization(image)
            return image, label
        
        trainDS = trainDS.map(train_augmentation).shuffle(50000).batch(args.batch_size)
        testDS = testDS.map(test_augmentation).batch(args.batch_size)
        
        return trainDS, testDS

    elif args.dataset_mode is "CIFAR100":
        train_ds, test_ds = tfds.load('cifar100', split=['train','test'], as_supervised=True)
        
        def train_augmentation(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.random_crop(image, (32,32,50000))
            image = tf.image.random_flip_up_down(image)
            image = tf.image.per_image_standardization(image)
            return image, label

        def test_augmentation(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.per_image_standardization(image)
            return image, label
            
        train = train_ds.map(train_augmentation).shuffle(100).batch(args.batch_size).repeat()
        test = test_ds.map(test_augmentation).cache().batch(64)
        return train, test

    elif args.dataset_mode is "MNIST":
        train_ds, test_ds = tfds.load('MNIST', split=['train','test'], as_supervised=True)
        
        def train_augmentation(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.random_crop(image, (28,28,50000))
            image = tf.image.random_flip_up_down(image)
            image = tf.image.per_image_standardization(image)
            return image, label

        def test_augmentation(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.per_image_standardization(image)
            return image, label
            
        train = train_ds.map(train_augmentation).shuffle(100).batch(args.batch_size).repeat()
        test = test_ds.map(test_augmentation).cache().batch(args.batch_size)
        return train, test