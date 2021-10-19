import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.datasets import cifar10, cifar100, mnist

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

        (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
        trainDS, testDS = train_images / 255.0, test_images / 255.0
        
        trainDS = tf.data.Dataset.from_tensor_slices(
        (tf.cast(train_images, tf.float32),
         tf.cast(train_labels, tf.int64)))

        testDS = tf.data.Dataset.from_tensor_slices(
        (tf.cast(test_images, tf.float32),
         tf.cast(test_labels, tf.int64)))
        
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

    elif args.dataset_mode is "MNIST":

        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        trainDS, testDS = train_images / 255.0, test_images / 255.0
        
        trainDS = tf.data.Dataset.from_tensor_slices(
        (tf.cast(train_images, tf.float32),
         tf.cast(train_labels, tf.int64)))

        testDS = tf.data.Dataset.from_tensor_slices(
        (tf.cast(test_images, tf.float32),
         tf.cast(test_labels, tf.int64)))
        
        def train_augmentation(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.random_crop(image, (28,28,3))
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