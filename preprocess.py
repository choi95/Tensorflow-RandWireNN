import tensorflow as tf
import tensorflow_datasets as tfds


def load_data(args):
    if args.dataset_mode is "CIFAR10":
        train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)
        
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

        train = train_ds.map(train_augmentation).shuffle(100).batch(args.batch_size).repeat()
        test = test_ds.map(test_augmentation).cache().batch(args.batch_size)
        return train, test

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

