import tensorflow as tf
import argparse
import os
import time
from model import Custom_Model
from preprocess import load_data
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@tf.function
def train_step(images, labels, model, optimizer, loss_function, train_loss, train_accuracy):

    with tf.GradientTape() as tape: 
        predictions = model(images, training=True)
        pred_loss = loss_function(labels, predictions)
    
    gradients = tape.gradient(pred_loss, model.trainable_variables)  
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(pred_loss)
    train_accuracy.update_state(labels, predictions)

@tf.function
def test_step(images, labels, model, loss_function, test_loss, test_acc):
  
    predictions = model(images, training=False)
    loss = loss_function(labels, predictions)
    test_loss(loss)
    test_acc(labels, predictions)


def main():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs, (default: 100)')
    parser.add_argument('--p', type=float, default=0.75, help='graph probability, (default: 0.75)')
    parser.add_argument('--c', type=int, default=78, help='channel count for each node, (example: 78, 109, 154), (default: 154)')
    parser.add_argument('--k', type=int, default=3, help='conv2d kernel_size, (default: 3)')
    parser.add_argument('--m', type=int, default=5, help='number of edges to attach from a new node to existing nodes, (default: 5)')
    parser.add_argument('--graph-mode', type=str, default="WS", help="random graph, (Example: ER, WS, BA), (default: WS)")
    parser.add_argument('--node-num', type=int, default=32, help="Number of graph node (default n=32)")
    parser.add_argument('--learning-rate', type=float, default=1e-1, help='learning rate, (default: 1e-1)')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size, (default: 100)')
    parser.add_argument('--model-mode', type=str, default="CIFAR10", help='CIFAR10, CIFAR100, SMALL_REGIME, REGULAR_REGIME, (default: CIFAR10)')
    parser.add_argument('--dataset-mode', type=str, default="CIFAR10", help='Which dataset to use? (Example, CIFAR10, CIFAR100, MNIST), (default: CIFAR10)')
    parser.add_argument('--is-train', type=bool, default=True, help="True if training, False if test. (default: True)")
    parser.add_argument('--load-model', type=bool, default=False)

    args = parser.parse_args()

    train_dataset, test_dataset = load_data(args)
   
    if args.load_model:
        custom_model = Custom_Model(args.node_num, args.p, args.c, args.k, args.graph_mode, args.model_mode, args.dataset_mode, args.is_train)
        filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_mode_" + args.graph_mode + "_dataset_" + args.dataset_mode
        checkpoint = tf.load_weight('./checkpoint/' + filename + 'ckpt.t7')
        epoch = checkpoint['epoch']
        acc = checkpoint['acc']
        print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)
    else:
        custom_model = Custom_Model(args.node_num, args.p, args.c, args.k, args.graph_mode, args.model_mode, args.dataset_mode, args.is_train)

    inputs = tf.keras.Input(shape=(32,32,3), batch_size=args.batch_size)
    output = custom_model(inputs)
    custom_model.summary()
    model = tf.keras.Model(inputs, output)

    max_test_acc = 0  
    train_loss_history = []
    accuracy_history = []

    checkpoint = tf.train.Checkpoint(model)
    
    start_time = time.time()

    with open("./reporting/" + "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_mode_" + args.graph_mode + "_dataset_" + args.dataset_mode + ".txt", "w") as f:

        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        schedulers = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, 18000, 0.9, staircase=True)
        optimizer = tf.keras.optimizers.SGD(learning_rate=schedulers, momentum=0.9)

        for epoch in range(1, args.epochs + 1):
            
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
                       
            for images, labels in train_dataset:   
                train_step(images, labels, model, optimizer, loss_function, train_loss, train_accuracy)
            
            for test_images, test_labels in test_dataset:
                test_step(test_images, test_labels, model, loss_function, test_loss, test_accuracy)   
        
            template = 'epoch: {}, train_loss: {:.4f}, train_acc: {:.2%}, test_loss: {:.4f}, test_acc: {:.2%}'
            print (template.format(epoch,
                           train_loss.result(),
                           train_accuracy.result(),
                           test_loss.result(),
                           test_accuracy.result()))

            train_loss_history.append(train_loss.result())
            accuracy_history.append(train_accuracy.result())

            f.write("[Epoch {0:3d}] Test set accuracy: {1:.3f}%, , Best accuracy: {2:.2f}%".format(epoch, train_accuracy.result(), max_test_acc))
            f.write("\n ")
            
            if max_test_acc < train_accuracy.result():
                print('Saving..')
                
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')

                max_test_acc = train_accuracy.result()
            print("Training time: ", time.time() - start_time)
            f.write("Training time: " + str(time.time() - start_time))
            f.write("\n")
            


if __name__ == '__main__':
    main()
