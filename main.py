import tensorflow as tf
import argparse
import os
import time
from tqdm import tqdm
from model import Custom_Model
from preprocess import load_data
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs, (default: 100)')
    parser.add_argument('--p', type=float, default=0.75, help='graph probability, (default: 0.75)')
    parser.add_argument('--c', type=int, default=78, help='channel count for each node, (example: 78, 109, 154), (default: 154)')
    parser.add_argument('--k', type=int, default=4, help='each node is connected to k nearest neighbors in ring topology, (default: 4)')
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

    train_dataset, eval_dataset = load_data(args)

    if args.load_model:
        model = Custom_Model(args.node_num, args.p, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode, args.is_train)
        filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_mode_" + args.graph_mode + "_dataset_" + args.dataset_mode
        checkpoint = tf.load_weight('./checkpoint/' + filename + 'ckpt.t7')
        epoch = checkpoint['epoch']
        acc = checkpoint['acc']
        print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)
    else:
        model = Custom_Model(args.node_num, args.p, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode, args.is_train)

    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.9)
    #criterion = tf.keras.losses.CategoricalCrossentropy()
    #lr_history = LearningRate()
    callback = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: args.learning_rate * (0.1 ** (epoch // 30)),
        verbose=True
    )

    epoch_list = []
    test_loss_list = []
    test_acc_list = []
    train_acc_list = []
    train_loss_list = []
    max_test_acc = 0
    if not os.path.isdir("reporting"):
        os.mkdir("reporting")

    start_time = time.time()
    with open("./reporting/" + "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_mode_" + args.graph_mode + "_dataset_" + args.dataset_mode + ".txt", "w") as f:
        for epoch in range(1, args.epochs + 1):
            # scheduler = CosineAnnealingLR(optimizer, epoch)
            epoch_list.append(epoch)

            model.compile(loss='categorical_crossentropy', optimizer=optimizer)
            model.fit(train_dataset, epochs=epoch, steps_per_epoch=60000 // args.batch_size, callbacks=[callback])
            test_acc, test_loss = model.evaluate(eval_dataset)

            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            #train_loss_list.append(train_loss)
            #train_acc_list.append(train_acc)
            print('Test set accuracy: {0:.2f}%, Best accuracy: {1:.2f}%'.format(test_acc, max_test_acc))
            f.write("[Epoch {0:3d}] Test set accuracy: {1:.3f}%, , Best accuracy: {2:.2f}%".format(epoch, test_acc, max_test_acc))
            f.write("\n ")

            if max_test_acc < test_acc:
                print('Saving..')
                state = {
                    'model': model.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                filename = "c_" + str(args.c) + "_p_" + str(
                    args.p) + "_graph_mode_" + args.graph_mode + "_dataset_" + args.dataset_mode
                tf.save(state, './checkpoint/' + filename + 'ckpt.t7')
                max_test_acc = test_acc
                #draw_plot(epoch_list, train_loss_list, train_acc_list, test_acc_list)
            print("Training time: ", time.time() - start_time)
            f.write("Training time: " + str(time.time() - start_time))
            f.write("\n")


if __name__ == '__main__':
    main()
