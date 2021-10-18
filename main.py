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
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
  def __init__(self, initial_learning_rate):
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, epoch):
     return self.initial_learning_rate * (0.1 ** (epoch // 30))
     

@tf.function
def train_step(images, labels, model, optimizer, loss_function, loss_metric, accuracy_metric):
    # 미분을 위한 GradientTape을 적용합니다.
    with tf.GradientTape() as tape:
        # 1. 예측 (prediction)
        predictions = model(images)
        # 2. Loss 계산
        pred_loss = loss_function(labels, predictions)
    
    # 3. 그라디언트(gradients) 계산
    gradients = tape.gradient(pred_loss, model.trainable_variables)
    
    # 4. 오차역전파(Backpropagation) - weight 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # loss와 accuracy를 업데이트 합니다.
    loss_metric.update_state(pred_loss)
    accuracy_metric.update_state(labels, predictions)

@tf.function
def test_step(images, labels, model, loss_function, test_loss, test_acc):
    # 1. 예측 (prediction)
    predictions = model(images)
    # 2. Loss 계산
    loss = loss_function(labels, predictions)
    
    # Test셋에 대해서는 gradient를 계산 및 backpropagation 하지 않습니다.
    
    # loss와 accuracy를 업데이트 합니다.
    test_loss(loss)
    test_acc(labels, predictions)


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

    train_dataset, test_dataset = load_data(args)

    inputs = tf.keras.Input(shape=(32,32,3), batch_size=args.batch_size)

    if args.load_model:
        custom_model = Custom_Model(args.node_num, args.p, args.c, args.k, args.graph_mode, args.model_mode, args.dataset_mode, args.is_train)
        filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_mode_" + args.graph_mode + "_dataset_" + args.dataset_mode
        checkpoint = tf.load_weight('./checkpoint/' + filename + 'ckpt.t7')
        epoch = checkpoint['epoch']
        acc = checkpoint['acc']
        print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)
    else:
        custom_model = Custom_Model(args.node_num, args.p, args.c, args.k, args.graph_mode, args.model_mode, args.dataset_mode, args.is_train)
    
    output = custom_model(inputs)
    model = tf.keras.Model(inputs, output)
    
    model.summary()

    model.build(input_shape=next(iter(train_dataset))[0].shape)
    
    max_test_acc = 0

    if not os.path.isdir("reporting"):
        os.mkdir("reporting")
    
    train_loss_history = []
    accuracy_history = []

    start_time = time.time()
    with open("./reporting/" + "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_mode_" + args.graph_mode + "_dataset_" + args.dataset_mode + ".txt", "w") as f:
        
        #optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        loss_metric = tf.keras.metrics.Mean(name='train_loss')
        accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            

        for epoch in range(1, args.epochs + 1):
            
            myLRSchedule = MyLRSchedule(args.learning_rate)
            optimizer = tf.keras.optimizers.SGD(learning_rate=myLRSchedule(epoch), momentum=0.9)
            
            
            loss_metric.reset_states()
            accuracy_metric.reset_states()

            for images, labels in train_dataset:
                train_step(images, labels, model, optimizer, loss_function, loss_metric, accuracy_metric)
        
            for test_images, test_labels in test_dataset:
                #print(test_images, test_labels)
                test_step(test_images, test_labels, model, loss_function, loss_metric, accuracy_metric)
            
            train_loss_history.append(loss_metric.result())
            accuracy_history.append(accuracy_metric.result())
            print(f'Epoch {epoch}, Loss {loss_metric.result()}, Accuracy {accuracy_metric.result()}')

            #template = '에포크: {}, 손실: {:.5f}, 정확도: {:.2f}%, 테스트 손실: {:.5f}, 테스트 정확도: {:.2f}%'
            #print (template.format(epoch+1, train_loss.result(), train_acc.result(),test_loss.result(), test_acc.result()))

            f.write("[Epoch {0:3d}] Test set accuracy: {1:.3f}%, , Best accuracy: {2:.2f}%".format(epoch, accuracy_metric.result(), max_test_acc))
            f.write("\n ")
            
            if max_test_acc < accuracy_metric.result():
                print('Saving..')
                
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                saved_model_path = "/checkpoint"

                tf.saved_model.save(custom_model, saved_model_path)

                max_test_acc = accuracy_metric.result()
            print("Training time: ", time.time() - start_time)
            f.write("Training time: " + str(time.time() - start_time))
            f.write("\n")
            


if __name__ == '__main__':
    main()