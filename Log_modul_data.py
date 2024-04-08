# necessary imports
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

train_res_los = []
train_res_acc = []
test_res_los = []
test_res_acc = []

epoch_train_loss = tf.keras.metrics
epoch_train_acc = tf.keras.metrics
epoch_test_loss = tf.keras.metrics
epoch_test_acc = tf.keras.metrics

outputString = ""

def logging_after_train(network):
    epoch_train_loss = network.metric_loss.result()
    epoch_train_acc = network.metric_accuracy.result()

    print(f" Train accurracy: {epoch_train_acc}, loss: {epoch_train_loss}")

    train_res_los.append(epoch_train_loss.numpy())
    train_res_acc.append(epoch_train_acc.numpy())

    network.metric_loss.reset_states()
    network.metric_accuracy.reset_states()

def logging_after_test(network, epoch):
    epoch_test_loss = network.metric_loss.result()
    epoch_test_acc = network.metric_accuracy.result()

    print(f"in Epoch " + str(epoch) + f" - Test accuracy:{epoch_test_acc} , loss {epoch_test_loss}")

    test_res_los.append(epoch_test_loss.numpy())
    test_res_acc.append(epoch_test_acc.numpy())

    network.metric_loss.reset_states()
    network.metric_accuracy.reset_states()

def show_graph():
    plotting = {}

    plotting["train"] = [train_res_los, train_res_acc, test_res_los, test_res_acc]

    fig, axs = plt.subplots(2, 1, figsize=(15, 20))
    for ax, key in zip(axs.flat, plotting.keys()):
    
        train_l, train_a, test_l, test_a = plotting[key]
    
        line1, = ax.plot(train_l)
        line2, = ax.plot(train_a)
        line3, = ax.plot(test_l)
        line4, = ax.plot(test_a)
        ax.legend((line1,line2, line3, line4),("training loss", "train accuracy", "test loss", "test accuracy"))
        ax.set_title(key)
        ax.set(xlabel="epochs", ylabel="Loss/Accuracy")
        #ax.label_outer()
    
    plt.show()