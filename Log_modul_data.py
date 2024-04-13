# necessary imports
import tensorflow as tf
import matplotlib.pyplot as plt
import config

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


def log_gan(train_summary_writer, gan, noise, epoch):

    #
    # Write to TensorBoard
    #
    
    with train_summary_writer.as_default():

        if epoch != 0:
            generator_loss = gan.generator.metric_loss.result()
            tf.summary.scalar(f"generator_loss", generator_loss, step=epoch)
            gan.generator.metric_loss.reset_states()

            for metric in gan.discriminator.metrics:
                tf.summary.scalar(f"discriminator_{metric.name}", metric.result(), step=epoch)
                print(f"discriminator_{metric.name}: {metric.result()}")
                metric.reset_states()

        if epoch % 10 == 0:
            generated_imgs = gan.generator(noise, training=False)
            # Normalize
            generated_imgs = (generated_imgs + 1)/2
            tf.summary.image(name="generated_imgs",data = generated_imgs, step=epoch, max_outputs=config.batch_size)

            save_gan_plot(generated_imgs, epoch)
        
def save_gan_plot(examples, epoch, n = 5):
    examples = (examples + 1) / 2.0
	# plot images
    plt.figure(figsize=(10, 10))
    for i in range(n * n):
		# define subplot
        plt.subplot(n, n, 1 + i)
		# turn off axis
        plt.axis('off')
		# plot raw pixel data
        plt.imshow(examples[i])
	# save plot to file
    filename = 'ImageGan/result_e%05d.png' % (epoch)
    plt.savefig(filename)
    plt.close()