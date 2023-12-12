import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '.', 'path to store summary')

def main(_):
    # the path to save models
    save_path = './model/'
    print('reading wordembedding')
    wordembedding = np.load('./data/vec.npy')

    print('reading training data')
    train_y = np.load('./data/train_y.npy')
    train_word = np.load('./data/train_word.npy', allow_pickle=True)
    train_pos1 = np.load('./data/train_pos1.npy', allow_pickle=True)
    train_pos2 = np.load('./data/train_pos2.npy', allow_pickle=True)

    settings = network.Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(train_y[0])

    big_num = settings.big_num

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = network.GRU(is_training=True, word_embeddings=wordembedding, settings=settings)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.0005)

            train_op = optimizer.minimize(m.final_loss, global_step=global_step)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)

            merged_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train_loss', sess.graph)

            # Variables to track best accuracy and loss
            best_acc = 0.0
            best_loss = float('inf')

            # Lists to store accuracy and loss for each step
            acc_list = []
            loss_list = []

            # Create directories for saving logs
            log_dir_acc = './logs/accuracy/'
            log_dir_loss = './logs/loss/'
            os.makedirs(log_dir_acc, exist_ok=True)
            os.makedirs(log_dir_loss, exist_ok=True)

            def train_step(word_batch, pos1_batch, pos2_batch, y_batch, big_num):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch

                _, step, loss, accuracy, summary, l2_loss, final_loss = sess.run(
                    [train_op, global_step, m.total_loss, m.accuracy, merged_summary, m.l2_loss, m.final_loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                accuracy = np.reshape(np.array(accuracy), (big_num))
                acc = np.mean(accuracy)
                summary_writer.add_summary(summary, step)

                if step % 10 == 0:
                    tempstr = "{}: step {}, softmax_loss {:g}, acc {:g}".format(time_str, step, loss, acc)
                    print(tempstr)

                # Accumulate accuracy and loss for each step
                acc_list.append(acc)
                loss_list.append(loss)

            for one_epoch in range(settings.num_epochs):

                temp_order = list(range(len(train_word)))
                np.random.shuffle(temp_order)
                for i in range(int(len(temp_order) / float(settings.big_num))):

                    temp_word = []
                    temp_pos1 = []
                    temp_pos2 = []
                    temp_y = []

                    temp_input = temp_order[i * settings.big_num:(i + 1) * settings.big_num]
                    for k in temp_input:
                        temp_word.append(train_word[k])
                        temp_pos1.append(train_pos1[k])
                        temp_pos2.append(train_pos2[k])
                        temp_y.append(train_y[k])
                    num = 0
                    for single_word in temp_word:
                        num += len(single_word)

                    if num > 1500:
                        print('out of range')
                        continue

                    temp_word = np.array(temp_word)
                    temp_pos1 = np.array(temp_pos1)
                    temp_pos2 = np.array(temp_pos2)
                    temp_y = np.array(temp_y)

                    train_step(temp_word, temp_pos1, temp_pos2, temp_y, settings.big_num)

                    current_step = tf.train.global_step(sess, global_step)
                    if current_step > 3000 and current_step % 100 == 0:
                        print('saving model')
                        path = saver.save(sess, save_path + 'ATT_GRU_model', global_step=current_step)
                        tempstr = 'have saved model to ' + path
                        print(tempstr)

                # Calculate average accuracy and loss for the epoch
                avg_acc = np.mean(acc_list)
                avg_loss = np.mean(loss_list)

                # Print the average accuracy and loss for the epoch
                tempstr = "Epoch {}: avg_loss {:g}, avg_acc {:g}".format(one_epoch, avg_loss, avg_acc)
                print(tempstr)

                # Check if the current epoch has the best accuracy and loss
                if avg_acc > best_acc:
                    best_acc = avg_acc
                if avg_loss < best_loss:
                    best_loss = avg_loss

                log_path = './logs/summary.log'

                with open(log_path, 'a') as f:
                    for acc in acc_list:
                        f.write(f'Accuracy: {acc}\n')

                    avg_acc = np.mean(acc_list)
                    f.write(f'Average Accuracy: {avg_acc}\n')

                    for loss in loss_list:
                        f.write(f'Loss: {loss}\n')

                    avg_loss = np.mean(loss_list)
                    f.write(f'Average Loss: {avg_loss}\n')

                # Clear the lists for the next epoch
                acc_list.clear()
                loss_list.clear()

            # Print the best accuracy and loss
            print("Best accuracy: {:g}".format(best_acc))
            print("Best loss: {:g}".format(best_loss))

            # Plot the loss curve
            plt.plot(loss_list)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.show()

if __name__ == "__main__":
    tf.app.run()
    