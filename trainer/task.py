#!/usr/bin/env python
"""
Usage:

TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"]}, "task": {"index": 0, "type": "ps"}}' python -m trainer.task
TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"]}, "task": {"index": 0, "type": "worker"}}' python -m trainer.task
TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"]}, "task": {"index": 1, "type": "worker"}}' python -m trainer.task
"""

import datetime
import json
import os
import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch_number", 10, "Number of steps to run trainer")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/",
                    "The checkpoint directory")
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate")
FLAGS = flags.FLAGS


def main():
  # Create train dataset
  train_X = np.linspace(-1, 1, 100).reshape((100, 1))
  train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10
  optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
  start_training_time = datetime.datetime.now()

  # Run standalone training
  if os.environ.get('TF_CONFIG', "") == "":
    X_placeholder = tf.placeholder("float", shape=[None, 1])
    Y_placeholder = tf.placeholder("float", shape=[None, 1])
    w = tf.get_variable("w", [1], initializer=tf.random_normal_initializer())
    b = tf.get_variable("b", [1], initializer=tf.random_normal_initializer())
    loss = tf.reduce_sum(tf.square(Y_placeholder - X_placeholder * w - b))
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init_op)

      for epoch_index in range(FLAGS.epoch_number):
        _, loss_value = sess.run(
            [train_op, loss],
            feed_dict={X_placeholder: train_X,
                       Y_placeholder: train_Y})

        if epoch_index % 1 == 0:
          print("Epoch: {}, loss: {}".format(epoch_index, loss_value))

      w_value, b_value = sess.run([w, b])
      end_training_time = datetime.datetime.now()
      print("[{}] End of standalone training, w: {}, b:{}".format(
          end_training_time - start_training_time, w_value, b_value))

  # Run distributed training
  else:
    # Exampmle: {"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"]}, "task": {"index": 0, "type": "worker"}}
    tf_config_env = json.loads(os.environ.get("TF_CONFIG"))
    cluster_spec = tf_config_env.get("cluster")
    task_data = tf_config_env.get("task")
    task_type = task_data.get("type")
    task_index = task_data.get("index")

    cluster = tf.train.ClusterSpec(cluster_spec)
    server = tf.train.Server(
        cluster, job_name=task_type, task_index=task_index)

    if task_type == "ps":
      server.join()
    elif task_type == "worker":

      with tf.device(
          tf.train.replica_device_setter(
              worker_device="/job:worker/task:{}".format(task_index),
              cluster=cluster)):

        X_placeholder = tf.placeholder("float", shape=[None, 1])
        Y_placeholder = tf.placeholder("float", shape=[None, 1])
        w = tf.get_variable(
            "w", [1], initializer=tf.random_normal_initializer())
        b = tf.get_variable(
            "b", [1], initializer=tf.random_normal_initializer())
        loss = tf.reduce_sum(tf.square(Y_placeholder - X_placeholder * w - b))
        global_step = tf.contrib.framework.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

      # hooks=[tf.train.StopAtStepHook(last_step=100)]
      is_chief = task_index == 0
      with tf.train.MonitoredTrainingSession(
          master=server.target,
          is_chief=is_chief,
          checkpoint_dir=FLAGS.checkpoint_dir) as sess:
        while not sess.should_stop():

          for epoch_index in range(FLAGS.epoch_number):
            _, loss_value = sess.run(
                [train_op, loss],
                feed_dict={X_placeholder: train_X,
                           Y_placeholder: train_Y})

            if epoch_index % 1 == 0:
              print("Epoch: {}, loss: {}".format(epoch_index, loss_value))

          w_value, b_value = sess.run([w, b])
          end_training_time = datetime.datetime.now()
          print("[{}] End of standalone training, w: {}, b:{}".format(
              end_training_time - start_training_time, w_value, b_value))
          return


if __name__ == "__main__":
  main()
