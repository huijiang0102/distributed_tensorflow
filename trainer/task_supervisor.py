#!/usr/bin/env python

"""
Usage:

python -m trainer.task_supervisor

CUDA_VISIBLE_DEVICES='' TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002"], "master": ["127.0.0.1:3003"]}, "task": {"index": 0, "type": "ps"}}' python -m trainer.task_supervisor
CUDA_VISIBLE_DEVICES='0' TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002"], "master": ["127.0.0.1:3003"]}, "task": {"index": 0, "type": "worker"}}' python -m trainer.task_supervisor
CUDA_VISIBLE_DEVICES='1' TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002"], "master": ["127.0.0.1:3003"]}, "task": {"index": 0, "type": "master"}}' python -m trainer.task_supervisor
"""

import datetime
import json
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

flags = tf.app.flags
flags.DEFINE_integer('max_epochs', 100, 'Number of steps to run trainer.')
flags.DEFINE_string('checkpoint_dir', './checkpoint/',
                    'The checkpoint directory')
flags.DEFINE_string("tensorboard_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_integer('checkpoint_period', 1,
                     'Number of epochs to save checkpoint.')
flags.DEFINE_integer('export_version', 1, 'Version number of the model.')
flags.DEFINE_string('export_dir', '/tmp/linear_model/', 'The model directory')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_string("optimizer", "sgd", "Optimizer to train")
FLAGS = flags.FLAGS


def main():
  # Create train data
  train_X = np.linspace(-1, 1, 100)
  train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10
  learning_rate = FLAGS.learning_rate
  start_training_time = datetime.datetime.now()

  print("Use the optimizer: {}".format(FLAGS.optimizer))
  if FLAGS.optimizer == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif FLAGS.optimizer == "adadelta":
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  elif FLAGS.optimizer == "adagrad":
    optimizer = tf.train.AdagradOptimizer(learning_rate)
  elif FLAGS.optimizer == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif FLAGS.optimizer == "ftrl":
    optimizer = tf.train.FtrlOptimizer(learning_rate)
  elif FLAGS.optimizer == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
  else:
    print("Unknow optimizer: {}, exit now".format(FLAGS.optimizer))
    exit(1)

  # Run standalone training
  if os.environ.get('TF_CONFIG', "") == "":

    # Define the model
    keys_placeholder = tf.placeholder(tf.int32, shape=[None, 1])
    keys = tf.identity(keys_placeholder)
    X = tf.placeholder("float", shape=[None, 1])
    Y = tf.placeholder("float", shape=[None, 1])
    w = tf.Variable(0.0, name="weight")
    b = tf.Variable(0.0, name="bias")
    global_step = tf.Variable(0, name='global_step', trainable=False)
    loss = tf.reduce_sum(tf.square(Y - X * w - b))
    train_op = optimizer.minimize(loss, global_step=global_step)
    predict_op = X * w + b
    tf.summary.scalar('loss', loss)
    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()
    # saver = tf.train.Saver()

    with tf.Session() as sess:
      sess.run(init_op)
      print("Save tensorboard files into: {}".format(FLAGS.tensorboard_dir))
      writer = tf.summary.FileWriter(FLAGS.tensorboard_dir, sess.graph)

      print("Run training with epoch number: {}".format(FLAGS.max_epochs))
      for i in range(FLAGS.max_epochs):
        for (x, y) in zip(train_X, train_Y):
          x = np.array([[x]])
          y = np.array([[y]])
          sess.run(train_op, feed_dict={X: x, Y: y})

        if i % FLAGS.checkpoint_period == 0:
          x = np.array([[train_X[0]]])
          y = np.array([[train_Y[0]]])
          summary_value, loss_value, step = sess.run(
              [summary_op, loss, global_step],
              feed_dict={X: x,
                         Y: y})
          writer.add_summary(summary_value, step)
          print("Epoch: {}, loss: {}".format(i, loss_value))

      end_training_time = datetime.datetime.now()
      print("[{}] End of standalone training.".format(end_training_time -
                                                      start_training_time))
      print("Get the model, w: {}, b: {}".format(sess.run(w), sess.run(b)))
      export_inputs_signature = {"keys": keys_placeholder, "X": X}
      export_outputs_signature = {"keys": keys, "predict": predict_op}
      export_model(sess, export_inputs_signature, export_outputs_signature)

  # Run distributed training
  else:
    # Exampmle: {"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"], "master": ["127.0.0.1:3004"]}, "task": {"index": 0, "type": "master"}}
    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    task_data = env.get('task', None)
    cluster_spec = env["cluster"]
    task_type = task_data["type"]
    task_index = task_data["index"]

    cluster = tf.train.ClusterSpec(cluster_spec)
    server = tf.train.Server(cluster,
                             job_name=task_type,
                             task_index=task_index)

    if task_type == "ps":
      server.join()
    elif task_type == "worker" or task_type == "master":
      with tf.device(tf.train.replica_device_setter(
          worker_device="/job:{}/task:{}".format(task_type, task_index),
          cluster=cluster)):

        # Define the model
        keys_placeholder = tf.placeholder(tf.int32, shape=[None, 1])
        keys = tf.identity(keys_placeholder)
        X = tf.placeholder("float", shape=[None, 1])
        Y = tf.placeholder("float", shape=[None, 1])
        w = tf.Variable(0.0, name="weight")
        b = tf.Variable(0.0, name="bias")
        global_step = tf.Variable(0, name='global_step', trainable=False)
        loss = tf.reduce_sum(tf.square(Y - X * w - b))
        train_op = optimizer.minimize(loss, global_step=global_step)
        predict_op = X * w + b
        tf.summary.scalar('loss', loss)
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        #saver = tf.train.Saver()
        saver = tf.train.Saver(sharded=True)

        sv = tf.train.Supervisor(is_chief=(task_type == "master"),
                                 logdir=FLAGS.checkpoint_dir,
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60)

        try:
          with sv.managed_session(server.target) as sess:
            print(
                "Save tensorboard files into: {}".format(FLAGS.tensorboard_dir)
            )
            writer = tf.summary.FileWriter(FLAGS.tensorboard_dir, sess.graph)

            print(
                "Run training with epoch number: {}".format(FLAGS.max_epochs))
            for i in range(FLAGS.max_epochs):
              for (x, y) in zip(train_X, train_Y):
                x = np.array([[x]])
                y = np.array([[y]])
                sess.run(train_op, feed_dict={X: x, Y: y})

              if i % FLAGS.checkpoint_period == 0:
                x = np.array([[train_X[0]]])
                y = np.array([[train_Y[0]]])
                summary_value, loss_value, step = sess.run(
                    [summary_op, loss, global_step],
                    feed_dict={X: x,
                               Y: y})
                print("Epoch: {}, loss: {}".format(i, loss_value))
                if task_type == "master":
                  writer.add_summary(summary_value, step)

            end_training_time = datetime.datetime.now()
            print("[{}] End of distributed training.".format(
                end_training_time - start_training_time))

            if task_type == "master":
              print 'Exporting trained model to', FLAGS.export_dir
              model_exporter = exporter.Exporter(saver)
              model_exporter.init(
                  sess.graph.as_graph_def(),
                  named_graph_signatures={
                      'inputs':
                      exporter.generic_signature({"keys": keys_placeholder,
                                                  "X": X}),
                      'outputs':
                      exporter.generic_signature({"keys": keys,
                                                  "predict": predict_op})
                  })
              model_exporter.export(FLAGS.export_dir,
                                    tf.constant(FLAGS.export_version), sess)
              print 'Done exporting!'
        except Exception as e:
          # TODO: Refer to https://github.com/tensorflow/tensorflow/issues/5110
          print("Fail to export model, refer to tensorflow/tensorflow#5110")


def export_model(sess, inputs_signature, outputs_signature):
  # Export the model for generic inference service
  print 'Exporting trained model to', FLAGS.export_dir
  saver = tf.train.Saver(sharded=True)
  model_exporter = exporter.Exporter(saver)
  model_exporter.init(
      sess.graph.as_graph_def(),
      named_graph_signatures={
          'inputs': exporter.generic_signature(inputs_signature),
          'outputs': exporter.generic_signature(outputs_signature)
      })
  model_exporter.export(FLAGS.export_dir, tf.constant(FLAGS.export_version),
                        sess)
  print 'Done exporting!'


if __name__ == "__main__":
  main()
