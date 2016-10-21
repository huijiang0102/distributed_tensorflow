# Distributed TensorFlow

## Run in standalone

```
python -m trainer.task
```

It supports multiple parameters and you can try this.

```
python -m trainer.task --max_epochs 1000 --export_dir /tmp/linear_model1 --tensorboard_dir /tmp/tensorboard1 --optimizer sgd
```

## Run in distributed

Start the ps server.

```
TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002"], "master": ["127.0.0.1:3003"]}, "task": {"index": 0, "type": "ps"}}' python -m trainer.task
```

Start the worker.

```
TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002"], "master": ["127.0.0.1:3003"]}, "task": {"index": 0, "type": "worker"}}' python -m trainer.task
```

Start the master.

```
TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002"], "master": ["127.0.0.1:3003"]}, "task": {"index": 0, "type": "master"}}' python -m trainer.task
```
