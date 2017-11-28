# Distributed TensorFlow

## Run in standalone

```
python -m trainer.task
```

## Run in distributed

Start the ps server.

```
CUDA_VISIBLE_DEVICES='' TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"]}, "task": {"index": 0, "type": "ps"}}' python -m trainer.task
```

Start the workers.

```
CUDA_VISIBLE_DEVICES='' TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"]}, "task": {"index": 0, "type": "worker"}}' python -m trainer.task
 
CUDA_VISIBLE_DEVICES='' TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"]}, "task": {"index": 1, "type": "worker"}}' python -m trainer.task 
```
