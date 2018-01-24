# Distributed TensorFlow

## Run in standalone

```
python -m trainer.task
```

## Run in distributed

Start the ps server.

```
CUDA_VISIBLE_DEVICES='' TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002"], "master": ["127.0.0.1:3003"]}, "task": {"index": 0, "type": "ps"}}' python -m trainer.task
```

Start the worker.

```
CUDA_VISIBLE_DEVICES='' TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002"], "master": ["127.0.0.1:3003"]}, "task": {"index": 0, "type": "worker"}}' python -m trainer.task
```

Start the master.

```
CUDA_VISIBLE_DEVICES='' TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002"], "master": ["127.0.0.1:3003"]}, "task": {"index": 0, "type": "master"}}' python -m trainer.task
```

## Serving

Run with [simple_tensorflow_serving](https://github.com/tobegit3hub/simple_tensorflow_serving).

```
simple_tensorflow_serving --port=8500 --model_base_path="./saved_model"
```

Predict with `curl` or other clients.

```
curl -H "Content-Type: application/json" -X POST -d '{"keys": [[11.0], [2.0]], "features": [[1], [2]]}' http://127.0.0.1:8500
```
