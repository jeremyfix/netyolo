# What is this ?

A simple network synchronous client/server to process an image with Yolo
(ultralytics) on a remote GPU and sending a response back to the client.


# How to use it

## Running the server

On the GPU, you need to run :

```bash
uv venv --python 3.11  $TMPDIR/venv
source $TMPDIR/venv/bin/activate
uv pip install ultralytics flask Pillow
```

And then :

```
python server.py
```

This will start the server waiting for incoming connections

## Running the client

From the client side, you can make the server process an image with a curl
command from the command line such as :

```bash
curl -X POST -H "Content-Type: multipart/form-data" -F "image=@/path/to/your/image.jpg"  http://remote_host_ip:5000/classify
```

You then get a json response you can parse with the following keys :

- top1: class name of the top1
- top1conf : a float in [0, 1] of the top1 confidence
- top5: class names of the top5
- top5conf : confidences of the top5
- names: the names of all the recognized classes
