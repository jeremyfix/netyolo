# What is this ?

A simple network synchronous client/server to process an image with Yolo
(ultralytics) on a remote GPU and sending a response back to the client.


# How to use it

## Running the server

On the GPU, you need to run :

```bash
python3 -m venv
source venv/bin/activate
python -m pip install ultralytics flask Pillow
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
curl -X POST -H "Content-Type: multipart/form-data" \
     -F "image=@/path/to/your/image.jpg" \
     http://remote_host_ip:5000/process_image
```

