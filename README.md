# Super Resolution Application
 Building an application using a SRGAN to return higher res images

## Usage

To run the cloud gpu-server:
```
floyd run --gpu --env tensorflow-2.1 'python train.py' --data [MASKED]
```

To run the training locally:
```
python train.py
```

## generate images

To generate a higher res image:
```
python generate.py --image "filename"
```

## Acknowledgements

* [Research paper](https://arxiv.org/abs/1609.04802)