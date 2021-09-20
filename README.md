# Real-Time Traffic Light Detection and Classification

This repository represents a comvolutopnal neural network that founds and classifies traffic lights on the video in real-time.

This model consists of 2 parts:
* detection of the traffic light and enclosing it in a rectangle bounding box,
* classification of the found traffic light into one of three classes: red, green or yellow.

<img width="1043" alt="TT model" src="https://user-images.githubusercontent.com/75208340/113351201-30750780-9343-11eb-8a4c-c7ef851c2d5e.png">


## Detection

For detection of the traffic lights pretrained YOLOv5s was used. YOLOv5s was chosen from the others for its speed. As predictions should be made in real-time, it is optimal to chose the fastest model.

Pretrained ckeckpoints can be seen in the table. 

<img width="437" alt="image" src="https://user-images.githubusercontent.com/75208340/113350090-a8423280-9341-11eb-80b0-c8b1f5e87973.png">

* APtest denotes COCO test-dev2017 server results, all other AP results denote val2017 accuracy.
* All AP numbers are for single-model single-scale without ensemble or TTA. Reproduce mAP by python test.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65
* SpeedGPU averaged over 5000 COCO val2017 images using a GCP n1-standard-16 V100 instance, and includes image preprocessing, FP16 inference, postprocessing and NMS. NMS is 1-2ms/img. Reproduce speed by python test.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45
* All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation). 
* Test Time Augmentation (TTA) runs at 3 image sizes. Reproduce TTA by python test.py --data coco.yaml --img 832 --iou 0.65 --augment

<img width="289" alt="image" src="https://user-images.githubusercontent.com/75208340/113350170-c9a31e80-9341-11eb-9a32-7fba5019925e.png">

Yolo model encloses each traffic light in a rectangle bounding box. On the next step, those bounding boxes are used to crop image, which is then resized to 64x32 ot insure that all inputs to classifier are the same.

## Classification

For classification we present S&P model. Its architecture consists of 4 convolutional blocks, composed of:
* convolutional layer with 3x3 kernel and padding=1
* batch normalization
* activation function LeakyReLU
* max pooling with 2x2 kernel

The classifier returns one of 3 classes: red, green or yellow for each traffic light, detected by Yolo.
(Yellow may contain unknown traffic lights as there was no separately labeled ‘unknown’ class in the dataset, which was used for classificator training).

For training Lisa dataset from Kaggle was used: https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset 

Lisa database is collected in San Diego, California, USA. The database provides four day-time and two night-time sequences, providing 23 minutes and 25 seconds of driving in Pacific Beach and La Jolla, San Diego. The stereo image pairs are acquired using the Point Grey’s Bumblebee XB3 (BBX3-13S2C-60) which contains three lenses which capture images with a resolution of 1280 x 960, each with a Field of View(FoV) of 66°. Where the left camera view is used for all the test sequences and training clips. The training clips consists of 13 daytime clips and 5 nighttime clips.

For this model training only day train data was used. 
The training set consists of 24 683 images, the validation set consists of 13 127 images.

Hyperparameters of the model were optimized with Optuna. Optimized parameters are:
* amount of convolutional blocks: 1
* amount of out channels: 16
* size of hidden layer: 256
* batchsize: 16
* learning rate: 0.00012484125769088617
* learning factor: 0.13325486000996511
* learning patience: 4
* optimizer: Adam

Here are the graphs of training process with validation. Different set of graphs for trainings with the different hyperparameters.
They show loss and accuracy on each epoch.

![Unknown](https://user-images.githubusercontent.com/75208340/113351320-5dc1b580-9343-11eb-831c-2b56145f6a94.png)
![Unknown-3](https://user-images.githubusercontent.com/75208340/113361241-2f000b00-9354-11eb-9385-59f2d7ce6195.png)
