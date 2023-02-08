# Inter IIT Tech PS 3 - Flask Server

# Setup

- Make sure your python version is `3.9.13`
- Create a virtual env `python3 -m venv venv`
- Activate the venv `source venv/bin/activate`
- Install all dependencies `pip install -r requirements.txt`

# About our method

- We use a deep convolutional neural network called the U-Net to remove the background of image
- We then identify the approximate edges of the object and calculate the edge-lengths in pixel space.
- Thereafter, we make the pixel to centimeters conversion with the help of a reference object of known dimensions.
- Finally, we use the edge-lengths in centimeters to obtain the volumetric weight of the given object.
