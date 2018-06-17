import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import csv
from sklearn.model_selection import train_test_split
import random
import sklearn
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dense, Flatten

HEIGHT, WIDTH, CHANNELS = 66, 200, 3 


def read_driver_log(path):
    """Read driving_log of simulator and extract path to center, left and right camera images and steering angle."""
    data_set = []
    with open(path) as csvfile:
        dir_name = os.path.dirname(path)
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            data_set.append({'center': os.path.join(dir_name, line[0].strip()), 
                             'left': os.path.join(dir_name, line[1].strip()), 
                             'right': os.path.join(dir_name, line[2].strip()), 
                             'steering': float(line[3])})
    return data_set


def get_image(sample):
    """Get either the iamge of center, left or right camera and change steering angle appropriately."""
    label = random.choice(['left', 'center', 'right'])
    image = mpimg.imread(sample[label])
    image = image.astype('uint8')
    if label=='left':
        angle = sample['steering'] + 0.2
    elif label=='center':
        angle = sample['steering']
    if label=='right':
        angle = sample['steering'] - 0.2
    return image, angle


def pre_process(image):
    """Pre-process image:
       - convert to other colorspace
       - crop image
       - resize image
    """
    # transform to other colorspace
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    # crop image
    image = image[65:125, :, :]
    # resize image
    image = cv2.resize(image, (WIDTH, HEIGHT))
    return image


def flip(image, angle, prob):
    """Flip every 'prob' image and invert its steering angle."""
    if np.random.rand()<prob:
        image = cv2.flip(image, 1)
        angle = -1*angle
    return image, angle


def brightness(image):
    """Change brightness of image randomly (maximum 30%)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    scale = 1.0 + np.random.uniform(-0.3, 0.3)
    hsv[:,:,2] =  hsv[:,:,2] * scale
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augmentation(image, angle):
    """Apply augmentation techniques to image.
    
    - flip image
    - change brightness of image
    
    Args:
       image -- orginal image
       angle -- original streering angle
    """
    image, angle = flip(image, angle, 0.5)
    image = brightness(image)
    return image, angle

        
def batch_generator(data_set, batch_size=32, is_training=False):
    """Return batches of size batch_size for each function call."""
    num_samples = len(data_set)
    # one iteration of while-loop corresponds to one epoch
    while True:
        data_set = sklearn.utils.shuffle(data_set)
        for offset in range(0, num_samples, batch_size):
            batch = data_set[offset:offset+batch_size]
            X_train = np.empty([batch_size, HEIGHT, WIDTH, CHANNELS], dtype=np.uint8)
            y_train = np.empty(batch_size)
            for idx, sample in enumerate(batch):
                # choose randomly one of the three images (to get the right batch_size)
                image, angle = get_image(sample)
                if is_training:
                    image, angle = augmentation(image, angle)
                image = pre_process(image)                
                X_train[idx] = image
                y_train[idx] = angle
            yield X_train, y_train


def nivida_model():
    """Return handle to 'NIVIDA' model."""
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(HEIGHT, WIDTH, CHANNELS)))
    model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    return model


def main():
    data_set = read_driver_log('./my_data/driving_log.csv')
    
    training_set, validation_set = train_test_split(data_set, test_size=0.2)
    training_generator = batch_generator(training_set, batch_size=32, is_training=True)
    validation_generator = batch_generator(validation_set, batch_size=32)

    model = nivida_model()
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(training_generator, 
                    samples_per_epoch=len(training_set), 
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_set), 
                    nb_epoch=5,
                    verbose=1)
    model.save('model.h5')


if __name__=='__main__':
    main()




