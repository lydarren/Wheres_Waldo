from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, SGD
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image

#TODO clean up
def folder_to_array(path, label):
    data_location = os.listdir(path)
    for data in data_location:
        image = Image.open(path + '/' + data)
        arr = np.asarray(image, dtype="uint8")
        imgs.append(arr)
        labels.append(int(label))
        

#helper function to create a folder of augmented images
def aug_image_folder(path):
    print("Augmenting images:")
    img_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.2,
                                 height_shift_range=0.2, zoom_range=0,
                                 horizontal_flip=False, vertical_flip=False,
                                 channel_shift_range=10.)

    aug = img_gen.flow_from_directory(os.getcwd() + "/64x64/", target_size=(64, 64),
                                      color_mode="rgb", class_mode="binary",
                                      classes=["waldo"],
                                      save_to_dir=(os.getcwd() + "/a_waldos/"),
                                      save_format='jpg')
    
    aug_imgs = [next(aug)[0].astype(np.uint8) for i in range(20)]
    
    return aug_imgs
        

def create_model(shape):
    
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same',
                     activation="relu", input_shape=shape))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    
    model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    
        
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(num_classes, activation="softmax"))
    return model


def maxpool_dropout_vgg(shape):  
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),padding='same',
                     activation="relu", input_shape=shape))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(num_classes, activation="softmax"))
    return model
    
    
    
def train_model(x_train, x_test, y_train, y_test, model=None, name='model.h5'):
    if model is None:
        model = create_model()    
    
    #standard sgd
    opt = SGD(lr=learning_rate)
    #vgg sgd
    #opt = SGD(lr=learning_rate, momentum=0.9)
    
    model.compile(loss="categorical_crossentropy", 
                  optimizer=opt, metrics=["accuracy"])
    
    history = model.fit(x_train, y_train, batch_size=batch_size, 
                        epochs=epochs, validation_data=(x_test, y_test),
                        verbose=1, shuffle=True)
    
    model.save(name)
    
    model.summary()
    show_graph(history)
    return model


def is_arr_waldo(arr):
    
    arr = arr.astype("float32")
    arr /= 255
    a = np.expand_dims(arr, axis = 0)
    pred = model.predict_classes(a)
    
    prob = model.predict(a)
    
    if int(pred) == 1:
        return True, prob[0][1]
    
    return False, prob[0][0]
    
    
def show_graph(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_acc"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)
    

def find_waldo(image_path, k):
    image = Image.open(image_path)
    print(image.size)
    arr = np.asarray(image, dtype="uint8")
    height = len(arr)
    width = len(arr[0])
    #todo make whiteout more efficient
    img_file_white = Image.new("RGB", (width, height), "white")
    img_blended = Image.blend(image, img_file_white, 0.8)
    
    print("Height: ", height)
    print("Width: ", width)
    for i in range(0, height-64,32):
        for j in range(0, width-64,32):
            b, prob = is_arr_waldo(arr[i:i+64,j:j+64])
            if(b):
                ssave = Image.fromarray(arr[i:i+64,j:j+64], "RGB")
                img_blended.paste(ssave, (j, i))
                
    img_blended.save("solved/" + str(k) + ".jpg")
    
    
def load(model_name):
    model = load_model(model_name)
    model.load_weights(model_name)
    return model
    
batch_size = 64
num_classes = 2
epochs = 100
learning_rate = 0.01
model_name = "model_name.h5"

#model = load(model_name)

imgs = []
labels = []

path_one = os.getcwd() + "/64x64/waldo"
path_two = os.getcwd() + "/64x64/notwaldo"
path_three = os.getcwd() + "/aug_waldos/"

folder_to_array(path_one, 1)
folder_to_array(path_three, 1)
folder_to_array(path_one, 1)
folder_to_array(path_three, 1)
folder_to_array(path_two, 0)

data = np.array(imgs)
data = data.astype("float32")
input_shape = data[0].shape

print("Input shape: ", input_shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=2)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

print("X_train shape: ", x_train.shape)
print("X_test shape: ", x_test.shape)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print("Y_train shape: ", y_train.shape)
print("Y test shape: ", y_test.shape)


model = create_model(input_shape)
model = train_model(x_train, x_test, y_train, y_test, model, name=model_name)

score = model.evaluate(x_test, y_test, verbose=0)

print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

for x in range(59, 75):
    find_waldo(os.getcwd() + "/testing/" + str(x) + ".jpg", x)
