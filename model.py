import csv
import cv2
import numpy as np
import pickle

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from PIL import Image

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

print('Read data...')
def readData(data):
    lines = []

    with open(data + '/driving_log.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)

    lines = lines[1:]

    images = []
    measurements = []
    correction = [0.0,0.2,-0.2]
    count = 0
    zero = 0
    zc = 0
    for line in lines:
        measurement = float(line[3])
        #if(abs(measurement) < 0.5 and zero != 0):
        #    zero = (zero + 1) % 6
        #    continue
        count += 1
        if(count % 500 == 0):
            print("count {}".format(count))
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = data + '/IMG/' + filename
            #image = cv2.imread(current_path)
            image = Image.open(current_path)
            image = np.asarray(image)
            images.append(image)
            measurement += correction[i]
            measurements.append(measurement)
            images.append(cv2.flip(image,1))
            measurements.append((-1)*measurement)

    return images, measurements
images1, measurements1 = readData('data')
images2, measurements2 = readData('data_jungle')
images = images1 + images2
measurements = measurements1 + measurements2

X_train = np.asarray(images)
y_train = np.asarray(measurements)

print("Create the model...")

#model = load_model('model_serverr7.h5')

model = Sequential()
model.add(Cropping2D(cropping = ((70,25),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(1024))
#model.add(Dropout(0.5))
model.add(Dense(100))
#model.add(Dropout(0.5))
model.add(Dense(40))
#model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
#model.load_weights('model_weights.h5')
model.compile(loss = 'mse', optimizer = 'adam')
print("Training...")
print("Train model...")
# model.train_with_batch(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 1)
# model.fit_generator(X_train, y_train, samples_per_epoch=1000, nb_epoch=2,shuffle = True,validation_split = 0.2)
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, \
                          verbose=1, mode='auto')
callbacks_list = [earlystop]

model_info = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 6)
print("Save model...")
model.save_weights('model_weights.h5')
model.save('modell.h5')


model = load_model('modell.h5')
print(model.summary())

plot_model_history(model_info)
