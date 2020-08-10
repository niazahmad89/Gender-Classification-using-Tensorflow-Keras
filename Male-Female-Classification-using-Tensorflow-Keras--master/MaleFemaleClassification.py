import os, cv2, random
import numpy as np
import pandas as pd
#pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import ticker
import seaborn as sns
#matplotlib inline

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
#from keras.utils import np_utils



# loading labels for each image from csv
labels = pd.read_csv('results.csv')
labels = labels.iloc[:,0:2]
labels.head()

# Separating male labels
male_data = labels[labels['Gender'] == 0]
male_data.head()

# Splitting male data into train and test
test_male_data = male_data.iloc[-3:,:]
train_male_data = male_data.iloc[:-3,:]
#print(train_male_data)

# Separating female labels
female_data = labels[labels['Gender'] == 1]
female_data.head()

# Splitting male data into train and test
test_female_data = female_data.iloc[-3:,:]
train_female_data = female_data.iloc[:-3,:]


# Displaying image
#img=mpimg.imread('final/Raw_0016_011_20050913100034_Portrait.png')
#imgplot = plt.imshow(img)
#plt.show()

# total test data
test_indices = test_female_data.index.tolist() + test_male_data.index.tolist()
test_data = labels.iloc[test_indices,:]
test_data.head()

# total train data
train_data = pd.concat([labels, test_data, test_data]).drop_duplicates(keep=False)
train_data.head()

# checking count of male and females
sns.countplot(labels['Gender'])



# train and test with image name along with paths
path = 'C:/Users/niaza/MaleFemaleClassification/final/' # path of your image folder
train_image_name = [path+each for each in train_data['Filename'].values.tolist()]
print(train_image_name)
test_image_name = [path+each for each in test_data['Filename'].values.tolist()]

# preparing data by processing images using opencv
ROWS = 64
COLS = 64
CHANNELS = 3


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i % 5 == 0: print('Processed {} of {}'.format(i, count))

    return data


train = prep_data(train_image_name)
test = prep_data(test_image_name)


# plotting female and male side by side
def show_male_and_female():
    female = read_image(train_image_name[0])
    male = read_image(train_image_name[2])
    pair = np.concatenate((female, male), axis=1)
    plt.figure(figsize=(10, 5))
    plt.imshow(pair)
    plt.show()


show_male_and_female()



# splitting path of all images into male and female
train_male_image = []
train_female_image = []
for each in train_image_name:
    if each.split('/')[0] in train_male_data['Filename'].values:
        train_male_image.append(each)
    else:
       train_female_image.append(each)




optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'


def malefemale():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, padding='same', input_shape=(3, ROWS, COLS), activation='relu'))
    model.add(Convolution2D(32, 3, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format="channels_first"))
    #model.add(MaxPooling2D(pool_size=(2, 2),  data_format="channels_first"))

    model.add(Convolution2D(64, 3, 3, padding='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format="channels_first"))

    model.add(Convolution2D(128, 3, 3, padding='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format="channels_first"))
    #model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Convolution2D(256, 3, 3, padding='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format="channels_first"))
    #     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model


model = malefemale()

model.summary()

nb_epoch = 1000
batch_size = 16
labs = train_data.iloc[:, 1].values.tolist()
labs = np.array(labs)


## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
history = LossHistory()

model.fit(train, labs, batch_size=batch_size, epochs=nb_epoch,
              validation_split=0.25, verbose=0, shuffle=True, callbacks=[history, early_stopping])

predictions = model.predict(test, verbose=0)

loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VGG-16 Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,nb_epoch)[0::2])
plt.legend()
plt.show()



for i in range(0, 6):
    if predictions[i, 0] >= 0.5:
        print('I am {:.2%} sure this is a Female'.format(predictions[i][0]))
    else:
        print('I am {:.2%} sure this is a Male'.format(1 - predictions[i][0]))

    plt.imshow(test[i].T)
    plt.show()


