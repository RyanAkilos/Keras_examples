import os, cv2
import numpy as np
import math
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
# %%

train_data_path = 'F://data//train'
test_data_path = 'F://data//validation'
data_dir_list = os.listdir(train_data_path)

print(data_dir_list)

img_rows = 150
img_cols = 150
num_channel = 3
epochs = 1
batch_size = 32

# Define the number of classes
num_train_classes = 3
num_test_classes = 3
num_classes = 3

train_img_data_list = []
test_img_data_list = []

for dataset in data_dir_list:
    train_img_list = os.listdir(train_data_path + '/' + dataset)
    print('Loaded the images of train dataset-' + '{}\n'.format(dataset))
    test_img_list = os.listdir(test_data_path + '/' + dataset)
    print('Loaded the images of test dataset-' + '{}\n'.format(dataset))
    for train_img in train_img_list:
        train_input_img = cv2.imread(train_data_path + '/' + dataset + '/' + train_img)
        train_input_img_resize = cv2.resize(train_input_img, (150, 150))
        train_img_data_list.append(train_input_img_resize)
    for test_img in test_img_list:
        test_input_img = cv2.imread(test_data_path + '/' + dataset + '/' + test_img)
        test_input_img_resize = cv2.resize(test_input_img, (150, 150))
        test_img_data_list.append(test_input_img_resize)


train_img_data = np.array(train_img_data_list)
train_img_data = train_img_data.astype('float32')
train_img_data /= 255
print(train_img_data.shape)
test_img_data = np.array(test_img_data_list)
test_img_data = test_img_data.astype('float32')
test_img_data /= 255
print(test_img_data.shape)

# Assigning Labels
num_of_train_samples = train_img_data.shape[0]
train_labels = np.ones((num_of_train_samples,), dtype='int64')
train_labels[0:1000] = 0
train_labels[1000:2000] = 1
train_labels[2000:3000] = 2
print(num_of_train_samples)
num_of_test_samples = test_img_data.shape[0]
test_labels = np.ones((num_of_test_samples,), dtype='int64')
test_labels[0:400] = 0
test_labels[400:800] = 1
test_labels[800:1200] = 2
print(num_of_test_samples)

Y_train = np_utils.to_categorical(train_labels, num_train_classes)
x_train, y_train = shuffle(train_img_data, Y_train, random_state=2)
Y_test = np_utils.to_categorical(test_labels, num_test_classes)
x_test, y_test = shuffle(test_img_data, Y_test, random_state=2)

# Build Model
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(img_rows, img_cols, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))


# ROC curve, Micro-average ROC curve and ROC area
fpr = dict()
tpr = dict()
roc_auc = dict()
y_score = model.predict_proba(x_test)

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
print(all_fpr)
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= num_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'cornflowerblue', 'darkorange'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])

plt.plot([0, 1], [0, 1], 'k--', color='navy', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True positive rate')
plt.title('Multi-class ROC curve')
plt.legend(loc="lower right")
plt.show()
