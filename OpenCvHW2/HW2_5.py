# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HW2_2.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import PIL
import datetime


from tensorflow.python.keras.models import load_model
from PIL import ImageOps
from keras.models import Model
from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger                             
from keras.regularizers import l2
from keras.applications.resnet50 import ResNet50
from keras.applications import resnet50
from keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm
tqdm.pandas()

from PyQt5 import QtCore, QtGui, QtWidgets

train_dir = "kaggle/train/"
file_list = os.listdir(train_dir)

#Splitting data into training set and vali sets
train_imgs, tmp_imgs = train_test_split(file_list, train_size=0.8)    
# val_imgs, test_imgs = train_test_split(tmp_imgs, train_size=0.5)
# print(len(train_imgs), len(val_imgs), len(test_imgs))

DOG = "dog"
CAT = "cat"
TRAIN_TOTAL = len(train_imgs)
labels = []
df_train = pd.DataFrame({
    'filename': train_imgs
})
# print(df_train.columns.tolist())
# df_valid = pd.DataFrame({
#     'filename': val_imgs
# })
# print(df_valid.columns.tolist())
# df_test = pd.DataFrame({
#     'filename': test_imgs
# })
# print(df_test.columns.tolist())
# print(len(df_train), len(df_valid), len(df_test))

# Collect the image labels (cat/dog), width/height, and aspect ratio to take a look at the shapes. How far are they from a square shape that the ResNet expects as input?

idx = 0
img_sizes = []
widths = np.zeros(TRAIN_TOTAL, dtype=int)
heights = np.zeros(TRAIN_TOTAL, dtype=int)
aspect_ratios = np.zeros(TRAIN_TOTAL) #defaults to type float
for filename in train_imgs:
    if "cat" in filename.lower():
        labels.append(CAT)
    else:
        labels.append(DOG)
    img = PIL.Image.open(f"{train_dir}/{filename}")
    img_size = img.size
    img_sizes.append(img_size)
    widths[idx] = img_size[0]
    heights[idx] = img_size[1]
    aspect_ratios[idx] = img_size[0]/img_size[1]
    img.close()
    idx += 1

df_train["filename"] = train_imgs
df_train["cat_or_dog"] = labels
label_encoder = LabelEncoder()
df_train["cd_label"] = label_encoder.fit_transform(df_train["cat_or_dog"])
df_train["size"] = img_sizes
df_train["width"] = widths
df_train["height"] = heights
df_train["aspect_ratio"] = aspect_ratios
# df_train.head()

# Some basic attributes for training:
batch_size = 32
img_size = 224 #299 is the input size for some of the pre-trained networks
epochs = 5
    
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(256, 346)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout.addWidget(self.lineEdit)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout.addWidget(self.pushButton_4)
############################################################################
############################################################################
############################################################################        
        self.pushButton.clicked.connect(self.createRestnet)
        self.pushButton_2.clicked.connect(self.showHistory)
        self.pushButton_3.clicked.connect(self.predict_img)
        self.pushButton_4.clicked.connect(self.plot_Augmented_9)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 256, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "1. Show Model Structure"))
        self.pushButton_2.setText(_translate("MainWindow", "2. Show TensorBoard"))
        self.pushButton_3.setText(_translate("MainWindow", "3. Test"))
        self.pushButton_4.setText(_translate("MainWindow", "4. Data Augmentation"))
        
        
    
    def create_generators(self, validation_perc, shuffle=False, horizontal_flip=False, 
                          zoom_range=0, w_shift=0, h_shift=0, rotation_range=0, shear_range=0,
                          fill_zeros=False, preprocess_func=None):
        #the "nearest" mode copies image pixels on borders when shifting/rotation/etc to cover empty space
        fill_mode = "nearest"
        if fill_zeros:
            #with constant mode, we fill created empty space with zeros
            fill_mode = "constant"
            
        #rescale changes pixels from 1-255 integers to 0-1 floats suitable for neural nets
        rescale = 1./255
        if preprocess_func is not None:            
            rescale = None
    
        train_datagen=ImageDataGenerator(
            rescale = rescale, 
            validation_split = validation_perc, #0.25, #subset for validation. seems to be subset='validation' in flow_from_dataframe
            horizontal_flip = horizontal_flip,
            zoom_range = zoom_range,
            width_shift_range = w_shift,
            height_shift_range=h_shift,
            rotation_range=rotation_range,
            shear_range=shear_range,
            fill_mode=fill_mode,
            cval=0,#this is the color value to fill with when "constant" mode used. 0=black
            preprocessing_function=preprocess_func
        )
    
        #Keras has this two-part process of defining generators. 
        train_generator=train_datagen.flow_from_dataframe(
            dataframe=df_train,
            directory=train_dir,
            x_col="filename",
            y_col="cat_or_dog",
            batch_size=batch_size, 
            shuffle=shuffle,
            class_mode="binary",
            #classes=lbls, #list of ouput classes. if not provided, inferred from data
            target_size=(img_size,img_size)
            )
        print('11')
    
        return train_generator, train_datagen    
    
    # In[31]:
    
    
    def plot_batch_9(self):
        # In[27]:
        
        train_generator, valid_generator, test_generator, train_datagen = self.create_generators(0, False, False, 0, 0, 0)        
        train_generator.class_indices        
        class_map = {v: k for k, v in train_generator.class_indices.items()}
        
        
        # In[30]:
        
        font = {'family' : 'normal',
                'weight' : 'normal',
                'size'   : 22}
        
        matplotlib.rc('font', **font)   
    
        train_generator.reset()
        # configure batch size and retrieve one batch of images
        plt.clf() #clears matplotlib data and axes
        #for batch in train_generator:
        plt.figure(figsize=[30,30])
        batch = next(train_generator)
        for x in range(0,9):
        #    print(train_generator.filenames[x])
            plt.subplot(3, 3, x+1)
            plt.imshow(batch[0][x], interpolation='nearest')
            item_label = batch[1][x]
            item_label = class_map[int(item_label)]
            plt.title(item_label)
    
        plt.show()       
    
    def create_model(self, trainable_layer_count):
        base_model = ResNet50(include_top=False,
                       weights='imagenet', #loading weights from dataset, avoiding need for internet conn
                       input_shape=(img_size, img_size, 3))
       #base_model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        if trainable_layer_count == "all":

            for layer in base_model.layers:
                layer.trainable = True
        else:

            for layer in base_model.layers:
                layer.trainable = False

            for layer in base_model.layers[-trainable_layer_count:]:
                layer.trainable = True
        print("base model has {} layers".format(len(base_model.layers)))

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dropout(0.5)(x)

        final_output = Dense(1, activation='sigmoid', name='softmax')(x)
        model = Model(base_model.input, final_output)
        
        return model    
    
    def plot_loss_and_accuracy(self, fit_history):
        plt.clf()
        plt.plot(fit_history.history['acc'])
        plt.plot(fit_history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.clf()
        # summarize history for loss
        plt.plot(fit_history.history['loss'])
        plt.plot(fit_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()    
    
    # In[66]:
    
    def createRestnet(self):
        model = self.create_model(5)
        model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())        
        return model

    def showHistory(self):
        image = cv2.imread('epoch7.jpg')
        height, width, channels = image.shape
        cv2.imshow('Epoch History', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predict_img(self):
        test_dir = "kaggle/test/"
        file_list = os.listdir(test_dir)    
        
        input_num = self.lineEdit.text()
        if input_num:
            input_num = int(input_num)
            idx = input_num - 1
        else:
            input_num = 1
            idx = 0
            
        filename = file_list[idx]
        print(filename)
        img = PIL.Image.open(f"{test_dir}/{filename}")   
        
        net = load_model('model-resnet50-final.h5')
        
        cls_list = ['cat', 'dog']        

        x = img.resize((224, 224))
        x = np.array(x)
        x = np.expand_dims(x, axis = 0)
        pred = net.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        # print(pred)
        # print(top_inds)
        # for i in top_inds:
        #     print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
        plt.title('Class: ' + cls_list[top_inds[0]])
        plt.imshow(img)
        plt.show()
        
    # In[50]:    
    def plot_Augmented_9(self):
        train_generator, train_datagen = self.create_generators(validation_perc = 0, 
                                                                rotation_range=40,                                                                              
                                                                shuffle = True, 
                                                                horizontal_flip = True, 
                                                                zoom_range = 0.2, 
                                                                w_shift = 0.2, 
                                                                h_shift = 0.2,
                                                                fill_zeros = True,
                                                                shear_range=0.2)        
        # In[51]:
                
        class_map = {v: k for k, v in train_generator.class_indices.items()}        
        
        font = {'family' : 'normal',
                'weight' : 'normal',
                'size'   : 22}
        
        matplotlib.rc('font', **font)       
        train_generator.reset()
        plt.clf() #clears matplotlib data and axes
        plt.figure(figsize=[30,30])
        batch = next(train_generator)

        
        for x in range(0,9):
        #    print(train_generator.filenames[x])
            plt.subplot(3, 3, x+1)
            plt.imshow(batch[0][x], interpolation='nearest')
            item_label = batch[1][x]
            item_label = class_map[int(item_label)]
            plt.title(item_label)
    
        plt.show()
        
        image = cv2.imread('Figure1.png')
        height, width, channels = image.shape
        cv2.imshow('Figure1', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    # def trainModel(self):    
    #     train_generator, valid_generator, test_generator, train_datagen = self.create_generators(0, False, False, 0, 0, 0)
    #     df_train.head()
                                               
    #     #the total number of images
    #     train_size = len(train_generator.filenames)
    #     #train_steps is how many steps per epoch Keras runs the genrator.
    #     train_steps = train_size/batch_size
    #     #use 2* number of images to get more augmentations in.
    #     train_steps = int(2*train_steps)
    #     #same for the validation set
    #     valid_size = len(valid_generator.filenames)
    #     valid_steps = valid_size/batch_size
    #     valid_steps = int(2*valid_steps)     
        
    #     # ## Model Callbacks    
    #     # create callbacks list    
        
    #     checkpoint = ModelCheckpoint('Resnet50_best.h5', monitor='val_loss', verbose=1, 
    #                                  save_best_only=True, mode='min', save_weights_only = True)
    #     reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
    #                                        verbose=1, mode='auto', epsilon=0.0001)
    #     early = EarlyStopping(monitor="val_loss", 
    #                           mode="min", 
    #                           patience=7)
        
    #     csv_logger = CSVLogger(filename='training_log.csv',
    #                            separator=',',
    #                            append=True)
        
    #     callbacks_list = [checkpoint, csv_logger, early]
    #     # callbacks_list = [checkpoint, csv_logger, reduceLROnPlat]
        
    #     train_generator.reset()
    #     valid_generator.reset()
    #     model = self.createRestnet()
    #     fit_history = model.fit_generator(
    #             train_generator,
    #             steps_per_epoch=train_steps,
    #             epochs = epochs,
    #             validation_data=valid_generator,
    #             validation_steps=valid_steps,
    #             callbacks=callbacks_list,
    #         verbose = 1
    #     )
    #     model.load_weights("Resnet50_best.h5")
                
    #     # As visible, at best this scores close to 98% validation accuracy at best. Nice. Clearly better than the other two above.
                
    #     pd.DataFrame(fit_history.history).head(20)    
    #     self.plot_loss_and_accuracy(fit_history)

# matplotlib bar chart (Figure1.png)
# left = np.array(["Origin", "Augmented"])
# height = np.array([83, 95])
# ax = plt.gca()
# minimum = height.min()-5
# maximum = height.max()+5
# ax.set_ylim([minimum,maximum])
# plt.bar(left, height, width=0.5)
# plt.show()        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

