# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CV_HW1.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
from PIL import Image
import math
# ---------Q5 libraries-------------
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary
# %matplotlib auto
# %matplotlib inline
batch_size = 250
transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False) 
        
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1097, 600)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
############################################################################
#########################   group 1   ######################################
############################################################################        
        self.groupBox_1 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_1.setFont(font)
        self.groupBox_1.setObjectName("groupBox_1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton = QtWidgets.QPushButton(self.groupBox_1)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton.setObjectName("pushButton")        
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_1)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_7.setObjectName("pushButton_7")
        self.verticalLayout.addWidget(self.pushButton_7)
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_1)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.pushButton_8.setFont(font)
        self.pushButton_8.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_8.setObjectName("pushButton_8")
        self.verticalLayout.addWidget(self.pushButton_8)
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_1)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        self.pushButton_9.setFont(font)
        self.pushButton_9.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_9.setObjectName("pushButton_9")
        self.verticalLayout.addWidget(self.pushButton_9)
        self.horizontalLayout.addWidget(self.groupBox_1)
        self.pushButton.clicked.connect(self.showImg)
        self.pushButton_7.clicked.connect(self.separationBGR)
        self.pushButton_8.clicked.connect(self.colorTransformation)
        self.pushButton_9.clicked.connect(self.blendImgs)
############################################################################
#########################   group 2   ######################################
############################################################################
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_10 = QtWidgets.QPushButton(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_10.setFont(font)
        self.pushButton_10.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_10.setObjectName("pushButton_10")
        self.verticalLayout_2.addWidget(self.pushButton_10)
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_11.setFont(font)
        self.pushButton_11.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_11.setObjectName("pushButton_11")
        self.verticalLayout_2.addWidget(self.pushButton_11)
        self.pushButton_12 = QtWidgets.QPushButton(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_12.setFont(font)
        self.pushButton_12.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_12.setObjectName("pushButton_12")
        self.verticalLayout_2.addWidget(self.pushButton_12)
        self.horizontalLayout.addWidget(self.groupBox_2)
        self.pushButton_10.clicked.connect(self.gaussianSmoothing)
        self.pushButton_11.clicked.connect(self.bilateralSmoothing)
        self.pushButton_12.clicked.connect(self.medianSmoothing)
############################################################################
#########################   group 3   ######################################
############################################################################        
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.pushButton_13 = QtWidgets.QPushButton(self.groupBox_3)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_13.setFont(font)
        self.pushButton_13.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_13.setObjectName("pushButton_13")
        self.verticalLayout_3.addWidget(self.pushButton_13)
        self.pushButton_14 = QtWidgets.QPushButton(self.groupBox_3)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_14.setFont(font)
        self.pushButton_14.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_14.setObjectName("pushButton_14")
        self.verticalLayout_3.addWidget(self.pushButton_14)
        self.pushButton_15 = QtWidgets.QPushButton(self.groupBox_3)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_15.setFont(font)
        self.pushButton_15.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_15.setObjectName("pushButton_15")
        self.verticalLayout_3.addWidget(self.pushButton_15)
        self.pushButton_16 = QtWidgets.QPushButton(self.groupBox_3)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_16.setFont(font)
        self.pushButton_16.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_16.setObjectName("pushButton_16")
        self.verticalLayout_3.addWidget(self.pushButton_16)
        self.horizontalLayout.addWidget(self.groupBox_3)
        self.pushButton_13.clicked.connect(self.showblurImg)
        self.pushButton_14.clicked.connect(self.sobelXImg)
        self.pushButton_15.clicked.connect(self.sobelYImg)
        self.pushButton_16.clicked.connect(self.magnitudeImg)
############################################################################
#########################   group 4   ######################################
############################################################################        
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pushButton_17 = QtWidgets.QPushButton(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_17.setFont(font)
        self.pushButton_17.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_17.setObjectName("pushButton_17")
        self.verticalLayout_4.addWidget(self.pushButton_17)
        self.pushButton_18 = QtWidgets.QPushButton(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_18.setFont(font)
        self.pushButton_18.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_18.setObjectName("pushButton_18")
        self.verticalLayout_4.addWidget(self.pushButton_18)
        self.pushButton_19 = QtWidgets.QPushButton(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_19.setFont(font)
        self.pushButton_19.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_19.setObjectName("pushButton_19")
        self.verticalLayout_4.addWidget(self.pushButton_19)
        self.pushButton_20 = QtWidgets.QPushButton(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_20.setFont(font)
        self.pushButton_20.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_20.setObjectName("pushButton_20")
        self.verticalLayout_4.addWidget(self.pushButton_20)
        self.horizontalLayout.addWidget(self.groupBox_4)
        self.pushButton_17.clicked.connect(self.resizeImg)
        self.pushButton_18.clicked.connect(self.translateImg)
        self.pushButton_19.clicked.connect(self.rotate_scaleImg)
        self.pushButton_20.clicked.connect(self.shearImg)
############################################################################
#########################   group 5   ######################################
############################################################################        
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_5)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.pushButton_21 = QtWidgets.QPushButton(self.groupBox_5)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_21.setFont(font)
        self.pushButton_21.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_21.setObjectName("pushButton_21")
        self.verticalLayout_5.addWidget(self.pushButton_21)
        self.pushButton_22 = QtWidgets.QPushButton(self.groupBox_5)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_22.setFont(font)
        self.pushButton_22.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_22.setObjectName("pushButton_22")
        self.verticalLayout_5.addWidget(self.pushButton_22)
        self.pushButton_23 = QtWidgets.QPushButton(self.groupBox_5)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_23.setFont(font)
        self.pushButton_23.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_23.setObjectName("pushButton_23")
        self.verticalLayout_5.addWidget(self.pushButton_23)
        self.pushButton_24 = QtWidgets.QPushButton(self.groupBox_5)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_24.setFont(font)
        self.pushButton_24.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_24.setObjectName("pushButton_24")
        self.verticalLayout_5.addWidget(self.pushButton_24)
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_5)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout_5.addWidget(self.lineEdit)
        self.pushButton_25 = QtWidgets.QPushButton(self.groupBox_5)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        self.pushButton_25.setFont(font)
        self.pushButton_25.setStyleSheet("QPushButton{ background-color: rgb(201, 211, 255); border-radius:12px;}\n"
"QPushButton:hover{border-radius:12px;background-color: rgb(179, 180, 255);color: rgb(255, 255, 255);}\n"
"QPushButton:pressed{border-radius:12px;background-corlor: rgb(179, 180, 255);}")
        self.pushButton_25.setObjectName("pushButton_25")
        self.verticalLayout_5.addWidget(self.pushButton_25)
        self.horizontalLayout.addWidget(self.groupBox_5)
        self.pushButton_21.clicked.connect(self.cifar10Imgs)
        self.pushButton_22.clicked.connect(self.printParameters)
        self.pushButton_23.clicked.connect(self.printSummary)
        self.pushButton_24.clicked.connect(self.vggTrain)
        self.pushButton_25.clicked.connect(self.vggTest)
############################################################################
############################################################################
############################################################################        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1097, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
############################################################################
############################################################################
############################################################################       
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "2021 Opendl HW1"))
        self.groupBox_1.setTitle(_translate("MainWindow", "1. Image Processing"))
        self.pushButton.setText(_translate("MainWindow", "1.1 Load Image File"))
        self.pushButton_7.setText(_translate("MainWindow", "1.2 Color Separation"))
        self.pushButton_8.setText(_translate("MainWindow", "1.3 Color Transformation"))
        self.pushButton_9.setText(_translate("MainWindow", "1.4 Blending"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2. Image Smoothing"))
        self.pushButton_10.setText(_translate("MainWindow", "2.1 Gaussian blur"))
        self.pushButton_11.setText(_translate("MainWindow", "2.2 Bilateral filter"))
        self.pushButton_12.setText(_translate("MainWindow", "2.3 Median filter"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3. Edge Detection"))
        self.pushButton_13.setText(_translate("MainWindow", "3.1 Gaussian Blur"))
        self.pushButton_14.setText(_translate("MainWindow", "3.2 Sobel X"))
        self.pushButton_15.setText(_translate("MainWindow", "3.3 Sobel Y"))
        self.pushButton_16.setText(_translate("MainWindow", "3.4 Magnitude"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4. Transforms"))
        self.pushButton_17.setText(_translate("MainWindow", "4.1 Resize"))
        self.pushButton_18.setText(_translate("MainWindow", "4.2 Translation"))
        self.pushButton_19.setText(_translate("MainWindow", "4.3 Rotation, Scaling"))
        self.pushButton_20.setText(_translate("MainWindow", "4.4 Shearing"))
        self.groupBox_5.setTitle(_translate("MainWindow", "5. VGG16 Test"))
        self.pushButton_21.setText(_translate("MainWindow", "5.1 Show Training Images"))
        self.pushButton_22.setText(_translate("MainWindow", "5.2 Show Hyperparameters"))
        self.pushButton_23.setText(_translate("MainWindow", "5.3 Show Model Structure"))
        self.pushButton_24.setText(_translate("MainWindow", "5.4 Show Accuracy and Loss"))
        self.pushButton_25.setText(_translate("MainWindow", "5.5 Test"))        
############################################################################
#########################   group 1 functions   ############################
############################################################################   
    def showImg(self):
        image = cv2.imread('Image/Q1_Image/Sun.jpg')
        height, width, channels = image.shape
        cv2.imshow('HW1-1', image)
        print("Height: ", height)
        print("Width: ", width)
        cv2.waitKey(0)
        cv2.destroyAllWindows()        
    
    def separationBGR(self):
        image = cv2.imread('Image/Q1_Image/Sun.jpg')
        colorBlue,colorGreen,colorRed = cv2.split(image)
        zeros = np.zeros(image.shape[:2], dtype = "uint8")
        mergedRed = cv2.merge([zeros,zeros,colorRed])
        mergedGreen = cv2.merge([zeros,colorGreen,zeros])
        mergedBlue = cv2.merge([colorBlue,zeros,zeros])
        cv2.imshow("Red Channel: ", mergedRed)
        cv2.imshow("Green Channel: ", mergedGreen)
        cv2.imshow("Blue Channel: ", mergedBlue)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def colorTransformation(self):
        image = cv2.imread('Image/Q1_Image/Sun.jpg')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        average_gray = np.zeros(image.shape[:2], dtype = "uint8")
        colorBlue,colorGreen,colorRed = cv2.split(image)

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                average = (int(colorRed[x,y]) + int(colorGreen[x,y]) + int(colorBlue[x,y])) / 3
                average_gray[x,y] = average
        
        cv2.imshow("Grayscale ", gray_image)
        cv2.imshow("Average weighted ", average_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
  
    def blendImgs(self):
        cv2.namedWindow('Blend')
        cv2.createTrackbar('Blend', 'Blend' , 0, 255, self.onTrackbar)
        self.onTrackbar(0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def onTrackbar(self,value):
        src1 = cv2.imread('Image/Q1_Image/Dog_Strong.jpg')
        src2 = cv2.imread('Image/Q1_Image/Dog_Weak.jpg')
        value = cv2.getTrackbarPos('Blend','Blend')
        alpha = (value / 255)
        beta = ( 1.0 - alpha )
        dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
        cv2.imshow('Blend', dst)
############################################################################
#########################   group 2 functions   ############################
############################################################################    
    def gaussianSmoothing(self):
        image = cv2.imread('Image/Q2_Image/Lenna_whiteNoise.jpg')
        blur_image = cv2.GaussianBlur(image, (5, 5), 0)
        cv2.imshow('Gaussian Blur', blur_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def bilateralSmoothing(self):
        image = cv2.imread('Image/Q2_Image/Lenna_whiteNoise.jpg')
        bilateral_image = cv2.bilateralFilter(image,9,90,90)
        cv2.imshow('Bilateral Filter', bilateral_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def medianSmoothing(self):
        image = cv2.imread('Image/Q2_Image/Lenna_pepperSalt.jpg')
        bilateral_3x3 = cv2.medianBlur(image, 3)
        bilateral_5x5 = cv2.medianBlur(image, 5)
        cv2.imshow('Median Filter 3x3', bilateral_3x3)
        cv2.imshow('Median Filter 5x5', bilateral_5x5)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
############################################################################
#########################   group 3 functions   ############################
############################################################################
    def create_gaussian_kernel(self, size, sigma = 1.0):
        
        #We create a gaussian kernel of dim : size*size
        #size must be a odd number   
        if(size%2 == 0):
            raise ValueError("Size must be an odd number")
        
        x, y = np.meshgrid(np.linspace(-2, 2, size, dtype=np.float32), np.linspace(-2, 2, size, dtype=np.float32))    
        rv = (np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))) / (2.0 * np.pi * sigma ** 2)    
        rv = rv / np.sum(rv)
        
        return rv
    
    # def convolve_pixel(self, image, kernel, i, j):
        
    #     # convolves the image kernel of pixel at (i,j). returns the original pixel
    #     # if the kernel extends beyond the borders of the image   
    #     if(len(image.shape) != 2):
    #         raise ValueError("Input image should be single chanelled")
    #     if(len(kernel.shape) != 2):
    #         raise ValueError("kernel should be two dimensional")
        
    #     k = kernel.shape[0]//2
        
    #     # checking whether kernel is beyong the border
    #     if i < k or j < k or i >= image.shape[0]-k or j >= image.shape[1]-k:
    #         return image[i, j]    
        
    #     else:
    #         img_value = 0
    #         for u in range(-k, k + 1):
    #             for v in range(-k, k + 1):
    #                 img_value += image[i-u][j-v] * kernel[k+u][k+v]        
    #         return img_value
        
    # def convolve_img(self, image, kernel):
        
    #     # returns the convoluted image in a new variable
    #     # kenel should have odd dimensions and image should be single chanelled and a two dim ndarray
        
    #     # make a copy of the original image in which we will return the result
    #     new_img = np.array(image)
        
    #     for i in range(0,image.shape[0]):
    #         for j in range(0,image.shape[1]):
    #             new_img[i][j] = self.convolve_pixel(image, kernel, i, j)
                                
    #     return new_img
    
    def read_img(self):
        img = Image.open('Image/Q3_Image/House.jpg')
        image = np.array(img)
        return image
    
    def to_gray_img(self, image):     
        R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
        gray_img = 0.2989 * R + 0.5870 * G + 0.1140 * B        
        return gray_img
    
    def show_img(self, img):
        img_show = Image.fromarray(img)
        img_show.show()
        
    def showblurImg(self): 
        image = self.read_img()        
        gray = self.to_gray_img(image).astype(np.uint8)
        kernel = self.create_gaussian_kernel(3)
        blur = self.conv2d(gray, kernel, padding="same")
        # self.show_img(gray)
        # self.show_img(blur)
        gray = Image.fromarray(gray)
        blur = Image.fromarray(blur)
        gray.show()
        blur.show()
############################################################################
    def conv2d(self, img, fil, strides=1, padding="valid"):
        filter_size = fil.shape[0]
        conv_img = np.zeros((img.shape[0], img.shape[1]))
        
        # padding needed to get the same size output of original img
        if padding.lower() == "same":
            p = math.floor((filter_size - 1) / 2)
            img = np.pad(img, p)
            
    
        img_height = img.shape[0]
        img_weight = img.shape[1]
        
        # chunk init img[0:3, 0:3] img[row_start:row_end, col_start:col_end]
        row_start = list(range(0, img_height - filter_size + 1, strides)) # all chunk starts  
        row_end = list(range(filter_size, img_height + 1, strides)) # all chunk ends
        col_start = list(range(0, img_weight - filter_size + 1, strides)) # all chunk starts  
        col_end = list(range(filter_size, img_weight + 1, strides)) # all chunk ends
        
    
        for row in range(len(row_start)):
            for col in range(len(col_start)):
    
                # get chunk that is the shape of the filter
                # img[0:3, 0:3]
                chunk = img[row_start[row]:row_end[row], col_start[col]:col_end[col]]
    
                # element-wise multiplication and summation
                conv_img[row, col] = np.sum(chunk * fil)
        print('row %d, col %d, out %d' % (row, col, conv_img[row, col]))
        return conv_img
    
    # def to_gauss_img(self):
    #     image = self.read_img()       
    #     gray = self.to_gray_img(image).astype(np.uint8)
    #     kernel = self.create_gaussian_kernel(3)
    #     gauss = self.conv2d(gray, kernel, padding="same")
        
    #     return gauss
    
    def sobel_img(self, fil):
        # blur = self.to_gauss_img()
        image = self.read_img()       
        gray = self.to_gray_img(image).astype(np.uint8)
        kernel = self.create_gaussian_kernel(3)
        blur = self.conv2d(gray, kernel, padding="same")
        
        sobel = self.conv2d(blur, fil, padding="same")        
        return sobel
    
    def sobelXImg(self):           
        sobelX_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        image = self.read_img()       
        gray = self.to_gray_img(image).astype(np.uint8)
        kernel = self.create_gaussian_kernel(3)
        blur = self.conv2d(gray, kernel, padding="same")
        
        # Sobel x
        img_sobelX = self.conv2d(blur, sobelX_filter, padding="same")
        # self.show_img(img_sobelX)
        gradX = Image.fromarray(img_sobelX)
        gradX.show()
        
    def sobelYImg(self):       
        sobelY_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        image = self.read_img()       
        gray = self.to_gray_img(image).astype(np.uint8)
        kernel = self.create_gaussian_kernel(3)
        blur = self.conv2d(gray, kernel, padding="same")
        
        # Sobel y
        img_sobelY = self.conv2d(blur, sobelY_filter, padding="same")
        # self.show_img(img_sobelY)
        gradY = Image.fromarray(img_sobelY)
        gradY.show()
        
    def magnitudeImg(self):
        sobelX_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobelY_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        img_sobelX = self.sobel_img(sobelX_filter)
        img_sobelY = self.sobel_img(sobelY_filter)
        m_img = np.sqrt(img_sobelX**2 + img_sobelY**2)
        # self.show_img(m_img)
        magnitudeImage = Image.fromarray(m_img)
        magnitudeImage.show()       
############################################################################
#########################   group 4 functions   ############################
############################################################################
    def resizeImg(self):
        image = cv2.imread('Image/Q4_Image/SQUARE-01.png')
        image_resize = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imshow('Image Resize', image_resize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def translateImg(self):
        image = cv2.imread('Image/Q4_Image/SQUARE-01.png')
        image_resize = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        
        M = np.float32([[1, 0, 0], [0, 1, 60]])
        image_translate = cv2.warpAffine(image_resize, M, (image.shape[1], image.shape[0]))
        cv2.imshow('Image Translation', image_translate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rotate_scaleImg(self):
        image = cv2.imread('Image/Q4_Image/SQUARE-01.png')
        image_resize = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        M = np.float32([[1, 0, 0], [0, 1, 60]])
        image_translate = cv2.warpAffine(image_resize, M, (image.shape[1], image.shape[0]))

        M = cv2.getRotationMatrix2D((128, 188), 10, 0.5)
        image_rotate_scale = cv2.warpAffine(image_translate, M, (image.shape[1], image.shape[0]))
        cv2.imshow('Image rotate and scale', image_rotate_scale)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def shearImg(self):
        image = cv2.imread('Image/Q4_Image/SQUARE-01.png')
        image_resize = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)        
        M = np.float32([[1, 0, 0], [0, 1, 60]])
        image_translate = cv2.warpAffine(image_resize, M, (image.shape[1], image.shape[0]))
        M = cv2.getRotationMatrix2D((128, 188), 10, 0.5)
        image_rotate_scale = cv2.warpAffine(image_translate, M, (image.shape[1], image.shape[0]))
        
        srcPoints = np.float32([[50,50],[200,50],[50,200]])
        dstPoints = np.float32([[10,100],[200,50],[100,250]])
        M = cv2.getAffineTransform(srcPoints, dstPoints)
        dst = cv2.warpAffine(image_rotate_scale,M,(image.shape[1],image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        cv2.imshow('Image Shearing', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
############################################################################
#########################   group 5 functions   ############################
############################################################################
    def initCifar10(self):
        self.batch_size = 250
        self.iter_size = 25
        
        # transform = transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        # self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        # self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)                                                 
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')                       
        self.net = torchvision.models.vgg16(pretrained=True)   #Loading VGG16 network parameters from pre training model        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr = 0.001, momentum = 0.9)
    
    def cifar10Imgs(self):
        self.initCifar10()
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)                                                        
        
        # get some random training images
        dataiter = iter(trainloader)
        images, labels = dataiter.next()
        
        plt.subplots(3,3)  
        for idx in np.arange(9):  
            image = images[idx].cpu().clone().detach().numpy()
            image = image.transpose(1,2,0)
            image = image / 2 + 0.5
            plt.subplot(3,3,idx+1)
            plt.imshow(image)  
            plt.title(self.classes[labels[idx].item()])
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    def printParameters(self):
        self.initCifar10()
        print('hyperparameters:')
        print('batch size: ', self.batch_size)
        print('learning rate: ', self.optimizer.param_groups[0]['lr'])
        print('optimizer: ', type(self.optimizer).__name__)
        
    def printSummary(self):
        self.initCifar10()
        summary(self.net, (3, 32, 32), self.batch_size)
        
    def vggTrain(self):
        batch_size = 250
        iter_size = 25
        
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)                                                 
                      
        net = torchvision.models.vgg16(pretrained=True)   #Loading VGG16 network parameters from pre training model        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
        
        val_correct_history = []
        train_correct_history = []
        train_loss_history = []        
        # get some random training images
        dataiter = iter(trainloader)
        images, labels = dataiter.next()
        
        for epoch in range(iter_size):  # loop over the dataset multiple times
        
            train_loss = 0
            correct = 0
            total = 0
            net.train()
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs(images), labels]
                inputs, labels = data
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()                
                train_loss += loss.item()     
                
            # print statistics loss and acc
            print('trainloader: %d, train_loss: %d' % (len(trainloader), train_loss))
            epoch_acc = 100.0 * correct / total 
            train_correct_history.append(epoch_acc)            
            epoch_loss = train_loss/len(trainloader)
            train_loss_history.append(epoch_loss)
            print('epoch: %d, epoch_acc: %d %%, epoch_loss: %f' % (epoch + 1, epoch_acc, epoch_loss))
            print('Finished Training')
        
        # ########################################################################
        # Let us look at how the network performs on the whole dataset.        
            print("Waiting for test...")
            with torch.no_grad():
                correct = 0
                total = 0
                self.net.eval()
                for data in testloader:
                    images, labels = data
                    # calculate outputs by running images through the network 
                    outputs = net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()         
                val_epoch_acc = 100.0 * correct / total 
                val_correct_history.append(val_epoch_acc)                
                print('epoch: %d, val_epoch_acc: %f %%' % (epoch + 1, val_epoch_acc))
                print(' ')
        ########################################################################
        #save trained model        
            PATH = './cifar_net_250batch_%d_epoch.pth' % (epoch + 1)
            torch.save(net.state_dict(), PATH) #save model weight
        ########################################################################    
            
        # summarize history for accuracy
        plt.subplot(211)
        plt.plot(train_correct_history)
        plt.plot(val_correct_history)
        plt.title('Accuracy')
        plt.ylabel('%')
        plt.xlabel('epoch')
        plt.legend(['Training', 'Testing'], loc='lower right')
        
        # summarize history for loss
        plt.subplot(212)
        plt.plot(train_loss_history)
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.tight_layout()
        plt.show()
        
    def vggTest(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)        
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
        
        # input_num = int(input("Enter a image: "))
        input_num = self.lineEdit.text()
        if input_num:
            input_num = int(input_num)
            idx = input_num - 1
        else:
            input_num = 1
            idx = 0
        net = torchvision.models.vgg16()   #Loading VGG16 network parameters from pre training model
        PATH = './cifar_net_250batch_25_epoch.pth'
        net.load_state_dict(torch.load(PATH))
        
        test_images = torch.utils.data.DataLoader(testset, batch_size=input_num, shuffle=False)
        # get some random training images
        dataiter = iter(test_images)
        images, labels = dataiter.next()        
        
        image = images[idx].cpu().clone().detach().numpy()
        image = image.transpose(1,2,0)
        image = image / 2 + 0.5        
        
        test = Variable(images)
        output = net(test)
        predictions = nnf.softmax(output[idx:], dim=1)
        classes_val = predictions[0,:10] * 100
        classes_np = classes_val.cpu().detach().numpy()        
        
        plot1 = plt.figure(1)
        plt.imshow(image)
        
        plot2 = plt.figure(2)
        x = np.arange(len(classes))
        plt.bar(x, classes_np)
        plt.xticks(x, classes)
        plt.xlabel('x-axis classes')
        plt.ylabel('y-axis %')        
        plt.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

