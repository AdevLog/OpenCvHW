# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HW2_1.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import numpy as np
import cv2 as cv
import glob
from PyQt5 import QtCore, QtGui, QtWidgets
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((11*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.


class Ui_MainWindow(object):
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    intrinsicMatrix = []
    distortion = []
    rvecs = []
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(313, 345)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setObjectName("groupBox_2")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit.setGeometry(QtCore.QRect(123, 32, 131, 24))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 70, 231, 26))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(22, 32, 95, 18))
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.groupBox_2)
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout.addWidget(self.pushButton_4)
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout.addWidget(self.pushButton_5)
        self.horizontalLayout.addWidget(self.groupBox)
############################################################################
############################################################################
############################################################################        
        self.pushButton.clicked.connect(self.findCorners)
        self.pushButton_2.clicked.connect(self.findIntrinsicMatrix)
        self.pushButton_3.clicked.connect(self.findExtrinsicMatrix)
        self.pushButton_4.clicked.connect(self.findDistortion)
        self.pushButton_5.clicked.connect(self.showUndistort)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 313, 21))
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
        self.groupBox.setTitle(_translate("MainWindow", "2. Calibration"))
        self.pushButton.setText(_translate("MainWindow", "2.1 Find Corners"))
        self.pushButton_2.setText(_translate("MainWindow", "2.2 Find Intrinsic"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2.3 Find Extrinsic"))
        self.pushButton_3.setText(_translate("MainWindow", "2.3 Find Extrinsic"))
        self.label.setText(_translate("MainWindow", "Select Image:"))
        self.pushButton_4.setText(_translate("MainWindow", "2.4 Find Distortion"))
        self.pushButton_5.setText(_translate("MainWindow", "2.5 Show Result"))    
    
    def findCorners(self):              

        images = glob.glob('Q2_Image/*.bmp')        
        for fname in images:
            img = cv.imread(fname)
            img = cv.resize(img, (800, 800), interpolation=cv.INTER_CUBIC)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (8,11), None)
            # If found, add object points, image points (after refining them)
            if ret == True:        
                Ui_MainWindow.objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                Ui_MainWindow.imgpoints.append(corners)
        
                # Draw and display the corners
                cv.drawChessboardCorners(img, (8,11), corners2, ret)
                cv.imshow('bmp', img)
                cv.waitKey(500)
        cv.destroyAllWindows()                  
        ret, Ui_MainWindow.intrinsicMatrix, Ui_MainWindow.distortion, Ui_MainWindow.rvecs, tvecs = cv.calibrateCamera(Ui_MainWindow.objpoints, Ui_MainWindow.imgpoints, gray.shape[::-1], None, None)
        
    def findIntrinsicMatrix(self):
        print("Intrinsic:")
        print(Ui_MainWindow.intrinsicMatrix)
        
    def findExtrinsicMatrix(self):
        input_num = self.lineEdit.text()
        if input_num:
            input_num = int(input_num)
            idx = input_num - 1
        else:
            input_num = 1
            idx = 0
        # Find the rotation and translation vectors.          
        # rotMat,_ = cv.Rodrigues(Ui_MainWindow.rvecs[idx])
        _, rvec, tvec, inliers = cv.solvePnPRansac(Ui_MainWindow.objpoints[idx], Ui_MainWindow.imgpoints[idx], Ui_MainWindow.intrinsicMatrix, Ui_MainWindow.distortion)
        rotMat,_ = cv.Rodrigues(rvec)
        # rotMat = np.append(rotMat, tvecs[0], axis=1)
        rotMat = np.c_[ rotMat, tvec ]
        print("Extrinsic:")
        print(rotMat)
        
    def findDistortion(self):
        print("Distortion:")
        print(Ui_MainWindow.distortion)
        
    def showUndistort(self):
        images = glob.glob('Q2_Image/*.bmp')
        for fname in images:
            img = cv.imread(fname)
            img = cv.resize(img, (600, 600), interpolation=cv.INTER_CUBIC)

            h,  w = img.shape[:2]
            newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(Ui_MainWindow.intrinsicMatrix, Ui_MainWindow.distortion, (w,h), 1, (w,h))
        
            # Undistort
            dst = cv.undistort(img, Ui_MainWindow.intrinsicMatrix, Ui_MainWindow.distortion, None, newCameraMatrix)
        
            # crop the image
            # x, y, w, h = roi
            # dst = dst[y:y+h, x:x+w]
            
            # concatanate image Horizontally
            Hori = np.concatenate((img, dst), axis=1)
            cv.imshow('Distorted and Undistorted image', Hori)
            # cv.imshow('Distorted image', img)
            # cv.imshow('Undistorted image', dst)
            cv.waitKey(500)
                        
        cv.destroyAllWindows()   

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

