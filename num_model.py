import random
import numpy as np
import xlrd
from openpyxl import Workbook
from openpyxl.reader.excel import load_workbook
from datetime import date,datetime
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.models import load_model,Model
from keras import backend as K
from load_dataset import load_dataset
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input,normalization,BatchNormalization,concatenate
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold,cross_val_score
from PIL import Image
from cv2 import cv2 as cv 
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from data_loading import data_loading



#此為3D-CNN模型(不包含形態辨識)
class Dataset:
	def __init__(self, path_name):
		#初始化
		self.train_images = None
		self.cutImage3 = None
		self.train_box = None
		self.cutImage2 = None
		self.train_labels = None
         
		#路徑
		self.path_name    = path_name
         
		#维度
		self.input_shape = None
         
	#載入資料集
	def load(self, img_rows = 70, img_cols = 11, 
		img_channels = 1, nb_classes = 2):

		#載入資料集到內存
		
		array_list,images,label,boxlist = data_loading(self.path_name)
		train_array = array_list
		train_labels = label
		train_images = images
		train_box = boxlist
		cutImage2 = array_list
		cutImage3 = array_list
		train_images = np.array(images)
		train_array = np.array(train_array)
		cutImage2 = np.array(array_list)	
		cutImage3 = np.array(array_list)
		train_box = np.array(train_box)

		train_images = train_images.reshape(train_images.shape[0], 56, 116, 3)
		train_array = train_array.reshape(train_array.shape[0], 70, 11, 1)
		cutImage2 = cutImage2.reshape(train_array.shape[0], 70, 11, 1)
		cutImage3 = cutImage3.reshape(train_array.shape[0], 70, 11, 1)
		print("array:",train_array.shape)
		print("image:",train_images.shape)
		print("cutImage2:",cutImage2.shape)
		print("cutImage3:",cutImage3.shape)

		self.train_array = train_array			
		self.train_images = train_images
		self.cutImage2 = cutImage2
		self.cutImage3 = cutImage3
		self.train_labels = train_labels
		self.train_box = train_box


		
class makeModel:
    
	MODEL_PATH = './model.h5'
	def __init__(self):
		self.model = None 
         
	def save_model(self, file_path = MODEL_PATH):
		self.model.save(file_path)
 
	def load_model(self, file_path = MODEL_PATH):
		self.model = load_model(file_path)   

	#建立模型，nb_classes為上漲、下跌
	def build_model(self, nb_classes = 2):

		
		data_input = Input(shape= (70,11,1),name="data_input")


		#卷積層
		data_hidden = Conv2D(filters=5, kernel_size=(3,3), activation='relu', padding='same')(data_input)
		data_hidden = Conv2D(filters=15, kernel_size=(3,3), activation='relu', padding='same')(data_hidden)
		data_hidden = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(data_hidden)
		data_hidden = MaxPool2D((2,2),padding='valid')(data_hidden)
		data_hidden = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='valid')(data_hidden)

		#平坦層
		data_hidden = Flatten()(data_hidden)


		#輸入I3、I2、I1
		main_inputI3 = Input(shape= (56,116,3),name="main_inputI3")
		inputI2 = Input(shape= (56,116,3),name="inputI2")
		inputI1 = Input(shape= (56,116,3),name="inputI1")

		#形態特徵輸入
		# box = Input(shape = (8,),name="box")

		#卷積層
		hidden = Conv2D(filters=5, kernel_size=(3,3),activation='relu', padding='same')(inputI1)
		hidden = BatchNormalization()(hidden)
		hidden = Conv2D(filters=5, kernel_size=(3,3),activation='relu', padding='same')(hidden)
		hidden = BatchNormalization()(hidden)
    
		#I3/I2透過通道數(channel)連接
		outputI3wI2 = concatenate([hidden,inputI2],axis=-1)

		hidden = Conv2D(filters=15, kernel_size=(3,3),activation='relu', padding='same')(outputI3wI2)
		hidden = BatchNormalization()(hidden)
		hidden = Conv2D(filters=15, kernel_size=(3,3), activation='relu', padding='same')(hidden)
		hidden = BatchNormalization()(hidden)

		#I3/I2/I1透過通道數(channel)連接
		outputI3toI1 = concatenate([hidden,main_inputI3],axis=-1)

		hidden = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(outputI3toI1)
		hidden = BatchNormalization()(hidden)
		hidden = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(hidden)
		hidden = BatchNormalization()(hidden)

		#inception網路
		inception1 = Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='valid',strides=2)(hidden)
		inception1 = BatchNormalization()(inception1)

		inception2 = Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='valid')(hidden)
		inception2 = BatchNormalization()(inception2)
		inception2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same',strides=2)(inception2)
		inception2 = BatchNormalization()(inception2)

		inception3 = Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='valid')(hidden)
		inception3 = BatchNormalization()(inception3)
		inception3 = Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same',strides=2)(inception3)
		inception3 = BatchNormalization()(inception3)

		inception4 = MaxPool2D((2,2),padding='same')(hidden)
		inception4 = Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='valid')(inception4)
		inception4 = BatchNormalization()(inception4)

		output = concatenate([inception1,inception2,inception3,inception4],axis=-1)

		hidden = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(output)
		hidden = BatchNormalization()(hidden)
		hidden = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(hidden)
		hidden = BatchNormalization()(hidden)

		#inception網路
		inception1 = Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='valid',strides=2)(hidden)
		inception1 = BatchNormalization()(inception1)

		inception2 = Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='valid')(hidden)
		inception2 = BatchNormalization()(inception2)
		inception2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same',strides=2)(inception2)
		inception2 = BatchNormalization()(inception2)

		inception3 = Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='valid')(hidden)
		inception3 = BatchNormalization()(inception3)
		inception3 = Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same',strides=2)(inception3)
		inception3 = BatchNormalization()(inception3)

		inception4 = MaxPool2D((2,2),padding='same')(hidden)
		inception4 = Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='valid')(inception4)
		inception4 = BatchNormalization()(inception4)

		output = concatenate([inception1,inception2,inception3,inception4],axis=-1)

		#卷積層
		hidden = Conv2D(filters=64, kernel_size=(3,3),activation='relu', padding='same')(output)
		hidden = BatchNormalization()(hidden)
		hidden = Conv2D(filters=64, kernel_size=(3,3),activation='relu', padding='same')(hidden)
		hidden = BatchNormalization()(hidden)
		hidden = Conv2D(filters=64, kernel_size=(3,3),activation='relu', padding='valid')(hidden)
		hidden = BatchNormalization()(hidden)
		hidden = Dropout(0.25)(hidden)
		#平坦層
		hidden = Flatten()(hidden)
		
		#3D-CNN特徵與YOLO形態特徵連接
		hidden = concatenate([hidden,box],axis=-1)
		hidden = concatenate([hidden,data_hidden],axis=-1)
		hidden = Dense(128,activation= 'relu')(hidden)
		hidden = Dropout(0.25)(hidden)
		main_output = Dense(2,activation= 'softmax')(hidden)
		
		self.model = Model(inputs = [main_inputI3,inputI2,inputI1,data_input],outputs = main_output)
		self.model.summary()

	
	#訓練模型函數
	def train(self, dataset, batch_size = 8, epochs = 50):        
		#優化器參數設定
		sgd = SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
		adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
		#kfold參數設定
		kfold = StratifiedKFold(n_splits=15, shuffle=True, random_state=7)#random_state = random seed

		#初始化變數
		cvscores = []#儲存accuracy
		loss_sum = []
		f1_scores = []
		recalls = []
		precisions = []
		f1_scores_decline = []
		recalls_decline = []
		precisions_decline = []
		#歸一化，將圖像的各像素值歸一化到0~1區間
		dataset.cutImage2 = cutImage2mid(dataset.cutImage2)
		dataset.cutImage3 = cutImage2right(dataset.cutImage3)

		dataset.cutImage3 = dataset.cutImage3.astype('float32')
		dataset.cutImage2 = dataset.cutImage2.astype('float32')
		dataset.cutImage2 /= 255
		dataset.cutImage3 /= 255
		dataset.train_images = dataset.train_images.astype('float32')  
		dataset.train_images /= 255
		x = 0
		num = 0
	
		#執行k-fold
		for train, test in kfold.split(dataset.train_images,dataset.train_labels):#train, test為訓練圖片與測試團片的index list

			self.build_model()
			num = num + 1#iteration number
			print("The %d run:"%(num))
			if x == 0:     
				dataset.train_labels = np.array(dataset.train_labels)                   
				dataset.train_labels = np_utils.to_categorical(dataset.train_labels, 2)
				x = 2
			#編譯模型
			self.model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['categorical_accuracy']) 
			#執行模型
			print(dataset.train_images[train].shape)
			self.model.fit([dataset.train_images[train],dataset.cutImage2[train],dataset.cutImage3[train],dataset.train_array[train]],dataset.train_labels[train],batch_size = batch_size,epochs = epochs,shuffle = True)
			#衡量模型
			loss, accuracy = self.model.evaluate([dataset.train_images[test],dataset.cutImage2[test],dataset.cutImage3[test],dataset.train_array[test]], dataset.train_labels[test], verbose=1)
			print("%s: %.2f%%" % (self.model.metrics_names[1], accuracy*100))
			y_true = [np.argmax(label) for label in dataset.train_labels[test]]
			#因原本label的格式為one-hot標籤，範例:image = [image1,image2,image3],labels=[[0,1]代表上漲,[1,0]代表下跌,[0,1]]
			#y_true為將標籤類型轉換為label
			# s = [1,0,1],rise為正樣本
			y_preds = self.model.predict([dataset.train_images[test],dataset.cutImage2[test],dataset.cutImage3[test],dataset.train_array[test]])
			#y_preds為預測的結果
			y_pred = [np.argmax(y_pred)for y_pred in y_preds]
			#因原本y_pred為one-hot標籤，範例:y_pred=[[0,1],[1,0],[0,1]],y_pred為將標籤類型轉換為y_pred = [1,0,1],rise為正樣本
			

			y_true=np.array(y_true)
			y_pred = np.array(y_pred)
			y_true_decline = (y_true - 1)*(-1)#將y_true中decline轉換為正樣本
			y_pred_decline =( y_pred -1)*(-1)#將y_pred中decline轉換為正樣本

			#儲存各種衡量指標
			recall = recall_m(y_true,y_pred)
			recall_decline = recall_m(y_true_decline,y_pred_decline)
			precision = precision_m(y_true,y_pred)
			precision_decline = precision_m(y_true_decline,y_pred_decline)
			f1_score = f1_m(y_true,y_pred)
			f1_score_decline = f1_m(y_true_decline,y_pred_decline)
			cvscores.append(accuracy * 100)
			loss_sum.append(loss)
			f1_scores.append(f1_score)
			f1_scores_decline.append(f1_score_decline)
			recalls.append(recall)
			recalls_decline.append(recall_decline)
			precisions.append(precision)
			precisions_decline.append(precision_decline)

		print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
		print("%.2f%% (+/- %.2f%%)" % (np.mean(f1_scores), np.std(f1_scores)))
		print("%.2f%% (+/- %.2f%%)" % (np.mean(f1_scores_decline), np.std(f1_scores_decline)))
		print("acc")
		print(cvscores)
		print("loss")
		print(loss_sum)
		print("f1")
		print(f1_scores)
		print("recall")
		print(recalls)
		print("precision")
		print(precisions)
		print("f1_decline")
		print(f1_scores_decline)
		print("recall_decline")
		print(recalls_decline)
		print("precision_decline")
		print(precisions_decline)
        

#評估模型recall函數
def recall_m(y_true, y_pred):
	TP = np.sum(y_true * y_pred)#TP
	P=np.sum(y_true)
	FN = P-TP #FN=P-TP
	recall = TP / (TP + FN )#TP/(TP+FN)
	return recall

#評估模型precision函數
def precision_m(y_true, y_pred):
	true_positives = np.sum(y_true * y_pred)
	predicted_positives = np.sum(y_pred)
	precision = true_positives / (predicted_positives)
	return precision

#評估模型f1-score函數
def f1_m(y_true, y_pred):
	precision = precision_m(y_true, y_pred)
	recall = recall_m(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall))

#將圖片切三等分的方法中，此方法為儲存中間(中期)圖片
def cutImage2mid(images):
	for img in images:
		for w in range(40,70):
			for i in range(0,11):
				img[w][i] = 0
		
	return (images)

#將圖片切三等分的方法中，此方法為儲存短期圖片
def cutImage2right(images):
	for img in images:
		for w in range(25,70):
			for i in range(0,11):
				img[w][i] = 0	
	return (images)

if __name__ == '__main__':
	#控制GPU使用量
	gpu_options = tf.GPUOptions(allow_growth=True)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	tf.keras.backend.set_session(sess)

	#載入資料集，"data"為同資料夾下的路徑
	dataset = Dataset("data")    
	dataset.load()

    #建立模型
	model = makeModel()
	
	#呼叫訓練模型
	model.train(dataset)
	print("組合數據與圖片")

	model.save_model(file_path = 'num_model.h5')