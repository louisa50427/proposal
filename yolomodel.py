import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from keras.utils import np_utils,plot_model,print_summary
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

#此為單獨使用形態辨識當作特徵的模型
class Dataset:
	def __init__(self, path_name):
		#初始化
		self.train_images = None
		self.train_labels = None
		self.train_box = None
         
		#路徑
		self.path_name    = path_name
         
		#维度
		self.input_shape = None
         
	#載入資料集
	def load(self, img_rows = 56, img_cols = 116, 
		img_channels = 3, nb_classes = 2):
		#載入資料集到內存
		images, labels, boxlist = load_dataset(self.path_name)

		train_images = images
		train_labels = labels   
		train_box = boxlist

		train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
		self.input_shape = (img_rows, img_cols, img_channels)            
             
		#box轉換為陣列
		train_box = np.array(train_box)
		                        
		self.train_images = train_images
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

		#形態特徵輸入
		box = Input(shape = (8,),name="box")

		main_output = Dense(nb_classes,activation= 'softmax')(box)
		
		
		self.model = Model(inputs = box,outputs = main_output)
		self.model.summary()
		
		

	
	#訓練模型函數
	def train(self, dataset, batch_size = 8, epochs = 200):        
		#優化器參數設定
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
		dataset.train_box = dataset.train_box.astype('float32')
		x = 0
		num = 0 #iteration number
		
		#執行k-fold
		for train, test in kfold.split(dataset.train_images,dataset.train_labels):

			self.build_model()
			num = num + 1
			print("The %d run:"%(num))#iteration number
			if x == 0:                        
				dataset.train_labels = np_utils.to_categorical(dataset.train_labels, 2)
				x = 2
			#編譯模型
			self.model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['categorical_accuracy']) 
			#執行模型
			self.model.fit(dataset.train_box[train],dataset.train_labels[train],batch_size = batch_size,epochs = epochs,shuffle = True)
			#衡量模型
			loss, accuracy = self.model.evaluate(dataset.train_box[test], dataset.train_labels[test], verbose=1)
			print("%s: %.2f%%" % (self.model.metrics_names[1], accuracy*100))
			y_true = [np.argmax(label) for label in dataset.train_labels[test]]
			#因原本label的格式為one-hot標籤，範例:image = [image1,image2,image3],labels=[[0,1],[1,0],[0,1]]
			#y_true為將標籤類型轉換為labels = [1,0,1],rise為正樣本
			y_preds = self.model.predict(dataset.train_box[test])
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




if __name__ == '__main__':
	#控制GPU使用量
	gpu_options = tf.GPUOptions(allow_growth=True)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	tf.keras.backend.set_session(sess)
	
	#載入資料集，"data/"為同資料夾下的路徑
	dataset = Dataset("data/")    
	dataset.load()
	#建立模型   
	model = makeModel()
	
	#呼叫訓練模型
	model.train(dataset)
	

