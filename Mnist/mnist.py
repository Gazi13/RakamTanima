import numpy as np
from keras.models import load_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt


"""Resimlerin boyutu 28*28=784"""
(xe,ye),(xt,yt) = mnist.load_data()
"""xe eğitimde kullanılan resimler
   ye eğitimde kullanılan resimlerin doğru sonuçları
   xt kullanılmamış resimler
   yt kullanılmamış resimlerin doğru sonuçları """
xe = xe.reshape(60000,784)
xt = xt.reshape(10000,784)

xe = xe.astype("float32")
xt = xt.astype("float32")
"""0-1 gibi renk elde etmek için """
xe = xe/255
xt = xt/255
"""Y'lerin içinde 1,4,2 gibi normal sayılar var 01 0001 01 haline getirmek gerekiyor
 Softmax kullanmak için bunu da böyle yapıyoruz  
 10 tane sayı var sette 1-2-3-4-5-6-7-8-9-10 onun için 10 yazdık """
ye = to_categorical(ye, 10)
yt = to_categorical(yt, 10)

#Yapay sinir ağı tipi Sequential (ben)
model = Sequential()
model.add(Dense(512,input_dim=784,activation="relu"))#ilk input katmanı
model.add(Dense(256,activation="relu"))#hidden 1
model.add(Dense(128,activation="relu"))#hidden 2
model.add(Dense(10,activation="softmax"))#son katman softmax seçimi ve 10 sayı için 10 çıkış var

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(xe,ye,batch_size=20,epochs=5,validation_data=(xt,yt))
model.save("mnistSaved.h5")






