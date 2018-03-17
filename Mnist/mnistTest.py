import numpy as np
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical


(xe,ye),(xt,yt) = mnist.load_data() # bunu tekrar yükledik çükü xt falan kullanıyoruz gene

xt = xt.reshape(10000,784)# resimleri tek satır array yaptık
#xt = xt.astype("float32") #bu da bir şeye yaramıyor burda heralde
#xt = xt/255 # Bu olmayınca daha iyi kolay anlaşılır oluyor

model = load_model("mnistSaved.h5") # eğitilmiş nn yüklüyoruz
Sonuc = model.predict(np.array([xt[160]])) #hesaplama işlemi
print(Sonuc) #hesaplanan sonuç
print(yt[160])# doğru sonuç