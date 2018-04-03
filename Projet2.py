from keras.preprocessing.image import ImageDataGenerator #Utilisé pour faire des transformations des images
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
#Pour évaluer le réseau
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.utils import plot_model
import numpy as np
import math
import pickle
import string
tailleImage =50
nbImage=100
NB_epoch=5

datagen = ImageDataGenerator(  
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

Polices=np.zeros((nbImage*26,tailleImage,tailleImage,3))
SortiesAttendues=np.zeros((nbImage*26,36)) #dimension nombre de polices * 36
i=0
alphabet = list(string.ascii_lowercase)
while i<nbImage:
    output=np.empty(36)
    #Lire les données du .txt pour récupérer les Paramètres
    print("numéro police: ",i+1)
    name="0"*(9-len(str(i+1)))+str(i+1)+".txt"
    with open("/Users/max/Documents/deepfont/ReTypographe_Sources/Ubuntu_FontsToPNGs/Fonts_examples/"+name,"r") as file:
        texte=file.read()
        texte=texte[0:-1]
        ligne = texte.split(",") # on utilise ':' comme séparateur
        cpt=0
        for elem in ligne:
            temp = elem.split(":")
            output[cpt]=temp[1]
            cpt=cpt+1
    upt=0
    for lettre in alphabet:
        img = load_img('/Users/max/Documents/deepfont/ReTypographe_Sources/Ubuntu_FontsToPNGs/Lettre/'+'police'+str(i+1)+'-'+lettre+'.png')
        #On met l'image dans un tableau numpy pour faciliter la manipulation
        Polices[i*26+upt]=img_to_array(img)
        upt=upt+1
    u=0
    while u<26:#Pour chaque lettre de la police les paramètres attendus en sortie sont les même
        SortiesAttendues[i*26+u]=output
        u=u+1
    i=i+1
file.close()
"""
e=0
for o in Polices:
    for t in o:
        for elem in t:
            if elem.count_non_zero()!=0:
                print(elem)
                e=1
            else:
                break
                """
print("Entrée: ",Polices.shape)
print("Sortie: ",SortiesAttendues.shape)
print("Une image ",Polices[2].shape)
"""
##########################
#On génère les images aléatoirement modifiées pour éviter le surapprentissage
i = 0  
for batch in datagen.flow(x, batch_size=1,  
                          save_to_dir='preview', save_prefix='cat', save_format='png'):
    i += 1
    if i > 20:
        break 
#Les images sont save dans preview
##########################
"""
############ CREATION ##########
def My_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3),
                            border_mode='valid',
                            input_shape=(tailleImage, tailleImage, 3) ) )
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(36))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
############ END CREATION ##########

# fix random seed for reproducibility
model = My_model()
######### ATTENTION ######
#ne pas mettre cette ligne pour la première création du fichier logique car il n'existe pas
#donc tu ne peux pas le load :) 
model.load_weights('my_model_weights.h5')
######### Fin de ATTENTION ####
plot_model(model, to_file='model.png')
hist= model.fit(Polices, SortiesAttendues, nb_epoch=NB_epoch, batch_size=16)
score = model.evaluate(Polices, SortiesAttendues, batch_size=16)
#On save les poids du model entrainé
model.save_weights('my_model_weights.h5')

#print('mse=%f, mae=%f, mape=%f' % (score[0],score[1],score[2]))
print(hist.history)
##On charge le fichier des loss et on écrit dedans

with open("/Users/max/Documents/TestResNeur/loss_file","rb") as file:
        mon_depickler=pickle.Unpickler(file)
        Tab_loss=mon_depickler.load()
        print("Avant écriture: ",Tab_loss)
        Tab_loss=Tab_loss+hist.history['loss']# On ajoute les nouveaux loss à la liste d'avant
        print("Après écriture: ",Tab_loss)
        file.close()
####PREMIERE Creation###
#ne pas faire le bloque with open juste au dessus :)
#Tab_loss = hist.history['loss']
########################
with open("/Users/max/Documents/TestResNeur/loss_file","wb") as file:
        mon_pickler=pickle.Pickler(file)
        mon_pickler.dump(Tab_loss) #On ajoute Tab_loss au document
        file.close()
abscisse=np.arange(len(Tab_loss))
plt.plot(abscisse,Tab_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.axis([0,len(Tab_loss)-1,min(Tab_loss),max(Tab_loss)])
plt.show()