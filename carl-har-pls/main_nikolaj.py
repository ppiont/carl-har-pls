import numpy as np
import pdb
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import ANN_functions as ANN
import tensorflow as tf

data_dict = scipy.io.loadmat('data/data.mat')

Xtest = data_dict['Xtest'] + data_dict['Ex_test']
ytest = data_dict['ytest'][:,0] + data_dict['ey_test'][:,0]

X1 = data_dict['X1']
X2 = data_dict['X2']

y1 = data_dict['y1'][:,0] + data_dict['ey'][:,0]
y2 = data_dict['y2'][:,0]


Ex = data_dict['Ex']
ey = data_dict['ey']

It = data_dict['It'][0,0]

MSE_X1 = []
MSE_X2 = []
std_X1 = []
std_X2 = []
for components in [1,2,3]:
    component_score_X1 = []
    component_score_X2 = []
    for i in range(It):
        PLSR_X1 = PLSRegression(n_components = components)
        PLSR_X2 = PLSRegression(n_components = components)
        PLSR_X1.fit(X1+Ex[:,:,i],y1+ey[:,i])
        PLSR_X2.fit(X2+Ex[:,:,i],y2+ey[:,i])

        component_score_X1.append(mean_squared_error(PLSR_X1.predict(Xtest),ytest))
        component_score_X2.append(mean_squared_error(PLSR_X2.predict(Xtest),ytest))

    MSE_X1.append(np.mean(component_score_X1))
    MSE_X2.append(np.mean(component_score_X2))
    std_X1.append(np.std(component_score_X1))
    std_X2.append(np.std(component_score_X2))

############## Create and Train Network  ##############
#L2 regularization
regu = 1e-6

#Number of epochs
num_epochs = 500

#Batch_size
batch_size = 4

ANN_MSE_X1 = []
ANN_MSE_X2 = []
ANN_std_X1 = []
ANN_std_X2 = []
for num_layers in [1,2,3]:
    layer_score_X1 = []
    layer_score_X2 = []
    for i in range(10):
        #Create an instance of you neural network model
        model = ANN.neural_net(regularization=regu,num_layers=num_layers,num_neurons=16)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        #Compile network
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        #Set up callback function. Necessary for early-stopping
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30)

        #Train network using model.fit
        history = model.fit(X1+Ex[:,:,i],y1+ey[:,i],validation_data=(Xtest, ytest),epochs=num_epochs,verbose=0)
        #history = model.fit(X1+Ex[:,:,0],y1+ey[:,0],epochs=num_epochs,verbose=1)
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        layer_score_X1.append(mean_squared_error(model.predict(Xtest), ytest))

        del model
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        print(i)

    ANN_MSE_X1.append(np.mean(layer_score_X1))
    ANN_std_X1.append(np.std(layer_score_X1))

plt.figure()
plt.errorbar([1,2,3],MSE_X1,yerr=std_X1,linewidth=3,label='X1 PLS')
plt.errorbar([1,2,3],MSE_X2,yerr=std_X2,linewidth=3,label='X2 PLS')
plt.errorbar([1,2,3],ANN_MSE_X1,yerr=ANN_std_X1,linewidth=3,label='X1 ANN')
plt.grid()
plt.legend(loc='best')
plt.xlabel('PLS Components / ANN Layers')
plt.ylabel('MSE')
plt.savefig('PLS_ANN_MSE')
plt.show()