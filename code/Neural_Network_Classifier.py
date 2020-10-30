import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import load_data_mnist
class lode_dataset:
    def Exam():
        train_X = np.loadtxt('./Exam/train/x.txt')
        train_y = np.loadtxt('./Exam/train/y.txt')
        test_X = np.loadtxt('./Exam/test/x.txt')
        test_y = np.loadtxt('./Exam/test/y.txt')
        return train_X,train_y,test_X,test_y
    def Iris():
        train_X = np.loadtxt('./Iris/train/x.txt')
        train_y = np.loadtxt('./Iris/train/y.txt')
        test_X = np.loadtxt('./Iris/test/x.txt')
        test_y = np.loadtxt('./Iris/test/y.txt')
        return train_X, train_y, test_X, test_y
    def MNIST():
        train,test=load_data_mnist.load_mnist('./train-images.idx3-ubyte','train-labels.idx1-ubyte','t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte')
        train_X = train[0]
        train_y = train[1]
        test_X = test[0]
        test_y = test[1]
        return train_X[:64], train_y[:64], test_X[:10], test_y[:10]
def Standardize(Standard_X,X):
    std = np.std(Standard_X)#Calculate the variance of the data
    mean = np.mean(Standard_X)#Calculate the mean of the data
    return (X - mean) / std#Return the result of data standardization
"""
Use the Adam Algorithm to optimize the gradient
"""
def Adam(v,s,g,i):
    v=0.9*v+0.1*g
    s=0.999*s+0.001*g**2
    v_correct = v/ (1 - 0.9 ** i)
    s_correct = s / (1 - 0.999 ** i)
    g=v_correct/(np.sqrt(s_correct) + 10 ** (-8))
    return v,s,g
class sigmoid:
    def activate(z):
        return 1 / (1 + np.exp(-z))
    def gradient(z):
        a=1 / (1 + np.exp(-z))
        return np.multiply(a,1-a)
class tanh:
    def activate(z):
        return ((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))
    def gradient(z):
        a=((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))
        return 1-a**2
class  ReLU:
    def activate(z):
        z[z<=0]=0
        return z
    def gradient(z):
        z[z<=0]=0
        z[z>0]=1
        return z
class ELU():
    def __init__(self,alpha=2):
        self.alpha=alpha
    def activate(self,z):
        z[z<=0]=self.alpha*(np.exp(z[z<=0])-1)
        return z
    def gradient(self,z):
        z[z<=0]=self.alpha*np.exp(z[z<=0])
        z[z>0]=1
        return z
class MSE:
    def activate(Hypothesis_X, y):
        return np.sum((Hypothesis_X-y)**2)/2
    def gradient(Hypothesis_X, y):
        return Hypothesis_X-y
class CrossEntropy:
    def activate(Hypothesis_X, y):
        return -np.sum(np.multiply(y,np.log(Hypothesis_X)))
    def gradient(Hypothesis_X, y):
        return -y/(Hypothesis_X+10**(-10))
class Arbitrary_Scale_Neural_Network_for_Classification():
    def __init__(self,Layer_scale=[2,2],activation_function=[sigmoid,sigmoid],
                 learning_rate=0.01,times_interation=500,batch_size=64,feature_num=2,class_num=2,loss_function=MSE):
        self.yita=learning_rate
        self.times=times_interation
        self.batch_size=batch_size
        self.feature_num=feature_num
        self.class_num=class_num
        self.Layer_scale=Layer_scale
        self.activation_function=activation_function
        self.loss_function=loss_function
        self.W=[0]
        self.b=[0]
        np.random.seed(2)
        self.W.append(np.random.randn(self.Layer_scale[0],self.feature_num))
        self.b.append(np.random.randn(self.Layer_scale[0],1))
        for i in range(len(self.Layer_scale)-1):
            self.W.append(np.random.randn(self.Layer_scale[i+1],self.Layer_scale[i]))
            self.b.append(np.random.randn(self.Layer_scale[i+1], 1))
        self.gradient_A=[0]*(len(self.Layer_scale)+1)
        self.gradient_W = [0] * (len(self.Layer_scale)+1)
        self.gradient_b = [0] * (len(self.Layer_scale)+1)
        self.gradient_Z = [0] * (len(self.Layer_scale)+1)
        self.s_W=[0] * (len(self.Layer_scale)+1)
        self.v_W=[0] * (len(self.Layer_scale)+1)
        self.s_b=[0] * (len(self.Layer_scale)+1)
        self.v_b=[0] * (len(self.Layer_scale)+1)
    def feedforword(self,X):
        self.A=[X]
        self.Z=[0]
        for i in range(len(self.Layer_scale)):
            self.Z.append(np.dot(self.W[i+1],self.A[i])+self.b[i+1])
            self.A.append(self.activation_function[i].activate(self.Z[i+1]))
        return self.A[-1]
    def backforword(self):

        self.gradient_Z[-1]=np.multiply(self.gradient_A[-1],self.activation_function[-1].gradient(self.Z[-1]))
        self.gradient_b[-1]=np.sum(self.gradient_Z[-1],axis=1).reshape(self.Layer_scale[-1],1)
        self.gradient_W[-1]=np.dot(self.gradient_Z[-1],self.A[-2].T)
        for i in range(len(self.Layer_scale)-1):
            self.gradient_A[-2-i]=np.dot(self.W[-1-i].T,self.gradient_Z[-1-i])
            self.gradient_Z[-2-i]=np.multiply(self.gradient_A[-2-i],self.activation_function[-2-i].gradient(self.Z[-2-i]))
            self.gradient_W[-2-i]=np.dot(self.gradient_Z[-2-i],self.A[-3-i].T)
            self.gradient_b[-2-i] = np.sum(self.gradient_Z[-2-i], axis=1).reshape(self.Layer_scale[-2-i],1)
        for i in range(len(self.Layer_scale)):
            self.s_W[-1-i],self.v_W[-1-i],self.gradient_W[-1-i]=Adam(self.s_W[-1-i],self.v_W[-1-i],self.gradient_W[-1-i],i+1)
            self.s_b[-1-i],self.v_b[-1-i],self.gradient_b[-1-i]=Adam(self.s_b[-1-i],self.v_b[-1-i],self.gradient_b[-1-i],i+1)
            self.W[-1-i]-=self.yita*self.gradient_W[-1-i]
            self.b[-1-i]-=self.yita*self.gradient_b[-1-i]
    def fit(self,X,y,test_X,test_y):
        self.X = np.asarray(X)
        self.y=np.asarray(y)
        y = np.eye(len(np.unique(y)))[y.astype(int)].T
        self.loss=[]
        index = 0
        """predict_y=np.asarray([0.1]*len(self.y))"""
        self.accuracy_train=[]
        self.accuracy_test=[]
        for i in range(self.times):
            next_index=index + self.batch_size
            if next_index<=self.X.shape[0]:
                X_batch = self.X[index:next_index].reshape(-1,self.feature_num)
                y_batch=y[:, index:next_index].reshape(self.class_num,-1)
                """Hypothesis_X = self.feedforword(X_batch)
                predict_y[index:next_index] = np.argmax(Hypothesis_X, axis=0)"""
            else:
                index2=next_index - self.X.shape[0]
                X_batch=np.concatenate((self.X[index:].reshape(-1,self.feature_num),self.X[:index2].reshape(-1,self.feature_num)),axis=0)
                y_batch = np.concatenate((y[:, index:].reshape(self.class_num,-1), y[:, :index2].reshape(self.class_num,-1)),axis=1)
                """Hypothesis_X = self.feedforword(X_batch)
                predict_mini=np.argmax(Hypothesis_X, axis=0)
                predict_y[index:] = predict_mini[:self.X.shape[0]-index]
                predict_y[:index2]=predict_mini[self.X.shape[0]-index:]"""
            index = next_index % self.X.shape[0]
            train_predict_y = self.predict(self.X)
            test_predict_y = self.predict(test_X)

            batch_Hypothesis_X=self.feedforword(X_batch.T)
            self.loss.append(self.loss_function.activate(batch_Hypothesis_X, y_batch))
            self.gradient_A[-1] = self.loss_function.gradient(batch_Hypothesis_X, y_batch)
            self.accuracy_train.append(self.accuracy(train_predict_y, self.y))
            self.accuracy_test.append(self.accuracy(test_predict_y,test_y))
            self.backforword()
    """def loss_function_calculation(self,Hypothesis_X,y):
        self.MSE=np.sum((Hypothesis_X-y)**2)/2
        self.MSE_Gradient=Hypothesis_X-y
        self.CrossEntropy=-np.sum(np.multiply(y,np.log(Hypothesis_X)))
        self.CrossEntropy_Gradient=y/Hypothesis_X"""
    def predict(self,X):
        Hypothesis_X = self.feedforword(X.T)
        predict_y=np.argmax(Hypothesis_X,axis=0)
        return predict_y
    def accuracy(self,predict_y,y):
        ac=np.sum(np.where(predict_y==y,1,0))/len(y)
        return ac
aa=Arbitrary_Scale_Neural_Network_for_Classification(Layer_scale=[800,800,10],
                                                activation_function=[sigmoid,sigmoid,sigmoid],
                                                     learning_rate=0.01,times_interation=2000,batch_size=63,
                                                     feature_num=784,class_num=10,loss_function=CrossEntropy)
train_X,train_y,test_X,test_y=lode_dataset.MNIST()
train_X_st=Standardize(train_X,train_X)
test_X_st=Standardize(train_X,test_X)
aa.fit(train_X_st,train_y,test_X_st,test_y)
predict_y_train=aa.predict(train_X_st)
print("training set accuracy:",aa.accuracy_train[-1])
predict_y_test=aa.predict(test_X_st)
print("test set accuracy:",aa.accuracy_test[-1])
print(aa.loss)

"""
Visualization
"""
fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # Create canvas and axis

left_border=int(np.min([np.min(train_X[:,0]),np.min(test_X[:,0])]))
right_border=int(np.max([np.max(train_X[:,0]),np.max(test_X[:,0])])+1)
bottom_border=int(np.min([np.min(train_X[:,1]),np.min(test_X[:,1])]))
top_border=int(np.max([np.max(train_X[:,1]),np.max(test_X[:,1])])+1)

ax[0].set_xlim(left_border, right_border)  # Set the x-axis range
ax[0].set_ylim(bottom_border, top_border)  # Set the y-axis range
ax[0].scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='Dark2',s=10,marker='o',label='Training set')  # Plot the training set distribution on the first axis
ax[0].scatter(test_X[:, 0], test_X[:, 1], c=test_y, cmap='Dark2',s=10,marker='x',label='Test set')  # Plot the test set distribution on the first axis
ax[0].legend()
ax[0].set_xlabel("Feature1")  # Set x-axis label
ax[0].set_ylabel("Feature2")  # Set y-axis label
ax[0].set_title("Classification Boundaries")  # Set title
ax[1].set_xlim(0, aa.times-1)  # Set the x-axis range
ax[1].set_ylim(int(np.min(aa.loss)), int(np.max(aa.loss)+1))  # Set the y-axis range
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Cost Function")
ax[1].set_title("Cost Change")
ax[2].set_xlim(0, aa.times-1)
ax[2].set_ylim(0, 1)
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("Accuracy")
ax[2].set_title("Accuracy Change")


# feature1 = np.linspace(left_border, right_border, 400)  # Generate 400 numbers evenly between 1.5 and 5
# feature2 = np.linspace(bottom_border, top_border, 400)  ##Generate 400 numbers evenly between 0 and 3
# XX1, XX2 = np.meshgrid(feature1, feature2)  # Generate grid point coordinate matrix
# XX = np.c_[XX1.ravel(), XX2.ravel()]  # Flatten and merge XX1 and XX2 to facilitate the prediction of the classification of each point on the grid
# XX_st=Standardize(train_X,XX)
# ZZ=aa.predict(XX_st).reshape(XX1.shape)
# cont = ax[0].contourf(XX1, XX2, ZZ, alpha=0.2, cmap='Set3')  # Draw classification boundaries

line_loss, = ax[1].plot([], [])  # Plot the loss function decline on the second axis
training_accuracy = ax[2].scatter([], [], label='training set', s=10,c='red')  # Plot the changes in the prediction accuracy of the training set on the third axis
test_accuracy = ax[2].scatter([], [], label='test set',s=10)  # Plot the changes in the prediction accuracy of the test set on the third axis
ite = np.arange(aa.times)  # Iteration count array


def animate(i):  # Define animation update function
    line_loss.set_data(ite[:i], aa.loss[:i])  # Update the cost for each Iteration
    training_accuracy.set_offsets(np.stack((ite[:i], aa.accuracy_train[:i]),axis=1))  # Update the prediction accuracy of the training set for each Iteration
    test_accuracy.set_offsets(np.stack((ite[:i], aa.accuracy_test[:i]),axis=1))  # Update the prediction accuracy of the test set for each Iteration
    return  line_loss, training_accuracy, test_accuracy


ani = animation.FuncAnimation(fig, animate, frames=aa.times, interval=1)  # Generate animation
plt.legend()  # Show legend
plt.show()
#ani.save('ThreeLayers_NeuralNetwork_Iris.gif',writer='imagemagick',fps=60)#Save dynamic images (imagemagick needs to be installed)
