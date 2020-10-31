"""
任意规模的神经网络分类器
-----------------------
内置Exam和Iris两个简单数据集和MNIST数据集（因为没有实现并行加速，所以无法在短时间内训练MNIST数据集，可以选择分割数据集的一部分进行训练）
支持Adam,Momentum,Adagrad,RMSprop等优化器
支持sigmoid,tanh,ReLU,ELU激活函数
支持MSE和CrossEntropy损失函数
支持自定义梯度下降每次迭代的mini-batch大小，隐藏层和输出层的规模以及每一层的激活函数
支持绘制与保存损失函数值，准确率，分类边界随迭代次数的变化图（可通过save_contour参数选择是否在每次更新权值后保存绘制分类边界变化的参数）
支持绘制与保存损失函数值，准确率随迭代次数的变化的静态图和最终的分类边界静态图
------------------------
适用于对于机器学习算法的学习与交流
设置Layer_scale=[2],activation_function=[None_activation]，loss_function=CrossEntropy，即可实现二分类的逻辑回归
设置Layer_scale=[c>=2],activation_function=[None_activation]，loss_function=CrossEntropy，即可实现c分类的softmax回归
------------------------
Author: cdb342
Date: 2020/10/31
Time: 17:13
Versions: 2.0
Homepage: https://github.com/cdb342
"""
import numpy as np
import Neural_Network_Classifier_Visulization as vlz
import load_data_mnist
"""
数据集导入类
Exam():导入Exam数据集
Iris():导入Iris数据集
MNIST():导入MNIST数据集
"""
class load_dataset:
    def Exam():
        train_X = np.loadtxt('./datasets/Exam/train/x.txt')
        train_y = np.loadtxt('./datasets/Exam/train/y.txt')
        test_X = np.loadtxt('./datasets/Exam/test/x.txt')
        test_y = np.loadtxt('./datasets/Exam/test/y.txt')
        return train_X,train_y,test_X,test_y
    def Iris():
        train_X = np.loadtxt('./datasets/Iris/train/x.txt')
        train_y = np.loadtxt('./datasets/Iris/train/y.txt')
        test_X = np.loadtxt('./datasets/Iris/test/x.txt')
        test_y = np.loadtxt('./datasets/Iris/test/y.txt')
        return train_X, train_y, test_X, test_y
    def MNIST():
        train,test=load_data_mnist.load_mnist('./datasets/MNIST/train-images.idx3-ubyte','./datasets/MNIST/train-labels.idx1-ubyte','./datasets/MNIST/t10k-images.idx3-ubyte','./datasets/MNIST/t10k-labels.idx1-ubyte')
        train_X = train[0]
        train_y = train[1]
        test_X = test[0]
        test_y = test[1]
        return train_X[:600], train_y[:600], test_X[:100], test_y[:100]#可以选择MNIST数据集的一部分进行操作
"""
数据标准化函数
使用左边数据的均值和方差对右边的数据进行标准化处理
"""
def Standardize(Standard_X,X):
    std = np.std(Standard_X)#计算数据的方差
    mean = np.mean(Standard_X)#计算数据的均值
    return (X - mean) / std#返回数据标准化的结果
"""
Momentum优化器
"""
class Momentum():
    def __init__(self,beta=0.9):
        self.v=0
        self.beta=beta
    def optimise(self,g,i):
        self.v = np.add(np.multiply(self.beta, self.v), np.multiply(1 - self.beta, g))
        v_correct = np.divide(self.v, (1 - self.beta ** i))
        return v_correct
"""
Adagrad优化器
"""
class Adagrad():
    def __init__(self):
        self.G=0
    def optimise(self,g,i=0):
        self.G=np.add(self.G,np.power(g,2))
        return np.divide(g,np.power(np.add(self.G,10**(-8)),1/2))
"""
RMSprop优化器
"""
class RMSprop():
    def __init__(self,beta=0.999):
        self.s=0
        self.beta=beta
    def optimise(self,g,i):
        self.s = np.add(np.multiply(self.beta, self.s), np.multiply(1 - self.beta, np.power(g, 2)))
        s_correct = np.divide(self.s, (1 - self.beta ** i))
        return np.divide(g,np.add(np.power(s_correct,1/2) , 10 ** (-8)))
"""
Adam优化器
"""
class Adam():
    def __init__(self,beta1=0.9,beta2=0.999):
        self.v=0
        self.s=0
        self.beta1=0.9
        self.beta2=0.999
    def optimise(self,g,i):
        self.v=np.add(np.multiply(self.beta1,self.v),np.multiply(1-self.beta1,g))
        self.s=np.add(np.multiply(self.beta2,self.s),np.multiply(1-self.beta2,np.power(g,2)))
        v_correct =np.divide( self.v, (1 - self.beta1 ** i))
        s_correct = np.divide(self.s , (1 - self.beta2 ** i))
        g=np.divide(v_correct,np.add(np.power(s_correct,1/2) , 10 ** (-8)))
        return g
"""
sigmoid类
activate:返回用sigmoid函数激活后的结果
gradient:返回sigmoid函数的导数
"""
class sigmoid:
    def activate(z):
        a=np.empty_like(z)
        a[z>=0]=1 / (1 + np.exp(-z[z>=0]))
        a[z<0]=np.exp(z[z<0])/(1+np.exp(z[z<0]))
        return a
    def gradient(z):
        a = np.empty_like(z)
        a[z >= 0] = 1 / (1 + np.exp(-z[z >= 0]))
        a[z < 0] = np.exp(z[z < 0]) / (1 + np.exp(z[z < 0]))
        return np.multiply(a,1-a)
"""
tanh类
activate:返回用tanh函数激活后的结果
gradient:返回tanh函数的导数
"""
class tanh:
    def activate(z):
        return ((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))
    def gradient(z):
        a=((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))
        return 1-a**2
"""
ReLU类
activate:返回用ReLU函数激活后的结果
gradient:返回ReLU函数的导数
"""
class ReLU:
    def activate(z):
        z[z<=0]=0
        return z
    def gradient(z):
        z[z<=0]=0
        z[z>0]=1
        return z
"""
ELU类
activate:返回用ELU函数激活后的结果
gradient:返回ELU函数的导数
"""
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
class None_activation:
    def activate(z):
        return z
    def gradient(z):
        return np.ones(z.shape)
"""
MSE类
calculate:返回均方根误差
gradient:返回均方根误差的导数
"""
class MSE:
    def calculate(Hypothesis_X, y):
        return np.sum((Hypothesis_X-y)**2)/2
    def gradient(Hypothesis_X, y):
        return Hypothesis_X-y
"""
CrossEntropy类
activate:返回交叉熵误差
gradient:返回交叉熵误差的导数
"""
class CrossEntropy:
    def calculate(Hypothesis_X, y):
        z=np.exp(Hypothesis_X) / np.sum(np.exp(Hypothesis_X), axis=0).reshape((1, -1))
        return -np.sum(np.multiply(np.log(z),y))
    def gradient(Hypothesis_X, y):
        z = np.exp(Hypothesis_X) / np.sum(np.exp(Hypothesis_X), axis=0).reshape((1, -1))
        return z-y
"""
神经网络主体
__init__:输入神经网络规模、每层的激活函数、学习率、迭代次数、batch大小、输入特征数、类别数、损失函数、优化器进行初始化
feedback:实现前向传播
backforword:反向传播更新梯度
fit:训练权值并储存每次迭代的损失函数和准确率
predict:预测分类结果
accuracy:计算准确率
"""
class Arbitrary_Scale_Neural_Network_for_Classification():
    def __init__(self,Layer_scale,activation_function,learning_rate=0.01,times_interation=500,
                 batch_size=64,feature_num=2,class_num=2,loss_function=MSE,optimizer=Adam,save_contour=False):
        self.Layer_scale = Layer_scale  # 隐藏层和输出层的规模
        self.activation_function = activation_function  # 每层的激活函数
        self.yita=learning_rate#学习率，默认为0.01
        self.times=times_interation#迭代次数，默认为500次
        self.batch_size=batch_size#batch大小，默认为64
        self.feature_num=feature_num#输入特征数，默认为2
        self.class_num=class_num#类别数，默认为2
        self.loss_function=loss_function#使用的损失函数，默认使用MSE
        self.optimizer_W=optimizer()#优化W的梯度的优化器
        self.optimizer_b=optimizer()#优化b的梯度的优化器
        self.save_contour=save_contour#是否在每次更新权值后生成并保存每个网格点的预测值
        """
        初始化权值和梯度矩阵
        为了使参数矩阵的索引与推导时的上标一致，在索引0的位置多添加了一个数，实际计算时不会用到
        """
        self.W=[0]
        self.b=[0]
        self.W.append(np.random.randn(self.Layer_scale[0],self.feature_num))
        self.b.append(np.random.randn(self.Layer_scale[0],1))
        for i in range(len(self.Layer_scale)-1):
            self.W.append(np.random.randn(self.Layer_scale[i+1],self.Layer_scale[i]))
            self.b.append(np.random.randn(self.Layer_scale[i+1], 1))
        self.gradient_A=[0]*(len(self.Layer_scale)+1)
        self.gradient_W = [0] * (len(self.Layer_scale)+1)
        self.gradient_b = [0] * (len(self.Layer_scale)+1)
        self.gradient_Z = [0] * (len(self.Layer_scale)+1)
    def feedforword(self,X):
        self.A=[X]
        self.Z=[0]
        for i in range(len(self.Layer_scale)):
            self.Z.append(np.dot(self.W[i+1],self.A[i])+self.b[i+1])#第i+1层未激活的值
            self.A.append(self.activation_function[i].activate(self.Z[i+1]))#第i+1层激活后的值
        return self.A[-1]#返回输出层
    def backforword(self,epo):
        """
        使用反向传播算法计算每一层的梯度
        """
        self.gradient_Z[-1]=np.multiply(self.gradient_A[-1],self.activation_function[-1].gradient(self.Z[-1]))
        self.gradient_b[-1]=np.sum(self.gradient_Z[-1],axis=1).reshape(self.Layer_scale[-1],1)
        self.gradient_W[-1]=np.dot(self.gradient_Z[-1],self.A[-2].T)
        for i in range(len(self.Layer_scale)-1):
            self.gradient_A[-2-i]=np.dot(self.W[-1-i].T,self.gradient_Z[-1-i])
            self.gradient_Z[-2-i]=np.multiply(self.gradient_A[-2-i],self.activation_function[-2-i].gradient(self.Z[-2-i]))
            self.gradient_W[-2-i]=np.dot(self.gradient_Z[-2-i],self.A[-3-i].T)
            self.gradient_b[-2-i] = np.sum(self.gradient_Z[-2-i], axis=1).reshape(self.Layer_scale[-2-i],1)
        """
        使用Adam算法对梯度进行优化并更新权值
        """
        self.gradient_W=self.optimizer_W.optimise(self.gradient_W, epo + 1)
        self.gradient_b=self.optimizer_b.optimise(self.gradient_b,epo+1)
        self.W -= np.multiply(self.yita , self.gradient_W)
        self.b -= np.multiply(self.yita ,self.gradient_b)
    def fit(self,train_X,train_y,test_X,test_y,interval=5):
        self.train_X=np.asarray(train_X)
        self.test_X=np.asarray(test_X)
        train_X_st = Standardize(self.train_X, self.train_X)  # 标准化训练集
        test_X_st = Standardize(self.train_X, self.test_X)  # 标准化测试集
        self.train_X_st = train_X_st
        self.test_X_st = test_X_st
        self.train_y=np.asarray(train_y)
        self.test_y = np.asarray(test_y)
        y = np.eye(len(np.unique(self.train_y)))[self.train_y.astype(int)].T#把标签转化为one-hot矩阵
        self.interval=interval
        self.loss=[]
        self.accuracy_train=[]
        self.accuracy_test=[]
        index = 0  # batch边界索引
        """
        e则生成网格矩阵，用以绘制分类边界
        """
        self.left_border = int(np.min([np.min(train_X[:, 0]), np.min(test_X[:, 0])]))
        self.right_border = int(np.max([np.max(train_X[:, 0]), np.max(test_X[:, 0])]) + 1)
        self.bottom_border = int(np.min([np.min(train_X[:, 1]), np.min(test_X[:, 1])]))
        self.top_border = int(np.max([np.max(train_X[:, 1]), np.max(test_X[:, 1])]) + 1)
        feature1 = np.linspace(self.left_border, self.right_border, 400)
        feature2 = np.linspace(self.bottom_border, self.top_border, 400)
        self.broadcast_feature1, self.broadcast_feature2 = np.meshgrid(feature1, feature2)  # 生成网格矩阵，用以绘制分类边界
        XX = np.c_[self.broadcast_feature1.ravel(), self.broadcast_feature2.ravel()]  # 平铺并合并self.broadcast_feature1和self.broadcast_feature2，用以预测每个点的分类值
        self.XX_st = Standardize(train_X, XX)  # 标准化
        self.ZZ=[]
        for i in range(self.times):
            """
            计算每次迭代的batch
            x_batch:每次迭代的输入特征集
            y_batch:每次迭代的标签集
            """
            next_index=index + self.batch_size
            if next_index<=self.train_X_st.shape[0]:
                X_batch = self.train_X_st[index:next_index].reshape(-1,self.feature_num)
                y_batch=y[:, index:next_index].reshape(self.class_num,-1)
            else:
                index2=next_index - self.train_X_st.shape[0]
                X_batch=np.concatenate((self.train_X_st[index:].reshape(-1,self.feature_num),self.train_X_st[:index2].reshape(-1,self.feature_num)),axis=0)
                y_batch = np.concatenate((y[:, index:].reshape(self.class_num,-1), y[:, :index2].reshape(self.class_num,-1)),axis=1)
            index = next_index % self.train_X_st.shape[0]
            if i%self.interval==0 or i==self.times-1:
                """
                如果save_contour=True，每迭代interval次预测每个网格点的类别并保存,以及保存迭代完成后的预测结果
                """
                if self.save_contour:
                    self.ZZ.append(self.predict(self.XX_st).reshape(self.broadcast_feature1.shape))
                """
                每迭代interval次计算并储存用每次更新的权值预测的测试集和训练集预测准确率,以及保存迭代完成后的预测准确率
                """
                train_predict_y = self.predict(self.train_X_st)
                test_predict_y = self.predict(self.test_X_st)
                self.accuracy_train.append(self.accuracy(train_predict_y, self.train_y))
                self.accuracy_test.append(self.accuracy(test_predict_y, self.test_y))
            """
            计算并储存用每次更新的权值预测后的损失函数值
            """
            batch_Hypothesis_X=self.feedforword(X_batch.T)
            if i%self.interval==0:
                self.loss.append(self.loss_function.calculate(batch_Hypothesis_X, y_batch))#每迭代interval次计算并储存一次,以及保存迭代完成后的损失函数值

            self.gradient_A[-1] = self.loss_function.gradient(batch_Hypothesis_X, y_batch)#更新gradiengt_A[-1]
            if i<self.times-1:#最后一次迭代过后不更新梯度
                self.backforword(i)#反向传播更新梯度
    def predict(self,X):
        Hypothesis_X = self.feedforword(X.T)
        predict_y=np.argmax(Hypothesis_X,axis=0)
        return predict_y
    def accuracy(self,predict_y,y):
        ac=np.sum(np.where(predict_y==y,1,0))/len(y)
        return ac

if __name__ == '__main__':
    aa = Arbitrary_Scale_Neural_Network_for_Classification(Layer_scale=[10,8,2],
                                                           activation_function=[ ELU(),tanh,None_activation],
                                                           learning_rate=0.01, times_interation=500, batch_size=64,
                                                           feature_num=2, class_num=2, loss_function=MSE,optimizer=Adam,save_contour=True)
    train_X, train_y, test_X, test_y = load_dataset.Exam()
    aa.fit(train_X, train_y, test_X, test_y,interval=20)
    print("training set accuracy:", aa.accuracy_train[-1])#输出训练集预测准确率
    print("test set accuracy:", aa.accuracy_test[-1])#输出测试集预测准确率
    print("loss change:",aa.loss)#输出损失函数变化值
    vis=vlz.visualize(aa)
    vis.show_all_animation()#展示损失函数值，准确率和分类边界的变化，仅适用于save_contour=True
    #vis.show_loss_and_accuracy_animation()#展示损失函数值，准确率的变化
    #vis.show_static()#静态展示展示损失函数值，准确率的变化和最终的分类边界
    #vis.ani.save('NeuralNetwork.gif',writer='imagemagick',fps=60)#保存动画 (需要安装imagemagick)
