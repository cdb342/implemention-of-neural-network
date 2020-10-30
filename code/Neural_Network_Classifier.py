import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import load_data_mnist
"""
数据集导入类
Exam():导入Exam数据集
Iris():导入Iris数据集
MNIST():导入MNIST数据集
"""
class lode_dataset:
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
Adam优化器
"""
# def Adam(v,s,g,i):
#     v=0.9*v+0.1*g
#     s=0.999*s+0.001*g**2
#     v_correct = v/ (1 - 0.9 ** i)
#     s_correct = s / (1 - 0.999 ** i)
#     g=v_correct/(np.sqrt(s_correct) + 10 ** (-8))
#     return v,s,g
def Adam(v,s,g,i):
    v=np.add(np.multiply(0.9,v),np.multiply(0.1,g))
    s=np.add(np.multiply(0.999,s),np.multiply(0.001,np.power(g,2)))
    v_correct =np.divide( v, (1 - 0.9 ** i))
    s_correct = np.divide(s , (1 - 0.999 ** i))
    g=np.divide(v_correct,np.add(np.power(s_correct,1/2) , 10 ** (-8)))
    return v,s,g
"""
sigmoid类
activate:返回用sigmoid函数激活后的结果
gradient:返回sigmoid函数的导数
"""
class sigmoid:
    def activate(z):
        a=np.zeros(z.shape)
        a[z>=0]=1 / (1 + np.exp(-z[z>=0]))
        a[z<0]=np.exp(z[z<0])/(1+np.exp(z[z<0]))
        return a
    def gradient(z):
        a = np.zeros(z.shape)
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
class  ReLU:
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
"""
MSE类
activate:返回用MSE函数激活后的结果
gradient:返回MSE函数的导数
"""
class MSE:
    def activate(Hypothesis_X, y):
        return np.sum((Hypothesis_X-y)**2)/2
    def gradient(Hypothesis_X, y):
        return Hypothesis_X-y
# class CrossEntropy:
#     def activate(Hypothesis_X, y):
#         activate_results=-np.sum(np.multiply(y[Hypothesis_X>0],np.log(Hypothesis_X[Hypothesis_X>0])))
#         return activate_results
#     def gradient(Hypothesis_X, y):
#         gradient_results=np.zeros(Hypothesis_X.shape)
#         gradient_results[Hypothesis_X>0]=-y[Hypothesis_X!=0]/(Hypothesis_X[Hypothesis_X!=0])
#         return gradient_results
"""
CrossEntropy类
activate:返回用CrossEntropy函数激活后的结果
gradient:返回CrossEntropy函数的导数
"""
class CrossEntropy:
    def activate(Hypothesis_X, y):
        return -np.sum(np.multiply(y,np.log(Hypothesis_X+10**(-8))))
    def gradient(Hypothesis_X, y):
        return -y/(Hypothesis_X+10**(-8))
"""
神经网络主体
__init__:输入神经网络规模、每层的激活函数、学习率、迭代次数、batch大小、输入特征数、类别数、损失函数进行初始化
feedback:实现前向传播
backforword:反向传播更新梯度
fit:训练权值并储存每次迭代的损失函数和准确率
predict:预测分类结果
accuracy:计算准确率
"""
class Arbitrary_Scale_Neural_Network_for_Classification():
    def __init__(self,Layer_scale,activation_function,
                 learning_rate=0.01,times_interation=500,batch_size=64,feature_num=2,class_num=2,loss_function=MSE):
        self.Layer_scale = Layer_scale  # 隐藏层和输出层的规模
        self.activation_function = activation_function  # 每层的激活函数
        self.yita=learning_rate#学习率，默认为0.01
        self.times=times_interation#迭代次数，默认为500次
        self.batch_size=batch_size#batch大小，默认为64
        self.feature_num=feature_num#输入特征数，默认为2
        self.class_num=class_num#类别数，默认为2
        self.loss_function=loss_function#使用的损失函数，默认使用MSE
        """
        初始化权值和梯度矩阵及Adam优化的参数
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
        self.s_W=[0] * (len(self.Layer_scale)+1)
        self.v_W=[0] * (len(self.Layer_scale)+1)
        self.s_b=[0] * (len(self.Layer_scale)+1)
        self.v_b=[0] * (len(self.Layer_scale)+1)
    def feedforword(self,X):
        self.A=[X]
        self.Z=[0]
        for i in range(len(self.Layer_scale)):
            self.Z.append(np.dot(self.W[i+1],self.A[i])+self.b[i+1])#第i+1层未激活的值
            self.A.append(self.activation_function[i].activate(self.Z[i+1]))#第i+1层激活后的值
        return self.A[-1]#返回输出层
    def backforword(self):
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
        # for i in range(len(self.Layer_scale)):
        #     self.s_W[-1-i],self.v_W[-1-i],self.gradient_W[-1-i]=Adam(self.s_W[-1-i],self.v_W[-1-i],self.gradient_W[-1-i],i+1)
        #     self.s_b[-1-i],self.v_b[-1-i],self.gradient_b[-1-i]=Adam(self.s_b[-1-i],self.v_b[-1-i],self.gradient_b[-1-i],i+1)
        #     self.W[-1-i]-=self.yita*self.gradient_W[-1-i]
        #     self.b[-1-i]-=self.yita*self.gradient_b[-1-i]
        self.s_W, self.v_W, self.gradient_W = Adam(self.s_W, self.v_W,self.gradient_W, i + 1)
        self.s_b, self.v_b, self.gradient_b = Adam(self.s_b, self.v_b,self.gradient_b, i + 1)
        self.W -= np.multiply(self.yita , self.gradient_W)
        self.b -= np.multiply(self.yita ,self.gradient_b)
    def fit(self,X,y,test_X,test_y):
        self.X = np.asarray(X)
        self.y=np.asarray(y)
        y = np.eye(len(np.unique(y)))[y.astype(int)].T#把标签转化为one-hot矩阵
        self.loss=[]
        index = 0#batch左边索引
        self.accuracy_train=[]
        self.accuracy_test=[]
        for i in range(self.times):
            """
            计算每次迭代的batch
            x_batch:每次迭代的输入特征集
            y_batch:每次迭代的标签集
            """
            next_index=index + self.batch_size
            if next_index<=self.X.shape[0]:
                X_batch = self.X[index:next_index].reshape(-1,self.feature_num)
                y_batch=y[:, index:next_index].reshape(self.class_num,-1)
            else:
                index2=next_index - self.X.shape[0]
                X_batch=np.concatenate((self.X[index:].reshape(-1,self.feature_num),self.X[:index2].reshape(-1,self.feature_num)),axis=0)
                y_batch = np.concatenate((y[:, index:].reshape(self.class_num,-1), y[:, :index2].reshape(self.class_num,-1)),axis=1)
            index = next_index % self.X.shape[0]
            """
            计算并储存用每次更新的权值预测的测试集和训练集预测准确率
            """
            train_predict_y = self.predict(self.X)
            test_predict_y = self.predict(test_X)
            self.accuracy_train.append(self.accuracy(train_predict_y, self.y))
            self.accuracy_test.append(self.accuracy(test_predict_y, test_y))
            """
            计算并储存用每次更新的权值预测后的损失函数
            """
            batch_Hypothesis_X=self.feedforword(X_batch.T)
            self.loss.append(self.loss_function.activate(batch_Hypothesis_X, y_batch))

            self.gradient_A[-1] = self.loss_function.gradient(batch_Hypothesis_X, y_batch)#更新gradiengt_A[-1]
            self.backforword()#反向传播更新梯度
    def predict(self,X):
        Hypothesis_X = self.feedforword(X.T)
        predict_y=np.argmax(Hypothesis_X,axis=0)
        return predict_y
    def accuracy(self,predict_y,y):
        ac=np.sum(np.where(predict_y==y,1,0))/len(y)
        return ac

if __name__ == '__main__':
    aa = Arbitrary_Scale_Neural_Network_for_Classification(Layer_scale=[8, 8, 3],
                                                           activation_function=[sigmoid, sigmoid, sigmoid],
                                                           learning_rate=0.01, times_interation=1000, batch_size=40,
                                                           feature_num=2, class_num=3, loss_function=CrossEntropy)
    train_X, train_y, test_X, test_y = lode_dataset.Iris()
    train_X_st = Standardize(train_X, train_X)#标准化训练集
    test_X_st = Standardize(train_X, test_X)#标准化测试集
    aa.fit(train_X_st, train_y, test_X_st, test_y)#
    predict_y_train = aa.predict(train_X_st)
    print("training set accuracy:", aa.accuracy_train[-1])#输出训练集预测准确率
    predict_y_test = aa.predict(test_X_st)
    print("test set accuracy:", aa.accuracy_test[-1])#输出测试集预测准确率
    print(aa.loss)

    """
    可视化部分
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # 创建画布和坐标轴
    """
    计算分布图的坐标轴端点位置
    """
    left_border=int(np.min([np.min(train_X[:,0]),np.min(test_X[:,0])]))
    right_border=int(np.max([np.max(train_X[:,0]),np.max(test_X[:,0])])+1)
    bottom_border=int(np.min([np.min(train_X[:,1]),np.min(test_X[:,1])]))
    top_border=int(np.max([np.max(train_X[:,1]),np.max(test_X[:,1])])+1)

    ax[0].set_xlim(left_border, right_border)  # 设置x轴坐标
    ax[0].set_ylim(bottom_border, top_border)  # 设置y轴坐标
    ax[0].scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='Dark2',s=10,marker='o',label='Training set')  # 在第一个坐标轴上绘制训练集的分布
    ax[0].scatter(test_X[:, 0], test_X[:, 1], c=test_y, cmap='Dark2',s=10,marker='x',label='Test set')  # 在第一个坐标轴上绘制测试集的分布
    ax[0].legend()#显示图例
    ax[0].set_xlabel("Feature1")  # 设置x轴标签
    ax[0].set_ylabel("Feature2")  # 设置y轴标签
    ax[0].set_title("Classification Boundaries")  # 设置标题
    ax[1].set_xlim(0, aa.times-1)
    ax[1].set_ylim(int(np.min(aa.loss)), int(np.max(aa.loss)+1))
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Cost Function")
    ax[1].set_title("Cost Change")
    ax[2].set_xlim(0, aa.times-1)
    ax[2].set_ylim(0, 1)
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("Accuracy")
    ax[2].set_title("Accuracy Change")

    """
    绘制分类边界
    仅适输入特征数为2的情况
    """
    feature1 = np.linspace(left_border, right_border, 400)
    feature2 = np.linspace(bottom_border, top_border, 400)
    XX1, XX2 = np.meshgrid(feature1, feature2)  # 生成网格矩阵，用以绘制分类边界
    XX = np.c_[XX1.ravel(), XX2.ravel()]  # 平铺并合并XX1和XX2，用以预测每个点的分类值
    XX_st=Standardize(train_X,XX)#标准化
    ZZ=aa.predict(XX_st).reshape(XX1.shape)
    cont = ax[0].contourf(XX1, XX2, ZZ, alpha=0.2, cmap='Set3')  # 在第一个坐标轴上绘制分类边界

    line_loss, = ax[1].plot([], [])  # 在第二个坐标轴上绘制损失函数下降情况
    training_accuracy = ax[2].scatter([], [], label='training set', s=10,c='red')  #在第三个坐标轴上绘制训练集预测准确率的变化
    test_accuracy = ax[2].scatter([], [], label='test set',s=10)  # 在第三个坐标轴上绘制测试集预测准确率的变化
    ite = np.arange(aa.times)  #根据迭代次数生成数组

    def animate(i):  # 动画更新函数
        line_loss.set_data(ite[:i], aa.loss[:i])  #更新每次迭代的损失函数
        training_accuracy.set_offsets(np.stack((ite[:i], aa.accuracy_train[:i]),axis=1))  # 更新训练集预测准确率随迭代的变化
        test_accuracy.set_offsets(np.stack((ite[:i], aa.accuracy_test[:i]),axis=1))  # 更新测试集预测准确率随迭代的变化
        return  line_loss, training_accuracy, test_accuracy

    ani = animation.FuncAnimation(fig, animate, frames=aa.times, interval=1)  #生成动画
    plt.legend()  #显示图例
    plt.show()
    #ani.save('ThreeLayers_NeuralNetwork_Iris.gif',writer='imagemagick',fps=60)#保存动画 (需要安装imagemagick)
