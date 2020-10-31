# 任意规模的神经网络分类器
### - 内置Exam和Iris两个简单数据集和MNIST数据集（因为没有实现并行加速，所以无法在短时间内训练MNIST数据集，可以选择分割数据集的一部分进行训练）
### 支持Adam,Momentum,Adagrad,RMSprop等优化器
### 支持sigmoid,tanh,ReLU,ELU激活函数
### 支持MSE和CrossEntropy损失函数
### 支持自定义梯度下降每次迭代的mini-batch大小，隐藏层和输出层的规模以及每一层的激活函数
### 支持绘制与保存损失函数值，准确率，分类边界随迭代次数的变化图（可通过save_contour参数选择是否在每次更新权值后保存绘制分类边界变化的参数）
### 支持绘制与保存损失函数值，准确率随迭代次数的变化的静态图和最终的分类边界静态图
### 适用于对于机器学习算法的学习与交流
### 设置Layer_scale=[2],activation_function=[None_activation]，loss_function=CrossEntropy，即可实现二分类的逻辑回归
### 设置Layer_scale=[c>=2],activation_function=[None_activation]，loss_function=CrossEntropy，即可实现c分类的softmax回归
