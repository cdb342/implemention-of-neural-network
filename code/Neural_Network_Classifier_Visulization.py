"""
专属于Nerual_Network_Classifier.py文件的可视化代码
可绘制二属性的分布图和分界线变化图
可绘制损失函数值下降图和准确率变化图
------------------------
Author: cdb342
Date: 2020/10/31
Time: 17:13
Homepage: https://github.com/cdb342
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
class visualize():
    def __init__(self,aa):
        self.aa=aa
        self.ite = np.arange(0, self.aa.times, self.aa.interval)  # 根据迭代次数生成数组
    def show_static(self):
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))  # 创建画布和坐标轴
        self.ax[0].set_xlim(self.aa.left_border, self.aa.right_border)  # 设置x轴坐标
        self.ax[0].set_ylim(self.aa.bottom_border, self.aa.top_border)  # 设置y轴坐标
        self.ax[0].scatter(self.aa.train_X[:, 0], self.aa.train_X[:, 1], c=self.aa.train_y, cmap='Dark2', s=10,marker='o', label='Training set')  # 在第一个坐标轴上绘制训练集的分布
        self.ax[0].scatter(self.aa.test_X[:, 0], self.aa.test_X[:, 1], c=self.aa.test_y, cmap='Dark2', s=10, marker='x',label='Test set')  # 在第一个坐标轴上绘制测试集的分布
        self.ax[0].legend()  # 显示图例
        self.ax[0].set_xlabel("Feature1")  # 设置x轴标签
        self.ax[0].set_ylabel("Feature2")  # 设置y轴标签
        self.ax[0].set_title("Classification Boundaries")  # 设置标题
        self.ax[1].set_xlim(0, int(self.aa.times / self.aa.interval - 1) * self.aa.interval)
        self.ax[1].set_ylim(int(np.min(self.aa.loss)), int(np.max(self.aa.loss) + 1))
        self.ax[1].set_xlabel("Iteration")
        self.ax[1].set_ylabel("Loss Function")
        self.ax[1].set_title("Loss Change")
        self.ax[2].set_xlim(0, int(self.aa.times / self.aa.interval - 1) * self.aa.interval)
        self.ax[2].set_ylim(0, 1)
        self.ax[2].set_xlabel("Iteration")
        self.ax[2].set_ylabel("Accuracy")
        self.ax[2].set_title("Accuracy Change")

        ZZ = self.aa.predict(self.aa.XX_st).reshape(self.aa.broadcast_feature1.shape)
        self.cont = self.ax[0].contourf(self.aa.broadcast_feature1, self.aa.broadcast_feature2, ZZ, alpha=0.2, cmap='Set3')  # 在第一个坐标轴上绘制分类边界
        self.line_loss = self.ax[1].plot(self.ite, self.aa.loss)  # 在第二个坐标轴上绘制损失函数下降情况
        self.training_accuracy = self.ax[2].scatter(self.ite, self.aa.accuracy_train[:len(self.ite)], label='training set', s=10,c='red')  # 在第三个坐标轴上绘制训练集预测准确率的变化
        self.test_accuracy = self.ax[2].scatter(self.ite, self.aa.accuracy_test[:len(self.ite)], label='test set', s=10)  # 在第三个坐标轴上绘制测试集预测准确率的变化
        plt.legend()
        plt.show()
    def show_all_animation(self):
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))  # 创建画布和坐标轴
        self.ax[0].set_xlim(self.aa.left_border, self.aa.right_border)  # 设置x轴坐标
        self.ax[0].set_ylim(self.aa.bottom_border, self.aa.top_border)  # 设置y轴坐标
        self.ax[0].scatter(self.aa.train_X[:, 0], self.aa.train_X[:, 1], c=self.aa.train_y, cmap='Dark2', s=10, marker='o',label='Training set')  # 在第一个坐标轴上绘制训练集的分布
        self.ax[0].scatter(self.aa.test_X[:, 0], self.aa.test_X[:, 1], c=self.aa.test_y, cmap='Dark2', s=10, marker='x',label='Test set')  # 在第一个坐标轴上绘制测试集的分布
        self.ax[0].legend()  # 显示图例
        self.ax[0].set_xlabel("Feature1")  # 设置x轴标签
        self.ax[0].set_ylabel("Feature2")  # 设置y轴标签
        self.ax[0].set_title("Classification Boundaries")  # 设置标题
        self.ax[1].set_xlim(0, int(self.aa.times / self.aa.interval - 1) * self.aa.interval)
        self.ax[1].set_ylim(int(np.min(self.aa.loss)), int(np.max(self.aa.loss) + 1))
        self.ax[1].set_xlabel("Iteration")
        self.ax[1].set_ylabel("Loss Function")
        self.ax[1].set_title("Loss Change")
        self.ax[2].set_xlim(0, int(self.aa.times / self.aa.interval - 1) * self.aa.interval)
        self.ax[2].set_ylim(0, 1)
        self.ax[2].set_xlabel("Iteration")
        self.ax[2].set_ylabel("Accuracy")
        self.ax[2].set_title("Accuracy Change")
        """
        初始化动画参数
        """
        self.cont = self.ax[0].contourf(self.aa.broadcast_feature1,self.aa.broadcast_feature2, self.aa.ZZ[0], alpha=0.2, cmap='Set3')  # 在第一个坐标轴上绘制分类边界
        self.line_loss, = self.ax[1].plot([], [])  # 在第二个坐标轴上绘制损失函数下降情况
        self.training_accuracy = self.ax[2].scatter([], [], label='training set', s=10, c='red')  # 在第三个坐标轴上绘制训练集预测准确率的变化
        self.test_accuracy = self.ax[2].scatter([], [], label='test set', s=10)  # 在第三个坐标轴上绘制测试集预测准确率的变化

        self.ani = animation.FuncAnimation(self.fig, self.animate_all, frames=int(self.aa.times / self.aa.interval), interval=1)  # 生成动画
        plt.legend()  # 显示图例
        plt.show()
    def show_loss_and_accuracy_animation(self):
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))  # 创建画布和坐标轴
        self.ax[0].set_xlim(self.aa.left_border, self.aa.right_border)  # 设置x轴坐标
        self.ax[0].set_ylim(self.aa.bottom_border, self.aa.top_border)  # 设置y轴坐标
        self.ax[0].scatter(self.aa.train_X[:, 0], self.aa.train_X[:, 1], c=self.aa.train_y, cmap='Dark2', s=10,marker='o', label='Training set')  # 在第一个坐标轴上绘制训练集的分布
        self.ax[0].scatter(self.aa.test_X[:, 0], self.aa.test_X[:, 1], c=self.aa.test_y, cmap='Dark2', s=10, marker='x',label='Test set')  # 在第一个坐标轴上绘制测试集的分布
        self.ax[0].legend()  # 显示图例
        self.ax[0].set_xlabel("Feature1")  # 设置x轴标签
        self.ax[0].set_ylabel("Feature2")  # 设置y轴标签
        self.ax[0].set_title("Classification Boundaries")  # 设置标题
        self.ax[1].set_xlim(0, int(self.aa.times / self.aa.interval - 1) * self.aa.interval)
        self.ax[1].set_ylim(int(np.min(self.aa.loss)), int(np.max(self.aa.loss) + 1))
        self.ax[1].set_xlabel("Iteration")
        self.ax[1].set_ylabel("Loss Function")
        self.ax[1].set_title("Loss Change")
        self.ax[2].set_xlim(0, int(self.aa.times / self.aa.interval - 1) * self.aa.interval)
        self.ax[2].set_ylim(0, 1)
        self.ax[2].set_xlabel("Iteration")
        self.ax[2].set_ylabel("Accuracy")
        self.ax[2].set_title("Accuracy Change")

        ZZ=self.aa.predict(self.aa.XX_st).reshape(self.aa.broadcast_feature1.shape)
        self.cont = self.ax[0].contourf(self.aa.broadcast_feature1, self.aa.broadcast_feature2, ZZ,alpha=0.2, cmap='Set3')  # 在第一个坐标轴上绘制分类边界
        """
        初始化动画参数
        """
        self.line_loss, = self.ax[1].plot([], [])  # 在第二个坐标轴上绘制损失函数下降情况
        self.training_accuracy = self.ax[2].scatter([], [], label='training set', s=10,c='red')  # 在第三个坐标轴上绘制训练集预测准确率的变化
        self.test_accuracy = self.ax[2].scatter([], [], label='test set', s=10)  # 在第三个坐标轴上绘制测试集预测准确率的变化

        self.ani = animation.FuncAnimation(self.fig, self.animate_loss_and_accuracy, frames=int(self.aa.times / self.aa.interval), interval=1)  # 生成动画
        plt.legend()  # 显示图例
        plt.show()
    def animate_all(self,i):  # 动画更新函数
        for c in self.cont.collections:  #移除先前生成的等高线图，加快动画更新速度
            c.remove()
        self.cont = self.ax[0].contourf(self.aa.broadcast_feature1,self.aa.broadcast_feature2, self.aa.ZZ[i], alpha=0.2, cmap='Set3')  # Update classification boundaries
        self.line_loss.set_data(self.ite[:i + 1], self.aa.loss[:i + 1])  # 更新每次迭代的损失函数
        self.training_accuracy.set_offsets(np.stack((self.ite[:i + 1], self.aa.accuracy_train[:i + 1]), axis=1))  # 更新训练集预测准确率随迭代的变化
        self.test_accuracy.set_offsets(np.stack((self.ite[:i + 1], self.aa.accuracy_test[:i + 1]), axis=1))  # 更新测试集预测准确率随迭代的变化
        return self.line_loss, self.training_accuracy, self.test_accuracy, self.cont
    def animate_loss_and_accuracy(self, i):  # 动画更新函数
        self.line_loss.set_data(self.ite[:i + 1], self.aa.loss[:i + 1])  # 更新每次迭代的损失函数
        self.training_accuracy.set_offsets(np.stack((self.ite[:i + 1], self.aa.accuracy_train[:i + 1]), axis=1))  # 更新训练集预测准确率随迭代的变化
        self.test_accuracy.set_offsets(np.stack((self.ite[:i + 1], self.aa.accuracy_test[:i + 1]), axis=1))  # 更新测试集预测准确率随迭代的变化
        return self.line_loss, self.training_accuracy, self.test_accuracy
