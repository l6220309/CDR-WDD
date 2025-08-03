import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
from scipy.linalg import fractional_matrix_power
from sklearn.metrics import confusion_matrix

def plot_feature_map(data, label, title):
    data_min, data_max = np.min(data, 0), np.max(data, 0)
    k = (15-(-15)) / (data_max-data_min)
    #data = (data - data_min) / (data_max - data_min)
    #temp_test = data[:, 0]


    data = -15.0 + k * (data-data_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        #plt.text(data[i, 0], data[i, 1], str(label[i]),
        #         color=plt.cm.Set1((label[i]+1.0) / 10.),
        #         fontdict={'weight': 'bold', 'size': 9})
        #label为 0 1 2 3分别对应左手（红色），右手（蓝色），双脚（绿色），舌头（紫色）
        plt.scatter(data[i, 0], data[i, 1],s=20, color=plt.cm.Set1((label[i]+1.0) / 10.), marker='o')
    #plt.xticks([])
    #plt.yticks([])
    plt.title(title)
    return fig

def plot_feature_map_src_and_tar(src_fea, src_label, tar_fea, tar_label, tsne_all_fea, title):

    data_min, data_max = np.min(tsne_all_fea, 0), np.max(tsne_all_fea, 0)
    k = (30.0-(-30.0)) / (data_max-data_min)
    #data = (data - data_min) / (data_max - data_min)
    #temp_test = data[:, 0]

    all_fea = -30.0 + k * (tsne_all_fea-data_min)

    fig = plt.figure()
    #ax = plt.subplot(111)
    for i in range(src_fea.shape[0]):
        #plt.text(data[i, 0], data[i, 1], str(label[i]),
        #         color=plt.cm.Set1((label[i]+1.0) / 10.),
        #         fontdict={'weight': 'bold', 'size': 9})
        #label为 0 1 2 3分别对应左手（红色），右手（蓝色），双脚（绿色），舌头（紫色）
        #plt.scatter(all_fea[i, 0], all_fea[i, 1], s=20, color=plt.cm.Set1( (src_label[i]+1.0) / 10.), marker='o', facecolors='none' )

        #some_marker = 'none'

        if src_label[i] == 0:
            some_marker = 'o'
        elif src_label[i] == 1:
            some_marker = 'd'
        elif src_label[i] == 2:
            some_marker = 's'
        else:
            some_marker = '^'
        random_f = random.random()

        if random_f < 0.15:
            plt.scatter(all_fea[i, 0], all_fea[i, 1], s=20, color=plt.cm.Set1( 1.0 / 10.), marker=some_marker, facecolors='none')

    for j in range(tar_fea.shape[0]):
        #plt.text(data[i, 0], data[i, 1], str(label[i]),
        #         color=plt.cm.Set1((label[i]+1.0) / 10.),
        #         fontdict={'weight': 'bold', 'size': 9})
        #label为 0 1 2 3分别对应左手（红色），右手（蓝色），双脚（绿色），舌头（紫色）

        #some_marker = 'none'

        if tar_label[j] == 0:
            some_marker = 'o'
        elif tar_label[j] == 1:
            some_marker = 'd'
        elif tar_label[j] == 2:
            some_marker = 's'
        else:
            some_marker = '^'
        #plt.scatter(all_fea[i + src_fea.shape[0], 0], all_fea[i + src_fea.shape[0], 1],s=20, color=plt.cm.Set1( (tar_label[i]+1.0) / 10.), marker='x')
        plt.scatter(all_fea[j + src_fea.shape[0], 0], all_fea[j + src_fea.shape[0], 1],s=20, color=plt.cm.Set1( 2.0 / 10.), marker=some_marker, facecolors='none')

    #plt.xticks([])
    #plt.yticks([])
    plt.title(title)
    return fig





def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,  # 这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
    plt.figure()
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize = 32)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize = 32)
    # 这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
    plt.savefig('Model_and_Result/ConfusionMats/confusionmatrix32.svg', format='svg', dpi=350)
    #plt.show()


def plt_confusion_quick_A():
    # coding=utf-8
    #import matplotlib.pyplot as plt
    #import numpy as np
    #from sklearn.metrics import confusion_matrix

    save_flg = True

    # confusion = confusion_matrix(y_test, y_pred)
    #Set A
    confusion = np.array([[0.84, 0.074, 0.035,  0.044, ],
                          [0.06, 0.83 , 0.06,   0.042, ],
                          [0.05, 0.033, 0.84,   0.064, ],
                          [0.05, 0.063, 0.065,  0.85, ],
                          ])



    plt.figure(figsize=(16, 14))  # 设置图片大小
    #plt.figure(figsize=(40, 40))  # 设置图片大小

    # 1.热度图，后面是指定的颜色块，cmap可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Greens)
    plt.colorbar()  # 右边的colorbar

    # 2.设置坐标轴显示列表
    indices = range(len(confusion))
    classes = ['Left hand', 'Right hand', 'Feet', 'Tongue']
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    plt.xticks(indices, classes, rotation=45, fontdict={'family' : 'Times New Roman', 'size':22})  # 设置横坐标方向，rotation=45为45度倾斜
    plt.yticks(indices, classes, fontdict={'family' : 'Times New Roman', 'size':22})

    # 3.设置全局字体
    # 在本例中，坐标轴刻度和图例均用新罗马字体['TimesNewRoman']来表示
    # ['SimSun']宋体；['SimHei']黑体，有很多自己都可以设置
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.sans-serif'] = ['TimesNewRoman']
    # plt.rcParams['axes.unicode_minus'] = False

    # 4.设置坐标轴标题、字体
    plt.ylabel('True label', fontdict={'family' : 'Times New Roman', 'size':26})
    plt.xlabel('Predicted label', fontdict={'family' : 'Times New Roman', 'size':26} )
    plt.title('Confusion matrix', fontdict={'family' : 'Times New Roman', 'size':26})

    #plt.xlabel('预测值')
    #plt.ylabel('真实值')
    #plt.title('混淆矩阵', fontsize=12, fontfamily="SimHei")  # 可设置标题大小、字体

    # 5.显示数据
    normalize = True
    fmt = '.2f' if normalize else 'd'
    thresh = confusion.max() / 2.

    for i in range(len(confusion)):  # 第几行
        for j in range(len(confusion[i])):  # 第几列
            plt.text(j, i, format(confusion[i][j], fmt),
                     fontdict={'family': 'Times New Roman', 'size': 26},
                     horizontalalignment="center",  # 水平居中。
                     verticalalignment="center",  # 垂直居中。
                     color="white" if confusion[i, j] > thresh else "black")

    # 6.保存图片
    if save_flg:
        #plt.savefig("./picture/confusion_matrix.png")
        plt.savefig('Model_and_Result/ConfusionMats/confusionmatrix32.svg', format='svg', dpi=500)

    # 7.显示
    plt.show()

def plt_confusion_quick_B():
    # coding=utf-8
    #import matplotlib.pyplot as plt
    #import numpy as np
    #from sklearn.metrics import confusion_matrix

    save_flg = True

    # confusion = confusion_matrix(y_test, y_pred)

    #Set B
    confusion = np.array([[0.88, 0.11],
                          [0.12, 0.89],
                          ])



    plt.figure(figsize=(16, 14))  # 设置图片大小

    # 1.热度图，后面是指定的颜色块，cmap可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Greens)
    plt.colorbar()  # 右边的colorbar

    # 2.设置坐标轴显示列表
    indices = range(len(confusion))
    #classes = ['Left hand', 'Right hand', 'Feet', 'Tongue']
    classes = ['Left hand', 'Right hand']
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    plt.xticks(indices, classes, rotation=45, fontdict={'family' : 'Times New Roman', 'size':22})  # 设置横坐标方向，rotation=45为45度倾斜
    plt.yticks(indices, classes, fontdict={'family' : 'Times New Roman', 'size':22})

    # 3.设置全局字体
    # 在本例中，坐标轴刻度和图例均用新罗马字体['TimesNewRoman']来表示
    # ['SimSun']宋体；['SimHei']黑体，有很多自己都可以设置
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.sans-serif'] = ['TimesNewRoman']
    # plt.rcParams['axes.unicode_minus'] = False

    # 4.设置坐标轴标题、字体
    plt.ylabel('True label', fontdict={'family' : 'Times New Roman', 'size':26})
    plt.xlabel('Predicted label', fontdict={'family' : 'Times New Roman', 'size':26} )
    plt.title('Confusion matrix', fontdict={'family' : 'Times New Roman', 'size':26})

    #plt.xlabel('预测值')
    #plt.ylabel('真实值')
    #plt.title('混淆矩阵', fontsize=12, fontfamily="SimHei")  # 可设置标题大小、字体

    # 5.显示数据
    normalize = True
    fmt = '.2f' if normalize else 'd'
    thresh = confusion.max() / 2.

    for i in range(len(confusion)):  # 第几行
        for j in range(len(confusion[i])):  # 第几列
            plt.text(j, i, format(confusion[i][j], fmt),
                     fontdict={'family': 'Times New Roman', 'size': 26},
                     horizontalalignment="center",  # 水平居中。
                     verticalalignment="center",  # 垂直居中。
                     color="white" if confusion[i, j] > thresh else "black")

    # 6.保存图片
    if save_flg:
        #plt.savefig("./picture/confusion_matrix.png")
        plt.savefig('Model_and_Result/ConfusionMats/confusionmatrix32.svg', format='svg', dpi=500)

    # 7.显示
    plt.show()


def euclidean_space_data_alignment(data):
    #EA处理之前的数据必须经过Band-pass filtering
    #一次trial的数据维度22*1000
    # X = 22 * 1000 , X^T = 1000 * 22, 协方差矩阵维度为 22 * 22

    # data维度:数据集A     n次实验 * 1 *  22样点 *  1125时间点
    # data维度:数据集B     n次实验 * 1 *  3采样点 * 1125时间点

    data_result = data

    num_of_trials = data.shape[0]
    cov_matrix_dim = data.shape[2]
    r_bar = np.zeros((cov_matrix_dim, cov_matrix_dim))
    for i in range(num_of_trials):
        x = data[i][0]
        x_transpose = x.transpose()
        r_temp = np.dot(x, x_transpose)
        r_bar = r_bar + r_temp

    r_bar = r_bar / (num_of_trials * num_of_trials)

    r_result = fractional_matrix_power(r_bar, -0.5)
    #  为特征值     为特征向量
    #eigen_values, eigen_vectors = linalg.eig(r_bar)
    #diagonal = np.diag(eigen_values**(-0.5))
    #r_result = eigen_vectors * diagonal * linalg.inv(eigen_vectors)

    for i in range(num_of_trials):
        x = data[i][0]
        data_result[i][0] = np.dot(r_result, x)

    return data_result



