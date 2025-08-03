""" 
Copyright (C) 2022 King Saud University, Saudi Arabia 
SPDX-License-Identifier: Apache-2.0 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the 
License at

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. 

Author:  Hamdi Altaheri 
"""

# Dataset BCI Competition IV-2a is available on 
# http://bnci-horizon-2020.eu/database/data-sets

import numpy as np
import scipy.io as sio
import utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from scipy.linalg import fractional_matrix_power

#%%
def euclidean_space_data_alignment(data):
    #EA处理之前的数据必须经过Band-pass filtering
    #一次trial的数据维度22*1000
    # X = 22 * 1000 , X^T = 1000 * 22, 协方差矩阵维度为 22 * 22

    # data维度:数据集A     400次实验 * 3采样点 * 1000时间点
    # data维度:数据集B     288次实验* 22样点 * 1000时间点
    data_result = data

    num_of_trials = data.shape[0]
    cov_matrix_dim = data.shape[1]
    r_bar = np.zeros((cov_matrix_dim,cov_matrix_dim))
    for i in range(num_of_trials):
        x = data[i]
        x_transpose = x.transpose()
        r_temp = np.dot(x,x_transpose)
        r_bar = r_bar + r_temp

    r_bar = r_bar / num_of_trials

    r_result = fractional_matrix_power(r_bar, -0.5)
    #  为特征值     为特征向量
    #eigen_values, eigen_vectors = linalg.eig(r_bar)
    #diagonal = np.diag(eigen_values**(-0.5))
    #r_result = eigen_vectors * diagonal * linalg.inv(eigen_vectors)

    for i in range(num_of_trials):
        x = data[i]
        data_result[i] = np.dot(r_result,x)

    return data_result


#%%
def load_data_transfer_learing_LOSO (dataset, data_path, subject):
    """ Loading and Dividing of the data set based on the
    'Leave One Subject Out' (LOSO) evaluation approach.
    LOSO is used for  Subject-independent evaluation.
    In LOSO, the model is trained and evaluated by several folds, equal to the
    number of subjects, and for each fold, one subject is used for evaluation
    and the others for training. The LOSO evaluation technique ensures that
    separate subjects (not visible in the training data) are usedto evaluate
    the model.

        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
            Here, the subject data is used  test the model and other subjects data
            for training
    """

    source_data, source_label= [], []
    for sub in range (0,9):
        #path = data_path+'s' + str(sub+1) + '/'
        path = data_path

        if dataset == 'A':
            X1, y1 = load_data_form_setA(path, sub+1, training=True)
            X2, y2 = load_data_form_setA(path, sub+1, training=False)
        elif dataset == 'B':
            X1, y1 = load_data_form_setB(path, sub+1, training=True)
            X2, y2 = load_data_form_setB(path, sub+1, training=False)

        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        if (sub == subject):
            target_data = X1
            target_label = y1
            target_tst_data = X2
            target_tst_label = y2

        elif (source_data == []):
            source_data = X
            source_label = y
        else:
            source_data = np.concatenate((source_data, X), axis=0)
            source_label = np.concatenate((source_label, y), axis=0)

    return source_data, source_label, target_data, target_label, target_tst_data, target_tst_label




#%%
def load_data_form_setA(data_path, subject, training, all_trials = True):
	""" Loading and Dividing of the data set based on the subject-specific 
    (subject-dependent) approach.
    In this approach, we used the same training and testing dataas the original
    competition, i.e., 288 x 9 trials in session 1 for training, 
    and 288 x 9 trials in session 2 for testing.  
   
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
        all_trials: bool
            if True, load all trials
            if False, ignore trials with artifacts 
	"""
    # Define MI-trials parameters
	n_channels = 22
	n_tests = 6*48 	
	window_Length = 7*250 

	class_return = np.zeros(n_tests)
	data_return = np.zeros((n_tests, n_channels, window_Length))

	NO_valid_trial = 0
	if training:
		a = sio.loadmat(data_path +'A0'+str(subject)+'T.mat')
	else:
		a = sio.loadmat(data_path+'A0'+str(subject)+'E.mat')

	a_data = a['data']
	for ii in range(0,a_data.size):  #a_data.size = 9
		a_data1 = a_data[0,ii]
		a_data2= [a_data1[0,0]]
		a_data3= a_data2[0]
		a_X 		= a_data3[0]
		a_trial 	= a_data3[1]
		a_y 		= a_data3[2]
		a_artifacts = a_data3[5]

		for trial in range(0,a_trial.size):
 			if(a_artifacts[trial] != 0 and not all_trials):
 			    continue
 			data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:n_channels])
 			class_return[NO_valid_trial] = int(a_y[trial])
 			NO_valid_trial +=1


	return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]


# %%
def load_data_form_setB(data_path, subject, training, all_trials=True):
    """ Loading and Dividing of the data set based on the subject-specific
    (subject-dependent) approach.
    In this approach, we used the same training and testing dataas the original
    competition, i.e., 288 x 9 trials in session 1 for training,
    and 288 x 9 trials in session 2 for testing.

        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
        all_trials: bool
            if True, load all trials
            if False, ignore trials with artifacts
    """
    # Define MI-trials parameters
    n_channels = 3
    n_tests = 6 * 48 * 2  #？开的足够大能够容纳就行
    window_Length = 7 * 250 #？开的足够大能够容纳就行

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path + 'B0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(data_path + 'B0' + str(subject) + 'E.mat')

    a_data = a['data']
    for ii in range(0, a_data.size):  # a_data.size = 9
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0, a_trial.size):
            if (a_artifacts[trial] != 0 and not all_trials):
                continue
            data_return[NO_valid_trial, :, :] = np.transpose(
                a_X[int(a_trial[trial]):(int(a_trial[trial]) + window_Length), :n_channels])
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial += 1

    return data_return[0:NO_valid_trial, :, :], class_return[0:NO_valid_trial]


#%%
def get_data_transfer_learning(dataset, path, subject, num_subjects, ea_preprocess, is_standard):
    # Define dataset parameters
    fs = 250          # sampling rate
    if dataset == 'A':
        t1 = int(1.5 * fs)  # start time_point
        t2 = int(6 * fs)  # end time_point
    elif dataset == 'B':
        t1 = int(2.5 * fs)  # start time_point
        t2 = int(7 * fs)  # end time_point


    T = t2-t1         # length of the MI trial (samples or time_points)


    # Loading and Dividing of the data set based on the
    # 'Leave One Subject Out' (LOSO) evaluation approach.
    #X_train, y_train, X_test, y_test = load_data_transfer_learing_LOSO(path, subject)

    source_data, source_label, target_data, target_label, target_tst_data, target_tst_label = load_data_transfer_learing_LOSO(dataset, path, subject)

    #TODO:在这里加EA处理
    # Prepare training data
    # EA处理
    #if (ea_preprocess):
    #    target_data = utils.euclidean_space_data_alignment(target_data)
    #    target_tst_data = utils.euclidean_space_data_alignment(target_tst_data)

    # N_tr是N_trials , N_ch是N_channels
    N_tr, N_ch, _ = source_data.shape
    #拿到1.5s到6s的数据，并且进行reshape
    #拿到2.5s到7s的数据，并且进行reshape
    source_data = source_data[:, :, t1:t2].reshape(N_tr, 1, N_ch, T)
    #source_label_sparse = source_label
    source_label_onehot = (source_label-1).astype(int)
    source_label_onehot = to_categorical(source_label_onehot)

    N_tr, N_ch, _ = target_data.shape
    #拿到1.5s到6s的数据，并且进行reshape
    target_data = target_data[:, :, t1:t2].reshape(N_tr, 1, N_ch, T)
    #target_label_sparse = target_label
    target_label_onehot = (target_label-1).astype(int)
    target_label_onehot = to_categorical(target_label_onehot)

    # Prepare testing data
    N_test, N_ch, _ = target_tst_data.shape
    target_tst_data = target_tst_data[:, :, t1:t2].reshape(N_test, 1, N_ch, T)
    #target_tst_label_sparse = target_tst_label
    target_tst_label_onehot = (target_tst_label-1).astype(int)
    target_tst_label_onehot = to_categorical(target_tst_label_onehot)

    #TODO:在这里加EA处理
    # Standardize the data
    #if (is_standard == True):
    #    X_train, X_test = standardize_data(X_train, X_test, N_ch)


    source_data = source_data.astype(np.float32)
    target_data = target_data.astype(np.float32)
    target_tst_data = target_tst_data.astype(np.float32)

    #if (ea_preprocess):
    #    target_tst_data = utils.euclidean_space_data_alignment(target_tst_data)
    #    source_data = utils.euclidean_space_data_alignment(source_data)
    #    target_data = utils.euclidean_space_data_alignment(target_data)

    return source_data, source_label_onehot,  target_data, target_label_onehot, target_tst_data, target_tst_label_onehot

#%%
def standardize_data(X_train, X_test, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]第二个通道和多任务相关？
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test

#%% non cross-subject way of loading data
def get_data(path, subject, LOSO = False, isStandard = True):
    # Define dataset parameters
    fs = 250          # sampling rate
    t1 = int(1.5*fs)  # start time_point
    t2 = int(6*fs)    # end time_point
    T = t2-t1         # length of the MI trial (samples or time_points)

    # Load and split the dataset into training and testing 
    if LOSO:
        # Loading and Dividing of the data set based on the 
        # 'Leave One Subject Out' (LOSO) evaluation approach. 
        X_train, y_train, X_test, y_test = load_data_LOSO(path, subject)
    else:
        # Loading and Dividing of the data set based on the subject-specific 
        # (subject-dependent) approach.In this approach, we used the same 
        # training and testing data as the original competition, i.e., trials 
        # in session 1 for training, and trials in session 2 for testing.

        #path = path + 's{:}/'.format(subject+1)
        #X_train 维度 288trials 22电极 1750时间点
        X_train, y_train = load_data(path, subject+1, True)
        #X_test  维度 288trials 22电极 1750时间点,目前这一步还没有对标签one hot化
        X_test, y_test = load_data(path, subject+1, False)

    # Prepare training data
    # N_tr是N_trials , N_ch是N_channels
    N_tr, N_ch, _ = X_train.shape
    #拿到1.5s到6s的数据，并且进行reshape
    X_train = X_train[:, :, t1:t2].reshape(N_tr, 1, N_ch, T)
    y_train_onehot = (y_train-1).astype(int)
    y_train_onehot = to_categorical(y_train_onehot)
    # Prepare testing data 
    N_test, N_ch, _ = X_test.shape 
    X_test = X_test[:, :, t1:t2].reshape(N_test, 1, N_ch, T)
    y_test_onehot = (y_test-1).astype(int)
    y_test_onehot = to_categorical(y_test_onehot)	

    #TODO:在这里加EA处理
    # Standardize the data
    if (isStandard == True):
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot






## 训练集和测试集结果
#    trn_data, trn_label, tst_data, tst_label = [], [], [], []
#
#    # 总共受害者的数目
#    all_sub_num = num_subjects
#    # 目标域受害者s数组下标
#    sub = subject - 1
#
#    source_list = list(range(all_sub_num))  # 从0开始到all_sub_num - 1
#    # 从所有数据中把目标域的目标受害者去掉，留下的全是源域的数据
#    source_list.pop(sub)
#
#    # 读入所有源领域的数据
#    for data_path_suffix in source_list:
#        # 文件命名是从1开始的。所以这里需要加1
#        data_path_suffix = data_path_suffix + 1
#        # 使用matlab库读入数据
#        # 对于其中一个受试者，数据的维度是：(288, 22, 625) 22代表22个头部采样点，625是实验时间点，288样本数（试验次数）
#        load_data(data_path_suffix)
#
#        raw = sio.loadmat(path + 'A0' + str(data_path_suffix) + 'T.mat')
#
#        trn_data_temp, trn_label_temp, tst_data_temp, tst_label_temp = raw['trnData'], raw['trnLabel'][0], raw['tstData'], raw['tstLabel'][0]
#
#        # EA处理
#        if ea_preprocess:
#            trn_data_temp = euclidean_space_data_alignment(trn_data_temp)
#            tst_data_temp = euclidean_space_data_alignment(tst_data_temp)
#
#        trn_data.append(trn_data_temp)
#        trn_label.append(trn_label_temp)
#        tst_data.append(tst_data_temp)
#        tst_label.append(tst_label_temp)
#
#    trn_data = np.concatenate(trn_data, axis=0)
#    trn_label = np.hstack(trn_label)
#
#    tst_data = np.concatenate(tst_data, axis=0)
#    tst_label = np.hstack(tst_label)
#
#    # 对于源领域的数据，直接合并测试和训练数据
#    source_data = np.concatenate((trn_data, tst_data), axis=0)
#    source_label = np.concatenate((trn_label, tst_label), axis=0)
#
#    # 目标域 受害者下标
#    target_list = list(set(range(all_sub_num)).difference(set(source_list)))
#
#    for data_path_suffix in target_list:
#        data_path_suffix = data_path_suffix + 1
#
#        raw = sio.loadmat('./data/dataset2a1000/' + str(data_path_suffix) + '.mat')
#
#        #对于目标领域的数据，分开测试和训练数据
#        target_data, target_label, target_tst_data, target_tst_label = raw['trnData'], raw['trnLabel'][0], raw['tstData'], raw['tstLabel'][0]
#
#        #EA处理
#        if (ea_preprocess):
#            target_data = euclidean_space_data_alignment(target_data)
#            target_tstData = euclidean_space_data_alignment(target_tstData)
#
#
#    return source_data, source_label, target_data, target_label, target_tst_data, target_tst_label
#
#
#
#
#