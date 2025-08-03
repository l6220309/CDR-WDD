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

#%%
import os
import time

import keras.losses
import numpy as np
import tensorflow as tf
import random
import argparse

import seaborn as sns

from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay


from tensorflow.keras.utils import plot_model

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import models
import center_loss_function
import utils
from preprocess import get_data_transfer_learning
from preprocess import get_data


#%%
#def draw_learning_curves(history):
#    plt.plot(history.history['accuracy'])
#    plt.plot(history.history['val_accuracy'])
#    plt.title('Model accuracy')
#    plt.ylabel('Accuracy')
#    plt.xlabel('Epoch')
#    plt.legend(['Train', 'val'], loc='upper left')
#    plt.show()
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('Model loss')
#    plt.ylabel('Loss')
#    plt.xlabel('Epoch')
#    plt.legend(['Train', 'val'], loc='upper left')
#    plt.show()
#    plt.close()
#
#def draw_confusion_matrix(cf_matrix, sub, results_path):
#    # Generate confusion matrix plot
#    display_labels = ['Left hand', 'Right hand','Foot','Tongue']
#    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
#                                display_labels=display_labels)
#    disp.plot()
#    disp.ax_.set_xticklabels(display_labels, rotation=12)
#    plt.title('Confusion Matrix of Subject: ' + sub )
#    plt.savefig(results_path + '/subject_' + sub + '.png')
#    plt.show()
#
#def draw_performance_barChart(num_sub, metric, label):
#    fig, ax = plt.subplots()
#    x = list(range(1, num_sub+1))
#    ax.bar(x, metric, 0.5, label=label)
#    ax.set_ylabel(label)
#    ax.set_xlabel("Subject")
#    ax.set_xticks(x)
#    ax.set_title('Model '+ label + ' per subject')
#    ax.set_ylim([0,1])
    
    
#%% Training 
def train(dataset_conf, train_conf, results_path):
    # Get the current 'IN' time to calculate the overall training time
    #in_exp = time.time()
    # Create a file to store the path of the best model among several runs
    #best_models = open(results_path + "/best models.txt", "w")
    # Create a file to store performance during training
    log_write = open(results_path + "/log.txt", "w")
    # Create a .npz file (zipped archive) to store the accuracy and kappa metrics
    # for all runs (to calculate average accuracy/kappa over all runs)
    #perf_allRuns = open(results_path + "/perf_allRuns.npz", 'wb')


    # Get dataset paramters
    n_sub = dataset_conf.n_sub
    dataset = dataset_conf.dataset
    data_path = dataset_conf.data_path
    is_standard = dataset_conf.is_standard
    ea_preprocess = dataset_conf.ea_preprocess
    # LOSO = dataset_conf.LOSO
    channels = dataset_conf.n_channels
    n_class = dataset_conf.n_classes


    # Get training hyperparamters
    batch_size = train_conf.batch_size
    epochs = train_conf.epochs
    patience = train_conf.patience
    lr = train_conf.lr
    LearnCurves = train_conf.LearnCurves # Plot Learning Curves?
    n_train = train_conf.n_train
    exp_name = train_conf.exp_name
    stop_criteria =  train_conf.stop_criteria

    w_adv = train_conf.w_adv
    w_t = train_conf.w_t
    w_s = train_conf.w_s
    w_c = train_conf.w_c


    #utils.plt_confusion_quick_A();

    # Initialize variables
    #acc = np.zeros((n_sub, n_train))
    #kappa = np.zeros((n_sub, n_train))

    #在模型的Input中，可以用dtype='int32'指定输入类型。

    # Iteration over subjects 一个人一个模型（每个个体搞一个模型），每次选一个人作为目标域
    #for sub in range(n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
    for sub in [2]:
        # Get the current 'IN' time to calculate the subject training time

        print('\nTraining on subject ', sub+1)
        log_write.write('\nTraining on subject ' + str(sub+1) + '\n')
        # Initiating variables to save the best subject accuracy among multiple runs.
        #BestSubAcc = 0   #每个个体跑多次，找到最优准确率
        #bestTrainingHistory = []

        # 结果为：model = "s5"    s表示subject,受害者，5表示第五个
        model = 's' + str(sub+1)
        # 存放结果的文件夹
        result_dir = 'Model_and_Result' + '/' + exp_name + '/' + model + dataset
        # 存放模型的文件夹
        #model_directory = result_dir + '/models'

        if tf.io.gfile.exists(result_dir):
            tf.io.gfile.rmtree(result_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # Get training and test data这里的wild card代表的是非独热编码的标签
        #X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data( data_path, sub, True , is_standard )
        # = dataFile.bciiv2a_multitask_da(subject=args.subject, dataset=args.dataset, data_len=args.data_len, ea_preprocess=True)
        source_data, source_label, target_data, target_label, tst_data, tst_label = get_data_transfer_learning(dataset, data_path, sub, dataset_conf.n_sub, ea_preprocess, is_standard)


        # Iteration over multiple runs
        training_time = 0.0
        testing_time = 0.0

        for train in range(n_train): # How many repetitions of training for subject i.
            # Get the current 'IN' time to calculate the 'run' training time

            # Create folders and files to save trained models for all runs
            #filepath = results_path + '/saved models/run-{}'.format(train+1)
            #if not os.path.exists(filepath):
                 #os.makedirs(filepath)
            #             filepath = filepath + '/subject-{}.h5'.format(sub+1)
            #
            #
            #=====================DRDA PART======================


            #pred train训练集预测结果
            predTrain = []    #prediction result of training
            idx_epoch = []

            # 模型加载和创建
                # 优化器相关
            lr_schedule = ExponentialDecay(initial_learning_rate=lr,
                                           decay_steps=10, decay_rate=0.99999, staircase=True)#global step填哪？

            g_opti = Adam(learning_rate=lr_schedule, beta_1=0.5)
            #如果要使用WGAN的损失，要用RMSprop
            d_opti = RMSprop(learning_rate=0.0005)
            #d_opti = Adam(learning_rate=lr_schedule)

            # 鉴别器
            D = getModel('ATCNet_Discriminator', dataset)
            '''
            D.compile(
                loss=binary_crossentropy,
                optimizer=d_opti
                #metrics=['accuracy']
            )
            '''
            D.compile(
                loss=models.d_loss_for_WGAN,
                optimizer=d_opti
                # metrics=['accuracy']
            )


            # 生成器（特征提取器 + 分类器）
            G = getModel('ATCNet_Generator', dataset)
            # 生成器 + 参数固定（暂时不可训练）的鉴别器
            Combined = models.Generator_Containing_Discriminator_Multiple_Outputs(G, D, in_chans=channels, n_classes=n_class)

            Combined.compile(
                loss=[
                     models.adv_loss_for_WGAN,
                     categorical_crossentropy,
                     categorical_crossentropy,
                     center_loss_function.CenterLoss(Combined.get_layer(name='center_loss_layer'))],
                loss_weights=[   #超参数设置
                     w_adv,
                     w_s,
                     w_t,
                     w_c],
                optimizer=g_opti)

            #Combined.summary()
            #plot_model(Combined, show_shapes=True)

            # 总的batch数目
            max_iter = int(len(source_data) / batch_size)

            # Adversarial ground truths
            #valid = np.ones((batch_size, 1)).astype(np.float32)
            #valid = (-1.0 * valid).astype(np.float32)
            #fake = np.zeros((batch_size, 1)).astype(np.float32)

            # WGAN的Adversarial ground truths
            valid = np.ones((batch_size, 1)).astype(np.float32)
            valid = (-1.0 * valid).astype(np.float32)
            fake = np.ones((batch_size, 1)).astype(np.float32)
            fake = (1.0 * fake).astype(np.float32)

            #开启epochs轮训练
            best_acc = 0.0
            result_file = result_dir + '/training_log_winsize_4feature.txt'

            for epoch in range(epochs):
                train_gloss = 0.0  #生成器损失
                #train_dloss = 0.0  #鉴别器损失

                train_dloss_source = 0.0  #鉴别器损失
                train_dloss_target = 0.0  #鉴别器损失

                classification_loss = 0.0  #分类器损失
                # 源域总共5780个样本(maxIter = 90,参与训练共5760个，样本，随机舍弃20个样本)
                # 目标域总共420个样本
                sam_list_s = list(range(len(source_data)))
                random.shuffle(sam_list_s)

                train_start_time = time.time()

                for itr in range(max_iter):
                    batch_trn_idx = sam_list_s[batch_size*itr:batch_size*(itr+1)]
                    # 源域随机取出一个batch 64个样本
                    signal_train_s = source_data[batch_trn_idx]

                    # 从目标域420个样本中，随机取出一个batch 64个样本
                    batch_idx = np.random.choice(target_data.shape[0], len(batch_trn_idx))
                    signal_train_t = target_data[batch_idx]

                    # 标签提取
                    label_train_s = source_label[batch_trn_idx]
                    label_train_t = target_label[batch_idx]

                    #in_sub = time.time()
                    # 生成器前向传播，得到潜在空间中提取到的特征
                    prob_s, latent_feat_s, _ = G.predict(signal_train_s)
                    prob_t, latent_feat_t, _ = G.predict(signal_train_t)

                    # 鉴别器前向传播
                    # d_global_logits_t = D.predict(latent_feat_t)
                    # d_global_logits_s = D.predict(latent_feat_s)
                    # -----------
                    # 鉴别器训练
                    # Train the discriminator
                    # -----------
                    # train_on_batch:
                    #   单输出模型，既有loss，也有metrics, 此时 y_pred 为一个列表，代表这个 mini-batch 的 loss 和 metrics,列表长度为 1+len(metrics)
                    D.trainable = True
                    for la in D.layers:
                        # clip D weights        for l in D.layers:
                        weights = la.get_weights()
                        weights = [np.clip(w, -0.01, 0.01) for w in weights]
                        la.set_weights(weights)
                        
                    d_loss_and_metrics_real = D.train_on_batch(x=latent_feat_t, y=valid)  # target样本尽可能产生-1
                    d_loss_and_metrics_fake = D.train_on_batch(x=latent_feat_s, y=fake)   # source样本尽可能产生1

                    #d_loss = 0.5 * np.add(d_loss_and_metrics_real[0], d_loss_and_metrics_fake[0])  # 鉴别器损失值
                    #d_loss = 0.5 * np.add(d_loss_and_metrics_real, d_loss_and_metrics_fake)  # 鉴别器损失值

                    # -----------
                    # 生成器训练
                    # Train the Generator
                    # -----------
                    D.trainable = False
                    for la in D.layers:
                        la.trainable = False
                    g_loss_and_metrics = Combined.train_on_batch(x=[signal_train_s, signal_train_t], y=[fake, label_train_s, label_train_t, tf.argmax(label_train_t, axis=1)])
                    g_loss = g_loss_and_metrics[0]



                    train_gloss = train_gloss + g_loss
                    train_dloss_source = train_dloss_source + d_loss_and_metrics_fake
                    train_dloss_target = train_dloss_target + d_loss_and_metrics_real

                    bce_logits_false = keras.losses.BinaryCrossentropy(from_logits=False)
                    c_loss_source = bce_logits_false(label_train_s, prob_s).numpy()
                    c_loss_target = bce_logits_false(label_train_t, prob_t).numpy()
                    classification_loss = classification_loss + c_loss_source + c_loss_target

                    predTrain.extend(prob_t)
                    idx_epoch.extend(batch_idx)

                train_end_time = time.time()
                per_epoch_train_time = (train_end_time - train_start_time)
                training_time += per_epoch_train_time



                aa = np.array(predTrain)
                accTrain = accuracy_score(np.argmax(target_label[idx_epoch], 1), np.argmax(aa, 1))

                label_test = tst_label  #320 * 2
                signal_test = tst_data
                if len(signal_test.shape) != 4:
                    signal_test = np.expand_dims(signal_test, axis=-1)

                signal_placeholder = signal_test   #only used for placeholding purposes

                test_start_time = time.time()

                _, _, prob_test, _ = Combined.predict([signal_test, signal_placeholder])
                _, latent_feat_on_test, _ = G.predict(signal_test)

                test_end_time = time.time()
                per_epoch_test_time = test_end_time - test_start_time
                testing_time += per_epoch_test_time

                #TODO:把这里的test loss算出来
                #test_loss_per_epoch = tf.losses.binary_cross_entropy(logits=prob_t, onehot_labels=label_test)   #+ tf.losses.get_regularization_loss()
                bce_logits_false = keras.losses.BinaryCrossentropy(from_logits=False)
                test_loss_per_epoch = bce_logits_false(label_test, prob_test).numpy()

                #GET ACC
                acc = accuracy_score(np.argmax(label_test, 1), np.argmax(prob_test, 1))
                kappa = cohen_kappa_score(np.argmax(label_test, 1), np.argmax(prob_test, 1))

                #TSNE FEATURE MAP PLOTTING for feature visulazition
                #tsne_result = TSNE(n_components=2, learning_rate=50, random_state=501, init='pca').fit_transform(latent_feat_on_test)  # 降至2维
                #label_for_viualization = [np.argmax(i) for i in label_test]
                #fig_feature_map = utils.plot_feature_map(tsne_result, label_for_viualization, 't-SNE embedding of the EEG deep features of ATC-DAN');

                #if (acc > 0.94):
                #    _, latent_feat_on_all_source, _ = G.predict(source_data)
                #    _, latent_feat_on_all_target, _ = G.predict(target_data)

                #    #first_dim_src = latent_feat_on_all_source.shape[0]
                #    #first_dim_tar = latent_feat_on_all_target.shape[0]

                #    #lab_dimension_src = (first_dim_src, )
                #    #lab_dimension_tar = (first_dim_tar, )

                #    #all_source_label = np.ones(lab_dimension_src, dtype=int)
                #    #all_target_label = np.zeros(lab_dimension_tar, dtype=int)


                #    all_source_label = [np.argmax(i) for i in source_label]
                #    all_target_label = [np.argmax(i) for i in target_label]

                #    fea_res = np.concatenate((latent_feat_on_all_source, latent_feat_on_all_target), axis=0)

                #    #TSNE FEATURE MAP PLOTTING for domain adaptation
                #    tsne_result = TSNE(n_components=2, learning_rate=50, random_state=501, init='pca').fit_transform(fea_res)  # 降至2维
                #    label_for_viualization = np.concatenate((all_source_label, all_target_label), axis=0)
                #    fig_feature_map = utils.plot_feature_map(tsne_result, label_for_viualization, 't-SNE embedding of the EEG deep features');

                #if (acc > 0.95):
                #_, latent_feat_on_all_source, _ = G.predict(source_data)
                #_, latent_feat_on_all_target, _ = G.predict(target_data)

                #fea_res = np.concatenate((latent_feat_on_all_source, latent_feat_on_all_target), axis=0)

                #all_source_label = [np.argmax(i) for i in source_label]
                #all_target_label = [np.argmax(i) for i in target_label]

                ##TSNE FEATURE MAP PLOTTING for domain adaptation
                #tsne_result = TSNE(n_components=2, learning_rate=45, random_state=501, init='pca').fit_transform(fea_res)  # 降至2维

                #fig_feature_map = utils.plot_feature_map_src_and_tar(latent_feat_on_all_source, all_source_label, latent_feat_on_all_target, all_target_label, tsne_result, 't-SNE embedding of the EEG deep features')

                ## SAVING THE FEATURE MAP
                #fig_path = 'Model_and_Result/DeepFeature/fig_%d_%f.svg' % (epoch, acc)
                #fig_feature_map.savefig(fig_path, format='svg', dpi=500)





                #PLOTTING CONFUSION MAT
                #label_notations_confusionMat = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
                #label_confusionMat = label_test.argmax(axis=-1)  # 将one-hot转化为label
                #conf_mat = confusion_matrix(y_true=label_confusionMat, y_pred=np.argmax(prob_test, 1))
                #utils.plot_confusion_matrix(conf_mat, normalize=False, target_names=label_notations_confusionMat, title='Confusion Matrix')



                print('[Epoch: %2d / %2d]\n' % (epoch+1, epochs))
                print('[Accuracy: %f ]\n' % acc)

                if(acc > best_acc):
                    best_acc = acc


                if os.path.exists(result_file):
                    append_write = 'a'  # append if already exists
                else:
                    append_write = 'w'  # make a new file if not

                with open(result_file, append_write) as text_file:
                    text_file.write("[EPOCH: %2d / %2d ] train gloss: %.4f, train dloss_source: %.4f, train dloss_target: %.4f, test_loss: %.4f, acc on train set: %.4f, acc on test set: %.4f, kappa: %.4f\n"
                           % (epoch + 1, epochs, train_gloss/max_iter, train_dloss_source/max_iter, train_dloss_target/max_iter, test_loss_per_epoch, accTrain, acc, kappa))

                #print('[EPOCH: %2d / %2d (global step = %d)] train gloss: %.4f, accT: %.4f, dloss: %.4f; test loss: %.4f, acc: %.4f, kappa: %.4f'
                #          % (epoch + 1, epochs, training_util.global_step(sess, global_step), train_gloss, accT, train_dloss, test_loss_val, acc, kappa))
                #with open(result_dir + '/training_log.txt', 'a') as text_file:
                #    text_file.write("[EPOCH: %2d / %2d (global step = %d)] train gloss: %.4f, accT: %.4f, dloss: %.4f; test loss: %.4f, acc: %.4f, kappa: %.4f'\n"
                #            % (epoch + 1, epochs, training_util.global_step(sess, global_step), train_gloss, accT, train_dloss, test_loss_val, acc, kappa))

                ## save model
                #checkpoint_path = model_directory + '/' + 'model.ckpt'
                #saver.save(sess, checkpoint_path, global_step=global_step)



                #if stop_criteria == 'test_loss':
                #    if test_loss_val < best_loss-0.0002:
                #        best_loss = test_loss_val
                #        best_acc = acc
                #        best_kappa = kappa
                #        stop_step = 0
                #        best_epoch = epoch
                #        best_global_step = training_util.global_step(sess, global_step)
                #    else:
                #        stop_step += 1
                #        if stop_step > early_stop_tolerance:
                #            # print('Early stopping is trigger at epoch: %2d. ----->>>>> Best loss: %.4f, acc: %.4f at epoch %2d (step = %2d)'
                #            #       %(epoch+1, best_loss, best_acc, best_epoch+1, best_global_step))
                #            #
                #            # with open(result_dir + '/training_log.txt', 'a') as text_file:
                #            #     text_file.write(
                #            #         'Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f at epoch %2d (step = %2d)\n'
                #            #         % (best_loss, best_acc, best_epoch+1, best_global_step))
                #            # s = open(model_directory + '/checkpoint').read()
                #            # s = s.replace('model_checkpoint_path: "model.ckpt-' + str(training_util.global_step(sess, global_step)) + '"', 'model_checkpoint_path: "model.ckpt-' + str(best_global_step) +'"')
                #            # f = open(model_directory + '/checkpoint', 'w')
                #            # f.write(s)
                #            # f.close()
                #            break
                #    elif stop_criteria == 'test_acc':
                #        print(stop_criteria)
                #        if (best_acc < acc) or (abs(best_acc - acc) < 0.0001 and val_loss_value < best_loss):
                #            best_acc = acc
                #            best_loss = val_loss_value
                #            best_kappa = kappa
                #            best_epoch = epoch
                #            best_global_step = training_util.global_step(sess, global_step)
            with open(result_file, 'a') as text_file:
                text_file.write("Best Acc for subject %d is: %f\n" % (sub, best_acc))
            #print('Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f, kappa: %.4f at epoch %2d (step = %2d)' %(best_loss, best_acc, best_kappa, best_epoch+1, best_global_step))

            #with open(result_dir + '/training_log.txt', 'a') as text_file:
            #    text_file.write(
            #        'Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f, kappa: %.4f at epoch %2d (step = %2d)\n'
            #        % (best_loss, best_acc, best_kappa, best_epoch+1, best_global_step))
            #s = open(model_directory + '/checkpoint').read()
            #s = s.replace('model_checkpoint_path: "model.ckpt-' + str(training_util.global_step(sess, global_step)) + '"', 'model_checkpoint_path: "model.ckpt-' + str(best_global_step) +'"')
            #f = open(model_directory + '/checkpoint', 'w')
            #f.write(s)
            #f.close()


        # Store the path of the best model among several runs
        #best_run = np.argmax(acc[sub, :])
        #filepath = '/saved models/run-{}/subject-{}.h5'.format(best_run+1, sub+1)+'\n'
        #best_models.write(filepath)
        ## Get the current 'OUT' time to calculate the subject training time
        #out_sub = time.time()
        ## Print & write the best subject performance among multiple runs
        #info = '----------\n'
        #info = info + 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(sub+1, best_run+1, ((out_sub-in_sub)/60))
        #info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(acc[sub, best_run], np.average(acc[sub, :]), acc[sub,:].std() )
        #info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(kappa[sub, best_run], np.average(kappa[sub, :]), kappa[sub,:].std())
        #info = info + '\n----------'
        #print(info)
        #log_write.write(info+'\n')
        # Plot Learning curves 
        #if (LearnCurves == True):
            #print('Plot Learning Curves ....... ')
            #draw_learning_curves(bestTrainingHistory)
          
    # Get the current 'OUT' time to calculate the overall training time
    #out_exp = time.time()
    #info = '\nTime: {:.1f} h   '.format((out_exp-in_exp)/(60*60))
    #print(info)
    #log_write.write(info+'\n')
    
    # Store the accuracy and kappa metrics as arrays for all runs into a .npz 
    # file format, which is an uncompressed zipped archive, to calculate average
    # accuracy/kappa over all runs.
    #np.savez(perf_allRuns, acc = acc, kappa = kappa)
    #
    ## Close open files
    #best_models.close()
    #log_write.close()
    #perf_allRuns.close()


#%% Evaluation

def test(model, dataset_conf, results_path, allRuns = True):
    # Open the  "Log" file to write the evaluation results 
    log_write = open(results_path + "/log.txt", "a")
    # Open the file that stores the path of the best models among several random runs.
    best_models = open(results_path + "/best models.txt", "r")   
    
    # Get dataset paramters
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    
    # Initialize variables
    acc_bestRun = np.zeros(n_sub)
    kappa_bestRun = np.zeros(n_sub)  
    cf_matrix = np.zeros([n_sub, n_classes, n_classes])

    # Calculate the average performance (average accuracy and K-score) for 
    # all runs (experiments) for each subject.
    if(allRuns): 
        # Load the test accuracy and kappa metrics as arrays for all runs from a .npz 
        # file format, which is an uncompressed zipped archive, to calculate average
        # accuracy/kappa over all runs.
        perf_allRuns = open(results_path + "/perf_allRuns.npz", 'rb')
        perf_arrays = np.load(perf_allRuns)
        acc_allRuns = perf_arrays['acc']
        kappa_allRuns = perf_arrays['kappa']
    
    # Iteration over subjects 
    for sub in range(n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
        # Load data
        _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub, LOSO, isStandard)
        # Load the best model out of multiple random runs (experiments).
        filepath = best_models.readline()
        model.load_weights(results_path + filepath[:-1])
        # Predict MI task
        y_pred = model.predict(X_test).argmax(axis=-1)
        # Calculate accuracy and K-score
        labels = y_test_onehot.argmax(axis=-1)
        acc_bestRun[sub] = accuracy_score(labels, y_pred)
        kappa_bestRun[sub] = cohen_kappa_score(labels, y_pred)
        # Calculate and draw confusion matrix
        cf_matrix[sub, :, :] = confusion_matrix(labels, y_pred, normalize='pred')
        draw_confusion_matrix(cf_matrix[sub, :, :], str(sub+1), results_path)
        
        # Print & write performance measures for each subject
        info = 'Subject: {}   best_run: {:2}  '.format(sub+1, (filepath[filepath.find('run-')+4:filepath.find('/sub')]) )
        info = info + 'acc: {:.4f}   kappa: {:.4f}   '.format(acc_bestRun[sub], kappa_bestRun[sub] )
        if(allRuns): 
            info = info + 'avg_acc: {:.4f} +- {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(
                np.average(acc_allRuns[sub, :]), acc_allRuns[sub,:].std(),
                np.average(kappa_allRuns[sub, :]), kappa_allRuns[sub,:].std() )
        print(info)
        log_write.write('\n'+info)
      
    # Print & write the average performance measures for all subjects     
    info = '\nAverage of {} subjects - best runs:\nAccuracy = {:.4f}   Kappa = {:.4f}\n'.format(
        n_sub, np.average(acc_bestRun), np.average(kappa_bestRun)) 
    if(allRuns): 
        info = info + '\nAverage of {} subjects x {} runs (average of {} experiments):\nAccuracy = {:.4f}   Kappa = {:.4f}'.format(
            n_sub, acc_allRuns.shape[1], (n_sub * acc_allRuns.shape[1]),
            np.average(acc_allRuns), np.average(kappa_allRuns)) 
    print(info)
    log_write.write(info)
    
    # Draw a performance bar chart for all subjects 
    draw_performance_barChart(n_sub, acc_bestRun, 'Accuracy')
    draw_performance_barChart(n_sub, kappa_bestRun, 'K-score')
    # Draw confusion matrix for all subjects (average)
    draw_confusion_matrix(cf_matrix.mean(0), 'All', results_path)
    # Close open files     
    log_write.close() 
'''
    
#%%
def getModel(model_name,dataset):

    # Select the model
    if(model_name == 'ATCNet_Generator'):
        if dataset == 'A':
            model = models.ATCNet_Generator(
                # Dataset parameters
                n_classes=4,
                in_chans=22,
                in_samples=1125,
                # Sliding window (SW) parameter
                n_windows=4,
                # Attention (AT) block parameter
                attention='mha',  # Options: None, 'mha','mhla', 'cbam', 'se'
                # Convolutional (CV) block parameters
                eegn_F1=8,
                eegn_D=2,
                eegn_kernelSize=32,
                eegn_poolSize=7,
                eegn_dropout=0.4,
                # Temporal convolutional (TC) block parameters
                tcn_depth=2,
                tcn_kernelSize=4,
                tcn_filters=16,
                tcn_dropout=0.4,
                tcn_activation='elu'
            )
        elif dataset == 'B':
            model = models.ATCNet_Generator(
                # Dataset parameters
                n_classes=2,
                in_chans=3,
                in_samples=1125,
                # Sliding window (SW) parameter
                n_windows=4,
                # Attention (AT) block parameter
                attention='mha',  # Options: None, 'mha','mhla', 'cbam', 'se'
                # Convolutional (CV) block parameters
                eegn_F1=8,
                eegn_D=2,
                eegn_kernelSize=16,
                eegn_poolSize=7,
                eegn_dropout=0.4,
                # Temporal convolutional (TC) block parameters
                tcn_depth=2,
                tcn_kernelSize=4,
                tcn_filters=16,
                tcn_dropout=0.4,
                tcn_activation='elu'
            )

    elif(model_name == 'ATCNet_Discriminator'):
        model = models.ATCNet_Discriminator()


    #elif(model_name == 'ATCNet'):
    #    # Train using the proposed model (ATCNet): https://doi.org/10.1109/TII.2022.3197419
    #    model = models.ATCNet(
    #        # Dataset parameters
    #        n_classes = 4,
    #        in_chans = 22,
    #        in_samples = 1125,
    #        # Sliding window (SW) parameter
    #        n_windows = 5,
    #        # Attention (AT) block parameter
    #        attention = 'mha', # Options: None, 'mha','mhla', 'cbam', 'se'
    #        # Convolutional (CV) block parameters
    #        eegn_F1 = 16,
    #        eegn_D = 2,
    #        eegn_kernelSize = 64,
    #        eegn_poolSize = 7,
    #        eegn_dropout = 0.3,
    #        # Temporal convolutional (TC) block parameters
    #        tcn_depth = 2,
    #        tcn_kernelSize = 4,
    #        tcn_filters = 32,
    #        tcn_dropout = 0.3,
    #        tcn_activation='elu',
    #        #fuse method
    #        fuse = 'concat'
    #        )
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model

    '''
    #=======================other methods======================================
    elif(model_name == 'TCNet_Fusion'):
        # Train using TCNet_Fusion: https://doi.org/10.1016/j.bspc.2021.102826
        model = models.TCNet_Fusion(n_classes = 4)      
    elif(model_name == 'EEGTCNet'):
        # Train using EEGTCNet: https://arxiv.org/abs/2006.00622
        model = models.EEGTCNet(n_classes = 4)          
    elif(model_name == 'EEGNet'):
        # Train using EEGNet: https://arxiv.org/abs/1611.08024
        model = models.EEGNet_classifier(n_classes = 4) 
    elif(model_name == 'EEGNeX'):
        # Train using EEGNeX: https://arxiv.org/abs/2207.12369
        model = models.EEGNeX_8_32(n_timesteps = 1125 , n_features = 22, n_outputs = 4)
    elif(model_name == 'DeepConvNet'):
        # Train using DeepConvNet: https://doi.org/10.1002/hbm.23730
        model = models.DeepConvNet(nb_classes = 4 , Chans = 22, Samples = 1125)
    elif(model_name == 'ShallowConvNet'):
        # Train using ShallowConvNet: https://doi.org/10.1002/hbm.23730
        model = models.ShallowConvNet(nb_classes = 4 , Chans = 22, Samples = 1125)
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model
    '''
    
#%%
def run():
    # Get dataset path
    #data_path = os.path.expanduser('~') + '/BCI Competition IV/BCI Competition IV-2a/' discarded path

    # Create a FOLDER to store the results of the experiment
    results_path = os.getcwd() + "/results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)   # Create a new directory if it does not exist

    # Set dataset parameters
    # dataset_conf = {'n_classes': 4, 'n_sub': 9, 'n_channels': 22, 'data_path': data_path, 'isStandard': True, 'LOSO': False}
    dataset_conf_parser = argparse.ArgumentParser()



    data_path = './data/OriginalDataSetA/'  #path for dataset A
    #data_path = './data/OriginalDataSetB/'   #path for dataset B

    dataset_conf_parser.add_argument('--dataset', default='A')
    dataset_conf_parser.add_argument('--n_classes', default=4, type=int) #A:4 B:2
    dataset_conf_parser.add_argument('--n_channels', default=22, type=int)#A:22 B:3
    dataset_conf_parser.add_argument('--ea_preprocess', default=False, type=bool)
    dataset_conf_parser.add_argument('--is_standard', default=False, type=bool)


    dataset_conf_parser.add_argument('--data_path', default=data_path)
    dataset_conf_parser.add_argument('--n_sub', default=9, type=int)
    dataset_conf = dataset_conf_parser.parse_args()

    # Set training hyperparamters
    train_conf_parser = argparse.ArgumentParser()

    train_conf_parser.add_argument('--batch_size', default=32, type=int)
    train_conf_parser.add_argument('--epochs', default=200, type=int)
    train_conf_parser.add_argument('--patience', default=300, type=int)           #change to 20 maybe?
    train_conf_parser.add_argument('--lr', default=0.001, type=float)
    train_conf_parser.add_argument('--n_train', default=1, type=int)              #一个模型训练多少次
    train_conf_parser.add_argument('--exp_name', default='ATC_GAN_WinSize4_EA')               #实验名
    train_conf_parser.add_argument('--LearnCurves', default=False, type=bool)      #whether to plot learning curve
    train_conf_parser.add_argument('--save_model', default=False, type=bool)       #是否保存模型
    train_conf_parser.add_argument('--stop_criteria', default='test_acc')         #

    #train_conf_parser.add_argument('--w_adv', default=0.05)
    train_conf_parser.add_argument('--w_adv', default=0.1)
    train_conf_parser.add_argument('--w_t', default=1.0)
    train_conf_parser.add_argument('--w_s', default=1.0)
    train_conf_parser.add_argument('--w_c', default=0.05)
    train_conf = train_conf_parser.parse_args()


    # Train the model
    train(dataset_conf, train_conf, results_path)

    # Evaluate the model based on the weights saved in the '/results' folder
    #model = getModel(train_conf.get('model'))
    #test(model, dataset_conf, results_path)




#%%
if __name__ == "__main__":

    tf.config.run_functions_eagerly(True)
    physical_gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_virtual_device_configuration(
        physical_gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*8)]
    )
    logical_gpus = tf.config.list_logical_devices("GPU")

    run()
