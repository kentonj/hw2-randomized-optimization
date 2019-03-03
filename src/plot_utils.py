import platform
import matplotlib as mpl
if platform.mac_ver()[0] != '':
    print('mac os version detected:', platform.mac_ver()[0], ' - switching matplotlib backend to TkAgg')
    mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import os
import itertools


def plot_confusion_matrix(algo_family, algo_list, classes, cmap=plt.cm.Blues, figure_action='show', figure_path='figures/cm', file_name=None):
    '''
    adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    '''
    f, axarr = plt.subplots(2, len(algo_list))
    tick_marks = np.arange(len(classes))
    plt.setp(axarr, xticks=tick_marks, xticklabels=classes, yticks=tick_marks, yticklabels=classes)
    for i in range(len(algo_list)):
        # i = column of subplot
        algo = algo_list[i]
        axarr[0, i].imshow(algo.get_cm(), interpolation='nearest', cmap=cmap)
        axarr[0, i].set_title(str(algo.model_type))
        
        thresh = algo.get_cm().max() / 2.
        for j, k in itertools.product(range(algo.get_cm().shape[0]), range(algo.get_cm().shape[1])):
            axarr[0, i].text(k, j, format(algo.get_cm()[j, k], 'd'),
                            horizontalalignment="center",
                            color="white" if algo.get_cm()[j, k] > thresh else "black")


        axarr[1, i].imshow(algo.get_normalized_cm(), interpolation='nearest', cmap=cmap)
        
        thresh = algo.get_normalized_cm().max() / 2.
        for j, k in itertools.product(range(algo.get_normalized_cm().shape[0]), range(algo.get_normalized_cm().shape[1])):
            axarr[1, i].text(k, j, format(algo.get_normalized_cm()[j, k], '.2f'),
                            horizontalalignment="center",
                            color="white" if algo.get_normalized_cm()[j, k] > thresh else "black")

    for ax in axarr.flat:
        ax.set(xlabel='Predicted label', ylabel='True label')
    for ax in axarr.flat:
        ax.label_outer()
    plt.tight_layout()
    if figure_action == 'show':
        plt.show()
    elif figure_action == 'save':
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        if file_name:
            plt.savefig(figure_path+'/'+file_name+'.png')
        else:
            plt.savefig(figure_path+'/'+str(algo.model_family)+'.png')
    plt.close()
    return None

def plot_model_family_learning_curves(model_family, algo_list, iter_based=False, figure_action='show', figure_path='figures/lc', file_name=None):
    line_type_dict = {
        'train':'-',
        'validation':'-.'
    }
    color_list = ['b','g','r','c','m','y','k','w', 'orange']
    
    plt.figure()
    plt.title('Learning Curves - ' + model_family)
    plt.ylabel('Score')
    
    if iter_based:
        plt.xlabel('Iterations')
    else:
        plt.ylim((0.0,1.0))
        plt.xlabel('Training Samples')
        
    algo_list = list(algo_list)
    for i in range(len(algo_list)):
        algo = algo_list[i]
        line_color = color_list[i]
        if iter_based:
            plt.plot(np.linspace(1,len(algo.get_iter_scores()), len(algo.get_iter_scores())), 
                    algo.get_iter_scores(), 
                    line_type_dict['train'], 
                    color=line_color,
                    label=(algo.model_type+' Training Score'))
        else:
            plt.plot(algo.train_sizes, 
                    algo.get_train_scores(), 
                    line_type_dict['train'], 
                    color=line_color,
                    label=(algo.model_type+' Training Score'))
            plt.plot(algo.train_sizes, 
                    algo.get_validation_scores(), 
                    line_type_dict['validation'], 
                    color=line_color,
                    label=(algo.model_type+' Validation Score'))
    plt.legend(loc='best')
    plt.grid()

    if figure_action == 'show':
        plt.show()
    elif figure_action == 'save':
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        if file_name:
            plt.savefig(figure_path+'/'+file_name+'.png')
        else:
            plt.savefig(figure_path+'/'+str(algo.model_family)+'.png')
    plt.close()
    return None