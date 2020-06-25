from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import os
import glob
import xml.etree.ElementTree as et
import xml
from tqdm import tqdm

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
"""
xml_path = '/Users/Michele/Desktop/Faster_resNet/New_Annotations/Damage_only'
old_res_file = 'C:/Users/MoMoJo/Desktop/Faster_resNet/old_results/outputResult_level_only.txt'
test_file_path =(
    'C:/Users/MoMoJo/Desktop/Faster_resNet/VOC/VOC2007/ImageSets/Main/test.txt')
train_file_path =(
    'C:/Users/MoMoJo/Desktop/Faster_resNet/VOC/VOC2007/ImageSets/Main/train.txt')
test_files = pd.read_csv(test_file_path, sep=" ", header=None)
train_files = pd.read_csv(train_file_path, sep=" ", header=None)
old_res = pd.read_csv(old_res_file, header=None)
old_res_csv = pd.DataFrame(columns=('file', 'class', 'score'), index = None)
gt_class = pd.read_csv("./Damage_only/Damage_only_ground_truth.csv", index_col = 0)
len(gt_class)
gt_f = gt_class['file'].to_numpy()
gt_c = gt_class['class'].to_numpy()
for i in range(len(old_res)-98):
    print(i)
    _name = old_res[0][i+98].split('[')[0].split(' ')[0]
    _num = int(old_res[0][i+98].split('[')[0].split(' ')[1])
    if _num !=0:
       _class = int(old_res[0][i+98].split('[')[0].split(' ')[3])
       _score = float(old_res[0][i+98].split('[')[1].split(']')[1])
    else:
       _class = gt_c[np.where(gt_f==_name.split('.')[0])]
       _score = 0

    old_res_csv = old_res_csv.append(
        pd.DataFrame({'file':[_name],'class':[_class], 'score':[_score]}),
        ignore_index=True)
old_res_csv.to_csv('old_res_level_only.csv')

files = glob.glob(xml_path+'*.xml')
files.sort()
gt_class = pd.DataFrame(columns=('file', 'class'), index = None)

for file in tqdm(train_files[0]):
    #print(file)
    #file = train_files[0][0]
    tree = et.parse(xml_path + file + '.xml')
    root = tree.getroot()
    list(root)
    gt_class=gt_class.append(pd.DataFrame({'file':[file],
        'class':[int(root[6][0].text)]}),ignore_index=True)

gt_class.to_csv('Damage_train_gt.csv')
"""
ori_path = './old_results/'
gt_class = pd.read_csv("./Damage_only/Damage_only_ground_truth.csv", index_col = 0)
est = pd.read_csv(ori_path + "old_res_level_only.csv", index_col = 0)

est_res = est.to_numpy()
est_class = est_res[:,1]
est_score = est_res[:,2]
gt = gt_class['class'].to_numpy()
n_class = 3
fig_name = 'Faster RCNN'
type = 'Damage'

def non_used():
    FP = confmat.sum(axis=0) - np.diag(confmat)
    FN = confmat.sum(axis=1) - np.diag(confmat)
    TP = np.diag(confmat)
    TN = confmat.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    precision = TP / (TP+FP)  # 查准率
    recall = TP / (TP+FN)
    f_s = 2*(precision*recall)/(precision+recall)

    cm_normalized = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)

    plt.matshow(cm_normalized, cmap=plt.cm.gray)
    plt.show()

    precision_score(gt.astype('int'),est_class.astype('int'),average=None)
    recall_score(gt.astype('int'),est_class.astype('int'),average=None)
    f1_score(gt.astype('int'),est_class.astype('int'), average=None)


classes = [1,2,3]

confmat= confusion_matrix(y_true=gt.astype('int'),y_pred=est_class.astype('int'))

def plot_confusion_matrix(cm, savename):#title='Confusion Matrix'
    plt.figure(figsize=(9,9))
    font = {'family' : 'Times New Roman', 'weight':'bold', 'size': 18}
    plt.rc('font', **font)
    plt.rcParams['axes.linewidth'] = 2
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=18, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    #plt.title(title)
    #plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual Label')
    plt.xlabel('Predict Label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-', linewidth = 2)
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(ori_path+savename, format='pdf', dpi = 1200)
    plt.show()

plot_confusion_matrix(confmat, 'evaluate/Confusion_matrix.pdf')#title='Confusion Matrix'


def read_data(est_class, est_score, gt, n_class):

    Class = dict()
    Score = dict()
    GT = dict()
    for i in range(n_class):
        _gt = gt.copy().astype('float64')
        _class = est_class.copy().astype('float64')
        _score = est_score.copy().astype('float64')
        score_off = 0.01* _score
        _gt[np.where(gt!=(i+1))] = 0
        GT[i+1] = _gt
        _class[np.where(_class!=(i+1))] = 0
        Class[i+1] = _class
        _score[np.where(_class!=(i+1))] =0.0001#score_off[np.where(_class!=(i+1))]
        Score[i+1] = _score

    return Class, Score, GT

def cal(Class, Score, GT):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i+1], tpr[i+1], _ = roc_curve(GT[i+1]==i+1, Score[i+1])
        roc_auc[i+1] = auc(fpr[i+1], tpr[i+1])

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_class):
        precision[i+1], recall[i+1], _ = precision_recall_curve(
            GT[i+1]==(i+1),Score[i+1], pos_label=True)
        average_precision[i+1] = average_precision_score(
            GT[i+1]==(i+1),Score[i+1], pos_label=True)

    return fpr, tpr, roc_auc, precision, recall, average_precision

def display_plots(fpr, tpr, roc_auc, precision, recall, average_precision):
    # Plot all ROC curves
    plt.figure(figsize=(9,9))
    font = {'family' : 'Times New Roman', 'weight':'bold', 'size': 18}
    plt.rc('font', **font)
    plt.rcParams['axes.linewidth'] = 2
    lines = []
    labels = []
    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    l, = plt.plot([0, 1], [0, 1], color='gray', lw=2, alpha=0.2)
    lines.append(l)
    labels.append('base line')

    linestyles = cycle(['-', '-.', ':'])
    for i, linestyle in zip(range(n_class), linestyles):
        l, = plt.plot(fpr[i+1], tpr[i+1], linestyle=linestyle, color='k', lw=4)
        lines.append(l)
        labels.append('ROC curve of class {0} (area = {1:0.2f})'
                      ''.format(i+1, roc_auc[i+1]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=18)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title(fig_name +  " " + type +' ROC Curve', fontsize=18)
    fig = plt.gcf()
    fig.subplots_adjust(top= 0.99, bottom=0.24)
    plt.legend(lines, labels, loc=(0.18, -.31), prop=font)
    plt.savefig(ori_path + 'evaluate/'+ fig_name + 'ROC-' + type + '.pdf', dpi = 1200)

   # plot P-R curves
    plt.figure(figsize=(9, 9))
    font = {'family' : 'Times New Roman', 'size': 18}
    plt.rc('font', **font)
    plt.rcParams['axes.linewidth'] = 2
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', lw=2, alpha=0.2) #
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')

    for i, linestyle in zip(range(n_class), linestyles):
        l, = plt.plot(recall[i+1], precision[i+1], linestyle=linestyle, color='k', lw=4)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i+1, average_precision[i+1]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=18)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.title(fig_name + " " + type + ' Precision-Recall Curve', fontsize=18)
    fig = plt.gcf()
    fig.subplots_adjust(top= 0.99, bottom=0.24)
    plt.legend(lines, labels, loc=(0.15, -.31), prop=font)
    fig.savefig(ori_path + 'evaluate/'+ fig_name + 'P-R-' + type + '.pdf', dpi = 1200)

if __name__ == '__main__':
    Class, Score, GT = read_data(est_class, est_score, gt, n_class)
    fpr, tpr, roc_auc, precision, recall, average_precision = cal(Class, Score, GT)
    display_plots(fpr, tpr, roc_auc, precision, recall, average_precision)



precision_score(gt.astype('float64'),est_class.astype('float64'),average=None)
recall_score(gt.astype('float64'),est_class.astype('float64'),average=None)

f1_score(gt.astype('float64'),est_class.astype('float64'), average=None)
