import numpy as np
from numpy import dot
import scipy.io as io
import argparse
import json
import os
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.feature import hog
import sys
import random 

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import tensorly as tl
from tensorly import tucker_to_tensor
from tensorly.decomposition import tucker

sys.path.append('./STMMMMMD')

from kernel import get_kernel
from import_export import dump_model, load_model
from conversion import numpy_json_encoder

from LSSVM import LSSTCTucker, LSSTM, LSSVC, TensorDecompositionEstimate


class iEEG_fMRIDataLoader():

    def __init__(self, X, Y, decompositionEstimator=None):
        if decompositionEstimator is not None:
            decomTensor = np.zeros((X.shape))
            print('decomposition method: ', decompositionEstimator.method, 'rank: ', decompositionEstimator.rank)
            for j in range(len(decomTensor)):
                decomTensor[j] = decompositionEstimator.decompostionEstimate(X[j])
            self.X = decomTensor
            self.Y = Y
        else:
            self.X = X
            self.Y = Y
    
    def getData(self, index=None, total=False):
        if total:
            return self.X[:], self.Y[:]
        return self.X[index], self.Y[index]

    def getShape(self):
        return self.X[0].shape[1:]

    def getLen(self):
        return len(self.X)

    # def getData(self, index):
    #     return self.trainData[index], self.trainLabel[index], self.testData[index], self.testLabel[index]
    
    def getShape(self):
        return self.trainData[0].shape[1:]

        
def generateRankCombinationTucker(R1, R2, R3):
    combinationList = []
    for r1 in R1:
        for r2 in R2:
            for r3 in R3:
                combinationList.append([r1, r2, r3])
    return combinationList

def generateRankCombinationTT(R1, R2):
    combinationList = []
    for r1 in R1:
        for r2 in R2:
            combinationList.append([1, r1, r2, 1])
    return combinationList

def generateSVMParamCombination(sigma, gamma=None):
    # sigma is regulation param
    # gamma is rbf param
    combinationList = []
    for s in sigma:
        if gamma is not None:
            for g in gamma:
                combinationList.append([s, g])
        else:
            combinationList.append([s, 1])
    return combinationList
    
def testModel(model, dataLoader, flatten=False):
    X, y = dataLoader.getData(total=True)

    random.seed(122914)
    np.random.seed(122914)

    # 初始化SVM分类器
    clf = model

    # 定义计数器，用于计算平均准确率
    count = 0
    num_iter = 5

    for i in range(num_iter):
        # 将训练集拆分为真正的训练集和验证集
        # x_train_real, x_valid, y_train_real, y_valid = train_test_split(x_train, y_train, test_size=0.5, stratify=y_train, random_state=i)

        # 将数据集拆分为训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        # 将数据集和标签reshape
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        y_train = y_train.reshape(y_train.shape[0],)
        y_test = y_test.reshape(y_test.shape[0],)

        # 训练分类器
        clf.fit(x_train, y_train)
        
        # 预测测试集的标签
        y_pred = clf.predict(x_test)
        
        # 计算准确率
        accuracy = np.mean(y_test == y_pred)
        
        # 将准确率加到计数器上
        count += accuracy
        
        # 输出准确率
        print(f'Iteration {i+1}: {accuracy}')
        
    # 计算平均准确率
    average_accuracy = count / num_iter
    print(f'Average accuracy: {average_accuracy}')
    return average_accuracy


def generateRankCombinationTucker(R1, R2, R3):
    combinationList = []
    for r1 in R1:
        for r2 in R2:
            for r3 in R3:
                combinationList.append([r1, r2, r3])
    return combinationList

def generateRankCombinationTT(R1, R2):
    combinationList = []
    for r1 in R1:
        for r2 in R2:
            combinationList.append([1, r1, r2, 1])
    return combinationList

def generateSVMParamCombination(sigma, gamma=None):
    # sigma is regulation param
    # gamma is rbf param
    combinationList = []
    for s in sigma:
        if gamma is not None:
            for g in gamma:
                combinationList.append([s, g])
        else:
            combinationList.append([s, 1])
    return combinationList


def testiEEG_fMRI(kernelIndex=2, methodIndex=0, dataRatio=1):
    saveJson = False
    
    # set use subject List
    idList = ['01', '02']
    # idList = ['01', '02', '03', '05', '06', '07', '09', '10', '12', '13', '14', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '30', '31', '32', '33', '34', '36', '37', '38', '39', '40', '41', '42', '43', '45', '46', '48', '49', '50', '51', '54', '55', '57', '58', '59', '60', '61', '63']
    print('test id', idList)

    # use to save accuracy
    stmDict = {}
    lsstmDict = {}

    # choose decomposition method
    methodList = ['Tucker', 'Tensor Train', 'CP', 'rank-n']
    method = methodList[0]

    # choose kernel method
    kernelList = ['linear', 'poly', 'rbf']
    kernel = kernelList[2]
    print('use kernel', kernel)

    # set decomposition rank
    # R1 = [4, 6, 8]
    # R2 = [10, 15]
    # R3 = [20, 40]

    R1 = [2]
    R2 = [20]
    R3 = [30] 
    # best param[16, 1]
    rankDict = {'Tucker': generateRankCombinationTucker(R1, R2, R3), 'Tensor Train': generateRankCombinationTT(R2, R3),
                'CP': [2**i for i in range(4)], 'rank-n': [2**i for i in range(4)]}
    rankList = rankDict[method]
    print('use decomposition method', method)

    # set svm param
    # sigma used to rbf kernel
    # gamma = [2**(i) for i in range(9)]
    # sigma = [2**(i) for i in range(9)]
    gamma = [2] 
    sigma = [2] 
    if kernel == 'rbf':
        svmParaList = generateSVMParamCombination(sigma, gamma)
    else: 
        svmParaList = generateSVMParamCombination(sigma)

    model_tucker = 'lsstm'
    model_svm = 'svm'
    # rootPath为测试数据所在路径
    rootPath = './testData/'
    
    # 计算选取的数据量
    useDataRatio = dataRatio
    
    for i, idx in enumerate(idList):
        X = np.load(rootPath + f'{idx}_spectrum.npy')
        Y = np.load(rootPath + f'{idx}_label.npy')

        totalNum = X.shape[0]

        # 将 X 和 Y 按 train_size 的比例划分为训练集和测试集
        if useDataRatio != 1:
            X, x_test, Y, y_test = train_test_split(X, Y, test_size=1-useDataRatio, random_state=122914, stratify=Y) 

        useNum = X.shape[0]
        print('total num: ', totalNum, 'useNum: ', useNum)

        stmDict[str(idx)] = {'acc': 0, 'param': []}
        lsstmDict[str(idx)] = {'acc': 0, 'param': []}

        for rank in rankList:
            
            decomposetionEstimator = TensorDecompositionEstimate(method=method, rank=rank)

            dataloader = iEEG_fMRIDataLoader(X, Y, decompositionEstimator=decomposetionEstimator)
            for svmParam in svmParaList:
                print(svmParam)
                try:
                    lsstm = LSSTM(gamma=svmParam[0]*256, kernel_=kernel, sigma=svmParam[1]*256)
                    tmp = testModel(lsstm, dataloader)

                    if lsstmDict[str(idx)]['acc'] < np.mean(tmp):
                        lsstmDict[str(idx)]['acc'] = np.mean(tmp)
                        lsstmDict[str(idx)]['param'].append(svmParam)

                    stm = SVC(kernel=kernel)
                    tmp = testModel(stm, dataloader, flatten=True)

                    if stmDict[str(idx)]['acc'] < np.mean(tmp):
                        stmDict[str(idx)]['acc'] = np.mean(tmp)
                        stmDict[str(idx)]['param'].append(svmParam)

                except np.linalg.LinAlgError:
                    print('SVD did not converge')
                    continue
                except Exception as e:
                    print('raise a exception')
                    print(e)
                    continue

    print(lsstmDict)
    print(stmDict)

    # json.dump(lsstmDict, open('./' + method + '_' + kernel + '_' + str(dataRatio) + 'lsstmACC.json', 'w'))
    # json.dump(stmDict, open('./' + method + '_' + kernel + '_' + str(dataRatio) + 'stmACC.json', 'w'))

def main():
    testiEEG_fMRI(kernelIndex=2, methodIndex=0)



if __name__ == '__main__':
    main()