import numpy as np
from numpy import dot, exp

import scipy.io as io
from scipy.spatial.distance import cdist

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

import tensorly as tl
from tensorly import tucker_to_tensor, cp_to_tensor, tt_to_tensor
from tensorly.decomposition import Tucker, TensorTrain, CP, tucker
from tensorly.decomposition import parafac

from kernel import get_kernel
from import_export import dump_model, load_model
from conversion import numpy_json_encoder

class LSSVC():

    def __init__(self, gamma=1, kernel='rbf', **kernel_params):
        # 超参数
        # gamma 松弛因子 正则化系数
        # kernel_ 核函数
        # kernel_params 核函数参数
        self.gamma = gamma
        self.kernel_ = kernel
        self.kernel_params = kernel_params

        # 模型参数
        self.alpha = None
        self.b = None
        self.x = None
        self.y = None
        self.y_labels = None

        # 获取核函数
        self.K = get_kernel(kernel, **kernel_params)
        
    # 根据推导得到的方程组进行求解
    def _optimize_parameters(self, X, y_values):
        sigma = np.multiply(y_values*y_values.T, self.K(X, X))
        
        A = np.block([
            [0, y_values.T],
            [y_values, sigma + self.gamma**-1 * np.eye(len(y_values))]
        ])
        B = np.array([0] + [1]*len(y_values))

        A_cross = np.linalg.pinv(A)

        solution = dot(A_cross, B)
        b = solution[0]
        alpha = solution[1:]

        return (b, alpha)

    def fit(self, X, y):
        y_reshaped = y.reshape(-1, 1) if y.ndim==1 else y

        self.x = X
        self.y = y_reshaped
        self.y_labels = np.unique(y_reshaped, axis=0)

        if len(self.y_labels) == 2:
            y_values = np.where((y_reshaped == self.y_labels[0]).all(axis=1), -1, +1)[:, np.newaxis]
            self.b, self.alpha = self._optimize_parameters(X, y_values)
        else:
            n_classes = len(self.y_labels)
            self.b = np.zeros(n_classes)
            self.alpha = np.zeros((n_classes, len(y_reshaped)))

            for i in range(n_classes):
                y_values = np.where((y_reshaped == self.y_labels[i]).all(axis=1), +1, -1)[:, np.newaxis]

                self.b[i], self.alpha[i] = self._optimize_parameters(X, y_values)

    def predict(self, X): 
        if self.alpha is None:
            raise Exception(
                "The model doesn't see to be fitted, try runing .fit() method first!"
            )
        
        X_reshape = X.reshape(1, -1) if X.ndim == 1 else X
        # print('use kernel', self.kernel_)
        KxX = self.K(self.x, X_reshape)

        if len(self.y_labels) == 2:
            # 二分类预测
            y_values = np.where((self.y == self.y_labels[0]).all(axis=1), -1, +1)[:, np.newaxis]
            y = np.sign(dot(np.multiply(self.alpha, y_values.flatten()), KxX) + self.b)
            
            # np.where(condition, x, y)
            # 如果condition在x中成立则返回x中的值，如果不成立返回y中的值
            # condition要能够广播到x和y的shape
            y_pred_labels = np.where(y==-1, self.y_labels[0], self.y_labels[1])
        else:
            y = np.zeros((len(self.y_labels), len(X)))
            for i in range(len(self.y_labels)):
                y_values = np.where((self.y == self.y_labels[i]).all(axis=1), +1, -1)[:, np.newaxis]
                y[i] = dot(np.multiply(self.alpha[i], y_values.flatten()), KxX) + self.b[i]
            
            predictions = np.argmax(y, axis=0)
            y_pred_labels = np.array([self.y_labels[i] for i in predictions])
        
        return y_pred_labels

    def dump(self, filepath='model', only_hyperparams=False):
        model_json = {
            'type': 'LSSVC',
            'hyperparameters': {
                'gamma': self.gamma,
                'kernel': self.kernel_,
                'kernel_params': self.kernel_params
            }
        }

        if (self.alpha is not None) and (not only_hyperparams):
            model_json['parameters'] = {
                'alpha': self.alpha,
                'b': self.b,
                'sv_x': self.x,
                'sv_y': self.y,
                'y_labels': self.y_labels
            }

        dump_model(model_dict=model_json, file_encoder=numpy_json_encoder, filepath=filepath)
    
    @classmethod
    def load(cls, filepath, only_hyperparams=False):
        model_json = load_model(filepath=filepath)

        if model_json['type'] != 'LSSVC':
            raise Exception(
                f"Model type '{model_json['type']}' doesn't match 'LSSVC'"
            )
        
        lssvc = LSSVC(gamma=model_json['gamma'], kernel=model_json['kernel'], **model_json['hyperparameters']['kernel_params'])

        if (model_json.get('parameters') is not None) and (not only_hyperparams):
            lssvc.alpha = np.array(model_json['parameters']['alpha'])
            lssvc.b = np.array(model_json['parameters']['b'])
            lssvc.x = np.array(model_json['parameters']['sv_x'])
            lssvc.y = np.array(model_json['parameters']['sv_y'])
            lssvc.y_labels = np.array(model_json['parameters']['y_labels'])
        
        return lssvc
    
# 进行张量分解
class LSSTCTucker():

    def __init__(self, gamma=1, kernel_='linear', **kernel_params):
        # 超参数
        # gamma 松弛因子 正则化系数
        # kernel_ 核函数
        # kernel_params 核函数参数
        self.gamma = gamma
        self.kernel_ = kernel_
        self.kernel_params = kernel_params
        self.rank = None

        self.testX = None

        # 模型参数
        self.alpha = None
        self.b = None
        self.x = None
        self.y = None
        self.y_labels = None

        # 获取核函数
        self.kernel_params = kernel_params
    

    def kernel(self, X, Y):
        # print('use kernel')
        # 一个一个算内积可以优化
        if self.kernel_ == 'linear':
            # print('use linear')
            numTensorX = X.shape[0]
            numTensorY = Y.shape[0]
            gramMatrix = np.zeros((numTensorX, numTensorY))
            for i in range(numTensorX):
                for j in range(numTensorY):
                    gramMatrix[i, j] = tl.tenalg.inner(X[i], Y[j])
            return gramMatrix
        elif self.kernel_ == 'poly':
            # print('use poly')
            numTensorX = X.shape[0]
            numTensorY = Y.shape[0]
            gramMatrix = np.zeros((numTensorX, numTensorY))
            for i in range(numTensorX):
                for j in range(numTensorY):
                    innerTensor = tl.tenalg.inner(X[i], Y[j])
                    gramMatrix[i, j] = (1 + innerTensor) ** (self.kernel_params.get('d', 3))
            return gramMatrix
        else:
            numTensorX = X.shape[0]
            numTensorY = Y.shape[0]
            gramMatrix = np.zeros((numTensorX, numTensorY))
            sigma = self.kernel_params.get('sigma', 1)
            for i in range(numTensorX):
                for j in range(numTensorY):
                    gramMatrix[i, j] = tl.norm(X[i]-Y[j], order=2)**2
            # print('use rbf')
            gramMatrix = np.exp(-gramMatrix/(sigma**2))
            return gramMatrix

    # 根据推导得到的方程组进行求解
    def _optimize_parameters(self, X, y_values):
        sigma = np.multiply(y_values*y_values.T, self.kernel(X, X))
        
        A = np.block([
            [0, y_values.T],
            [y_values, sigma + self.gamma**-1 * np.eye(len(y_values))]
        ])
        B = np.array([0] + [1]*len(y_values))

        A_cross = np.linalg.pinv(A)

        solution = dot(A_cross, B)
        b = solution[0]
        alpha = solution[1:]

        return (b, alpha)

    def fit(self, X, y, rank=None):
        y_reshaped = y.reshape(-1, 1) if y.ndim==1 else y
        numTensor = X.shape[0]
        tuckerTensor = np.zeros(X.shape)

        self.rank = rank

        for i in range(numTensor):
            decomTensor = tucker(X[i], rank=self.rank)
            estimateTensor = tucker_to_tensor(decomTensor)
            tuckerTensor[i] = estimateTensor

        self.x = tuckerTensor
        self.y = y_reshaped
        self.y_labels = np.unique(y_reshaped, axis=0)

        if len(self.y_labels) == 2:
            y_values = np.where((y_reshaped == self.y_labels[0]).all(axis=1), -1, +1)[:, np.newaxis]
            self.b, self.alpha = self._optimize_parameters(X, y_values)
        else:
            n_classes = len(self.y_labels)
            self.b = np.zeros(n_classes)
            self.alpha = np.zeros((n_classes, len(y_reshaped)))

            for i in range(n_classes):
                y_values = np.where((y_reshaped == self.y_labels[i]).all(axis=1), +1, -1)[:, np.newaxis]

                self.b[i], self.alpha[i] = self._optimize_parameters(X, y_values)

    def predict(self, X, haveInputTestData=False): 
        if self.alpha is None:
            raise Exception(
                "The model doesn't see to be fitted, try runing .fit() method first!"
            )
        
        # X has to be a tensor
        # X_reshape = X.reshape(1, -1) if X.ndim == 1 else X
        numTensor = X.shape[0]
        tuckerTensor = np.zeros(X.shape)

        for i in range(numTensor):
            decomTensor = tucker(X[i], rank=self.rank)
            estimateTensor = tucker_to_tensor(decomTensor)
            tuckerTensor[i] = estimateTensor
        X = tuckerTensor
        self.testX = X

        KxX = self.kernel(self.x, X)

        if len(self.y_labels) == 2:
            # 二分类预测
            y_values = np.where((self.y == self.y_labels[0]).all(axis=1), -1, +1)[:, np.newaxis]
            y = np.sign(dot(np.multiply(self.alpha, y_values.flatten()), KxX) + self.b)
            
            # np.where(condition, x, y)
            # 如果condition在x中成立则返回x中的值，如果不成立返回y中的值
            # condition要能够广播到x和y的shape
            y_pred_labels = np.where(y==-1, self.y_labels[0], self.y_labels[1])
        else:
            y = np.zeros((len(self.y_labels), len(X)))
            for i in range(len(self.y_labels)):
                y_values = np.where((self.y == self.y_labels[i]).all(axis=1), +1, -1)[:, np.newaxis]
                y[i] = dot(np.multiply(self.alpha[i], y_values.flatten()), KxX) + self.b[i]
            
            predictions = np.argmax(y, axis=0)
            y_pred_labels = np.array([self.y_labels[i] for i in predictions])
        
        return y_pred_labels

    def dump(self, filepath='model', only_hyperparams=False):
        model_json = {
            'type': 'LSSVC',
            'hyperparameters': {
                'gamma': self.gamma,
                'kernel': self.kernel_,
                'kernel_params': self.kernel_params
            }
        }

        if (self.alpha is not None) and (not only_hyperparams):
            model_json['parameters'] = {
                'alpha': self.alpha,
                'b': self.b,
                'sv_x': self.x,
                'sv_y': self.y,
                'y_labels': self.y_labels
            }

        dump_model(model_dict=model_json, file_encoder=numpy_json_encoder, filepath=filepath)
    
    @classmethod
    def load(cls, filepath, only_hyperparams=False):
        model_json = load_model(filepath=filepath)

        if model_json['type'] != 'LSSVC':
            raise Exception(
                f"Model type '{model_json['type']}' doesn't match 'LSSVC'"
            )
        
        lssvc = LSSVC(gamma=model_json['gamma'], kernel=model_json['kernel'], **model_json['hyperparameters']['kernel_params'])

        if (model_json.get('parameters') is not None) and (not only_hyperparams):
            lssvc.alpha = np.array(model_json['parameters']['alpha'])
            lssvc.b = np.array(model_json['parameters']['b'])
            lssvc.x = np.array(model_json['parameters']['sv_x'])
            lssvc.y = np.array(model_json['parameters']['sv_y'])
            lssvc.y_labels = np.array(model_json['parameters']['y_labels'])
        
        return lssvc
    

# 进行张量分解
class STuM():

    def __init__(self, gamma=1, kernel_='linear', **kernel_params):
        # 超参数
        # gamma 松弛因子 正则化系数
        # kernel_ 核函数
        # kernel_params 核函数参数
        self.gamma = gamma
        self.kernel_ = kernel_
        self.kernel_params = kernel_params
        self.rank = None

        self.testX = None

        # 模型参数
        self.alpha = None
        self.b = None
        self.x = None
        self.y = None
        self.y_labels = None

        # 获取核函数
        self.kernel_params = kernel_params
    

    def kernel(self, X, Y):
        # print('use kernel')
        # 一个一个算内积可以优化
        if self.kernel_ == 'linear':
            # print('use linear')
            numTensorX = X.shape[0]
            numTensorY = Y.shape[0]
            gramMatrix = np.zeros((numTensorX, numTensorY))
            for i in range(numTensorX):
                for j in range(numTensorY):
                    gramMatrix[i, j] = tl.tenalg.inner(X[i], Y[j])
            return gramMatrix
        elif self.kernel_ == 'poly':
            # print('use poly')
            numTensorX = X.shape[0]
            numTensorY = Y.shape[0]
            gramMatrix = np.zeros((numTensorX, numTensorY))
            for i in range(numTensorX):
                for j in range(numTensorY):
                    innerTensor = tl.tenalg.inner(X[i], Y[j])
                    gramMatrix[i, j] = (1 + innerTensor) ** (self.kernel_params.get('d', 3))
            return gramMatrix
        else:
            numTensorX = X.shape[0]
            numTensorY = Y.shape[0]
            gramMatrix = np.zeros((numTensorX, numTensorY))
            sigma = self.kernel_params.get('sigma', 1)
            for i in range(numTensorX):
                for j in range(numTensorY):
                    gramMatrix[i, j] = tl.norm(X[i]-Y[j], order=2)**2
            # print('use rbf')
            gramMatrix = np.exp(-gramMatrix/(sigma**2))
            return gramMatrix

    # 根据推导得到的方程组进行求解
    def _optimize_parameters(self, X, y_values):
        sigma = np.multiply(y_values*y_values.T, self.kernel(X, X))
        
        A = np.block([
            [0, y_values.T],
            [y_values, sigma + self.gamma**-1 * np.eye(len(y_values))]
        ])
        B = np.array([0] + [1]*len(y_values))

        A_cross = np.linalg.pinv(A)

        solution = dot(A_cross, B)
        b = solution[0]
        alpha = solution[1:]

        return (b, alpha)

    def fit(self, X, y, rank=None):
        y_reshaped = y.reshape(-1, 1) if y.ndim==1 else y
        numTensor = X.shape[0]
        tuckerTensor = np.zeros(X.shape)

        self.rank = rank

        for i in range(numTensor):
            decomTensor = tucker(X[i], rank=self.rank)
            estimateTensor = tucker_to_tensor(decomTensor)
            tuckerTensor[i] = estimateTensor

        self.x = tuckerTensor
        self.y = y_reshaped
        self.y_labels = np.unique(y_reshaped, axis=0)

        if len(self.y_labels) == 2:
            y_values = np.where((y_reshaped == self.y_labels[0]).all(axis=1), -1, +1)[:, np.newaxis]
            self.b, self.alpha = self._optimize_parameters(X, y_values)
        else:
            n_classes = len(self.y_labels)
            self.b = np.zeros(n_classes)
            self.alpha = np.zeros((n_classes, len(y_reshaped)))

            for i in range(n_classes):
                y_values = np.where((y_reshaped == self.y_labels[i]).all(axis=1), +1, -1)[:, np.newaxis]

                self.b[i], self.alpha[i] = self._optimize_parameters(X, y_values)

    def predict(self, X, haveInputTestData=False): 
        if self.alpha is None:
            raise Exception(
                "The model doesn't see to be fitted, try runing .fit() method first!"
            )
        
        # X has to be a tensor
        # X_reshape = X.reshape(1, -1) if X.ndim == 1 else X
        numTensor = X.shape[0]
        tuckerTensor = np.zeros(X.shape)

        for i in range(numTensor):
            decomTensor = tucker(X[i], rank=self.rank)
            estimateTensor = tucker_to_tensor(decomTensor)
            tuckerTensor[i] = estimateTensor
        X = tuckerTensor
        self.testX = X

        KxX = self.kernel(self.x, X)

        if len(self.y_labels) == 2:
            # 二分类预测
            y_values = np.where((self.y == self.y_labels[0]).all(axis=1), -1, +1)[:, np.newaxis]
            y = np.sign(dot(np.multiply(self.alpha, y_values.flatten()), KxX) + self.b)
            
            # np.where(condition, x, y)
            # 如果condition在x中成立则返回x中的值，如果不成立返回y中的值
            # condition要能够广播到x和y的shape
            y_pred_labels = np.where(y==-1, self.y_labels[0], self.y_labels[1])
        else:
            y = np.zeros((len(self.y_labels), len(X)))
            for i in range(len(self.y_labels)):
                y_values = np.where((self.y == self.y_labels[i]).all(axis=1), +1, -1)[:, np.newaxis]
                y[i] = dot(np.multiply(self.alpha[i], y_values.flatten()), KxX) + self.b[i]
            
            predictions = np.argmax(y, axis=0)
            y_pred_labels = np.array([self.y_labels[i] for i in predictions])
        
        return y_pred_labels

    def dump(self, filepath='model', only_hyperparams=False):
        model_json = {
            'type': 'LSSVC',
            'hyperparameters': {
                'gamma': self.gamma,
                'kernel': self.kernel_,
                'kernel_params': self.kernel_params
            }
        }

        if (self.alpha is not None) and (not only_hyperparams):
            model_json['parameters'] = {
                'alpha': self.alpha,
                'b': self.b,
                'sv_x': self.x,
                'sv_y': self.y,
                'y_labels': self.y_labels
            }

        dump_model(model_dict=model_json, file_encoder=numpy_json_encoder, filepath=filepath)
    
    @classmethod
    def load(cls, filepath, only_hyperparams=False):
        model_json = load_model(filepath=filepath)

        if model_json['type'] != 'LSSVC':
            raise Exception(
                f"Model type '{model_json['type']}' doesn't match 'LSSVC'"
            )
        
        lssvc = LSSVC(gamma=model_json['gamma'], kernel=model_json['kernel'], **model_json['hyperparameters']['kernel_params'])

        if (model_json.get('parameters') is not None) and (not only_hyperparams):
            lssvc.alpha = np.array(model_json['parameters']['alpha'])
            lssvc.b = np.array(model_json['parameters']['b'])
            lssvc.x = np.array(model_json['parameters']['sv_x'])
            lssvc.y = np.array(model_json['parameters']['sv_y'])
            lssvc.y_labels = np.array(model_json['parameters']['y_labels'])
        
        return lssvc

    
# 进行多种分解估计
class TensorDecompositionEstimate():

    def __init__(self, method=None, rank=None):
        self.method = method
        self.rank = rank

    def decompostionEstimate(self, X, method=None, rank=None):
        if method:
            self.method = method
        if rank:
            self.rank = rank
        else:
            rank = self.rank

        if self.method == 'Tucker':
            tuckerDecom = Tucker(rank=rank)
            decomTensor = tuckerDecom.fit_transform(X)
            estimateTensor = tucker_to_tensor(decomTensor)
            return estimateTensor
        elif self.method == 'Tensor Train':
            TTDecom = TensorTrain(rank=rank)
            decomTensor = TTDecom.fit_transform(X)
            estimateTensor = tt_to_tensor(decomTensor)
            return estimateTensor
        elif self.method == 'CP':
            CPDecom = CP(rank=rank)
            decomTensor = CPDecom.fit_transform(X)
            estimateTensor = tl.cp_to_tensor(decomTensor)
            return estimateTensor
        elif self.method == 'rank-n':
            rank_nDecom = parafac(X, rank=rank)
            return tl.cp_to_tensor(rank_nDecom)
        

# 直接传入分解好的模型进行计算
class LSSTM():

    def __init__(self, gamma=1, kernel_='linear', **kernel_params):
        # 超参数
        # gamma 松弛因子 正则化系数
        # kernel_ 核函数
        # kernel_params 核函数参数
        self.gamma = gamma
        self.kernel_ = kernel_
        self.kernel_params = kernel_params
        self.rank = None

        self.testX = None

        # 模型参数
        self.alpha = None
        self.b = None
        self.x = None
        self.y = None
        self.y_labels = None

        # 获取核函数
        self.kernel_params = kernel_params
    

    def kernel(self, X, Y):
        # print('use kernel')
        # 一个一个算内积可以优化
        if self.kernel_ == 'linear':
            # print('use linear')
            numTensorX = X.shape[0]
            numTensorY = Y.shape[0]
            # gramMatrix = np.zeros((numTensorX, numTensorY))
            # for i in range(numTensorX):
            #     for j in range(numTensorY):
            #         gramMatrix[i, j] = tl.tenalg.inner(X[i], Y[j])

            vectorX, vectorY = X.reshape((numTensorX, -1)), Y.reshape((numTensorY, -1))
            vectorCalGram = np.dot(vectorX, vectorY.T)
            # print((vectorCalGram-gramMatrix).sum())
            return vectorCalGram
        elif self.kernel_ == 'poly':
            # print('use poly')
            numTensorX = X.shape[0]
            numTensorY = Y.shape[0]
            gramMatrix = np.zeros((numTensorX, numTensorY))
            # for i in range(numTensorX):
            #     for j in range(numTensorY):
            #         innerTensor = tl.tenalg.inner(X[i], Y[j])
            #         gramMatrix[i, j] = (1 + innerTensor) ** (self.kernel_params.get('d', 3))

            vectorX, vectorY = X.reshape((numTensorX, -1)), Y.reshape((numTensorY, -1))
            vectorCalGram = (np.dot(vectorX, vectorY.T) + 1) ** (self.kernel_params.get('d', 3))
            # print((vectorCalGram-gramMatrix).sum())
            return vectorCalGram
        else:
            numTensorX = X.shape[0]
            numTensorY = Y.shape[0]
            sigma = self.kernel_params.get('sigma', 1)

            # gramMatrix = np.zeros((numTensorX, numTensorY))
            # for i in range(numTensorX):
            #     for j in range(numTensorY):
            #         gramMatrix[i, j] = tl.norm(X[i]-Y[j], order=2)**2
            # # print('use rbf')
            # gramMatrix = np.exp(-gramMatrix/(sigma**2))

            vectorX, vectorY = X.reshape((numTensorX, -1)), Y.reshape((numTensorY, -1))
            vectorCalGram = exp(-cdist(vectorX, vectorY) ** 2 / sigma ** 2)
            # print((vectorCalGram - gramMatrix).sum())

            return vectorCalGram

    # 根据推导得到的方程组进行求解
    def _optimize_parameters(self, X, y_values):
        sigma = np.multiply(y_values*y_values.T, self.kernel(X, X))
        
        A = np.block([
            [0, y_values.T],
            [y_values, sigma + self.gamma**-1 * np.eye(len(y_values))]
        ])
        B = np.array([0] + [1]*len(y_values))

        A_cross = np.linalg.pinv(A)

        solution = dot(A_cross, B)
        b = solution[0]
        alpha = solution[1:]

        return (b, alpha)

    def fit(self, X, y, rank=None):
        y_reshaped = y.reshape(-1, 1) if y.ndim==1 else y
        numTensor = X.shape[0]

        self.x = X
        self.y = y_reshaped
        self.y_labels = np.unique(y_reshaped, axis=0)

        if len(self.y_labels) == 2:
            y_values = np.where((y_reshaped == self.y_labels[0]).all(axis=1), -1, +1)[:, np.newaxis]
            self.b, self.alpha = self._optimize_parameters(X, y_values)
        else:
            n_classes = len(self.y_labels)
            self.b = np.zeros(n_classes)
            self.alpha = np.zeros((n_classes, len(y_reshaped)))

            for i in range(n_classes):
                y_values = np.where((y_reshaped == self.y_labels[i]).all(axis=1), +1, -1)[:, np.newaxis]

                self.b[i], self.alpha[i] = self._optimize_parameters(X, y_values)

    def predict(self, X, haveInputTestData=False): 
        if self.alpha is None:
            raise Exception(
                "The model doesn't see to be fitted, try runing .fit() method first!"
            )
        
        # X has to be a tensor
        # X_reshape = X.reshape(1, -1) if X.ndim == 1 else X
        numTensor = X.shape[0]
        self.testX = X

        KxX = self.kernel(self.x, X)

        if len(self.y_labels) == 2:
            # 二分类预测
            y_values = np.where((self.y == self.y_labels[0]).all(axis=1), -1, +1)[:, np.newaxis]
            y = np.sign(dot(np.multiply(self.alpha, y_values.flatten()), KxX) + self.b)
            
            # np.where(condition, x, y)
            # 如果condition在x中成立则返回x中的值，如果不成立返回y中的值
            # condition要能够广播到x和y的shape
            y_pred_labels = np.where(y==-1, self.y_labels[0], self.y_labels[1])
        else:
            y = np.zeros((len(self.y_labels), len(X)))
            for i in range(len(self.y_labels)):
                y_values = np.where((self.y == self.y_labels[i]).all(axis=1), +1, -1)[:, np.newaxis]
                y[i] = dot(np.multiply(self.alpha[i], y_values.flatten()), KxX) + self.b[i]
            
            predictions = np.argmax(y, axis=0)
            y_pred_labels = np.array([self.y_labels[i] for i in predictions])
        
        return y_pred_labels

    def dump(self, filepath='model', only_hyperparams=False):
        model_json = {
            'type': 'LSSVC',
            'hyperparameters': {
                'gamma': self.gamma,
                'kernel': self.kernel_,
                'kernel_params': self.kernel_params
            }
        }

        if (self.alpha is not None) and (not only_hyperparams):
            model_json['parameters'] = {
                'alpha': self.alpha,
                'b': self.b,
                'sv_x': self.x,
                'sv_y': self.y,
                'y_labels': self.y_labels
            }

        dump_model(model_dict=model_json, file_encoder=numpy_json_encoder, filepath=filepath)
    
    @classmethod
    def load(cls, filepath, only_hyperparams=False):
        model_json = load_model(filepath=filepath)

        if model_json['type'] != 'LSSVC':
            raise Exception(
                f"Model type '{model_json['type']}' doesn't match 'LSSVC'"
            )
        
        lssvc = LSSVC(gamma=model_json['gamma'], kernel=model_json['kernel'], **model_json['hyperparameters']['kernel_params'])

        if (model_json.get('parameters') is not None) and (not only_hyperparams):
            lssvc.alpha = np.array(model_json['parameters']['alpha'])
            lssvc.b = np.array(model_json['parameters']['b'])
            lssvc.x = np.array(model_json['parameters']['sv_x'])
            lssvc.y = np.array(model_json['parameters']['sv_y'])
            lssvc.y_labels = np.array(model_json['parameters']['y_labels'])
        
        return lssvc