'''
Created on 2018年8月24日

@author: admin
'''
from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = [float(x) for x in curLine]
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    return mat0, mat1

def regLeaf(dataSet):
    return mean(dataSet[:, -1])

def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS, bestIndex, bestValue = inf, 0, 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
        
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    
    return bestIndex, bestValue   

def createTree(dataSet, leafType= regLeaf, errType= regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat is None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal']= val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)      
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0

def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1: n] = dataSet[:, 0: n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is sigular, cannot do inverse,\ntry increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1: n+1] = inDat
    return float(X * model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] <= tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
 
    return yHat

if __name__ == "__main__":
    # # 测试数据集
    # testMat = mat(eye(4))
    # print testMat
    # print type(testMat)
    # mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    # print mat0, '\n-----------\n', mat1

    # # 回归树
    # myDat = loadDataSet('input/9.RegTrees/data1.txt')
    # # myDat = loadDataSet('input/9.RegTrees/data2.txt')
    # # print 'myDat=', myDat
    # myMat = mat(myDat)
    # # print 'myMat=',  myMat
    # myTree = createTree(myMat)
    # print myTree

    # # 1. 预剪枝就是：提起设置最大误差数和最少元素数
    # myDat = loadDataSet('input/9.RegTrees/data3.txt')
    # myMat = mat(myDat)
    # myTree = createTree(myMat, ops=(0, 1))
    # print myTree

    # # 2. 后剪枝就是：通过测试数据，对预测模型进行合并判断
    # myDatTest = loadDataSet('input/9.RegTrees/data3test.txt')
    # myMat2Test = mat(myDatTest)
    # myFinalTree = prune(myTree, myMat2Test)
    # print '\n\n\n-------------------'
    # print myFinalTree

    # # --------
    # # 模型树求解
    # myDat = loadDataSet('input/9.RegTrees/data4.txt')
    # myMat = mat(myDat)
    # myTree = createTree(myMat, modelLeaf, modelErr)
    # print myTree

    # # 回归树 VS 模型树 VS 线性回归
    trainMat = mat(loadDataSet('./bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('./bikeSpeedVsIq_test.txt'))
    # # 回归树
    myTree1 = createTree(trainMat, ops=(1, 20))
    print(myTree1)
    yHat1 = createForeCast(myTree1, testMat[:, 0])
    print("--------------\n")
    # print yHat1
    # print "ssss==>", testMat[:, 1]
    print("回归树:", corrcoef(yHat1, testMat[:, 1],rowvar=0)[0, 1])

    # 模型树
    myTree2 = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    yHat2 = createForeCast(myTree2, testMat[:, 0], modelTreeEval)
    print(myTree2)
    print("模型树:", corrcoef(yHat2, testMat[:, 1],rowvar=0)[0, 1])
 
    # 线性回归
    ws, X, Y = linearSolve(trainMat)
    print(ws)
    m = len(testMat[:, 0])
    yHat3 = mat(zeros((m, 1)))
    for i in range(shape(testMat)[0]):
        yHat3[i] = testMat[i, 0]*ws[1, 0] + ws[0, 0]
    print("线性回归:", corrcoef(yHat3, testMat[:, 1],rowvar=0)[0, 1])
          