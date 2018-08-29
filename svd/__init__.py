from numpy import linalg as la
from numpy import *

def loadExData3():
    # 利用SVD提高推荐效果，菜肴矩阵
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
           [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]


def loadExData2():
    # 书上代码给的示例矩阵
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def loadExData():
    """
    # 推荐引擎示例矩阵
    return[[4, 4, 0, 2, 2],
           [4, 0, 0, 3, 3],
           [4, 0, 0, 1, 1],
           [1, 1, 1, 2, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0]]
    """
    # # 原矩阵
    # return[[1, 1, 1, 0, 0],
    #        [2, 2, 2, 0, 0],
    #        [1, 1, 1, 0, 0],
    #        [5, 5, 5, 0, 0],
    #        [1, 1, 0, 2, 2],
    #        [0, 0, 0, 3, 3],
    #        [0, 0, 0, 1, 1]]

    # 原矩阵
    return[[0, -1.6, 0.6],
           [0, 1.2, 0.8],
           [0, 0, 0],
           [0, 0, 0]]

def ecludSim(inA, inB):
    return 1.0/(1.0 + la.norm(inA-inB))

def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]

def cosSim(inA, inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5*(num/denom)

def standEst(dataMat, user, simMeans, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeans(dataMat[overLap, item], dataMat[overLap, j])
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal
 
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = mat(eye(4) * Sigma[: 4])
    
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    print('dataMat', shape(dataMat))
    print('U[:, :4]', shape(U[:, :4]))
    print('Sig4.I', shape(Sig4.I))
    print('VT[:4, :]', shape(VT[:4, :]))
    print('xformedItems', shape(xformedItems))
    
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        # 计算估计评分
        return ratSimTotal/simTotal
         

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)

def analyse_data(Sigma, loopNum=20):
    Sig2 = Sigma**2
    SigmaSum = sum(Sig2)
    for i in range(loopNum):
        SigmaI = sum(Sig2[: i+1])
        print('主成分：%s, 方差占比：%s%%' % (format(i+1, '2.0f'), format(SigmaI/SigmaSum*100, '4.2f')))

def imgLoadData(filename):
    my1 = []
    for line in open(filename).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        my1.append(newRow)
    return mat(my1)

def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1,end='')
            else:
                print(0,end='')
        print('')

def imgCompress(numSV=3, thresh=0.8):
    myMat = imgLoadData('./0_5.txt')
    print('***original matrix***')
    printMat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    analyse_data(Sigma, 20)
    SigRecon = mat(eye(numSV) * Sigma[:numSV])
    reconMat = U[:, :numSV]*SigRecon*VT[:numSV, :]
    print("****reconstructed matrix using %d singular values *****" % numSV)
    printMat(reconMat, thresh)
    
if __name__ == '__main__':
    # # 对矩阵进行SVD分解(用python实现SVD)
    # Data = loadExData()
    # print('Data:', Data)
    # U, Sigma, VT = linalg.svd(Data)
    # # 打印Sigma的结果，因为前3个数值比其他的值大了很多，为9.72140007e+00，5.29397912e+00，6.84226362e-01
    # # 后两个值比较小，每台机器输出结果可能有不同可以将这两个值去掉
    # print('U:', U)
    # print('Sigma', Sigma)
    # print('VT:', VT)
    # print('VT:', VT.T)

    # # 重构一个3x3的矩阵Sig3
    # Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    # print(U[:, :3] * Sig3 * VT[:3, :])

    """
    # 计算欧氏距离
    myMat = mat(loadExData())
    # print(myMat)
    print(ecludSim(myMat[:, 0], myMat[:, 4]))
    print(ecludSim(myMat[:, 0], myMat[:, 0]))
    # 计算余弦相似度
    print(cosSim(myMat[:, 0], myMat[:, 4]))
    print(cosSim(myMat[:, 0], myMat[:, 0]))
    # 计算皮尔逊相关系数
    print(pearsSim(myMat[:, 0], myMat[:, 4]))
    print(pearsSim(myMat[:, 0], myMat[:, 0]))
    """

    # 计算相似度的方法
#     myMat = mat(loadExData2())
    # print(myMat)
    # 计算相似度的第一种方式
#     print(recommend(myMat, 1, estMethod=standEst))
    # 计算相似度的第二种方式
#     print(recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim))

    # 默认推荐（菜馆菜肴推荐示例）
#     print(recommend(myMat, 2))

    """
    # 利用SVD提高推荐效果
    U, Sigma, VT = la.svd(mat(loadExData2()))
    print(Sigma)                 # 计算矩阵的SVD来了解其需要多少维的特征
    Sig2 = Sigma**2              # 计算需要多少个奇异值能达到总能量的90%
    print(sum(Sig2))             # 计算总能量
    print(sum(Sig2) * 0.9)       # 计算总能量的90%
    print(sum(Sig2[: 2]))        # 计算前两个元素所包含的能量
    print(sum(Sig2[: 3]))        # 两个元素的能量值小于总能量的90%，于是计算前三个元素所包含的能量
    # 该值高于总能量的90%，这就可以了
    """

    # 压缩图片
    imgCompress(2)