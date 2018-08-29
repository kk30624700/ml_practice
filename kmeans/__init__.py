from numpy import *
from time import sleep
import matplotlib
from matplotlib import pyplot as plt
from cmath import inf

def loadDataSet(fileName):
    dataSet = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataSet.append(fltLine)
    return dataSet

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataMat, k):
    m, n = shape(dataMat)
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataMat[:, j])
        rangeJ = float(max(dataMat[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids

def kMeans(dataMat, k, distMeans=distEclud, createCent=randCent):
    m, n = shape(dataMat)
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataMat, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j, :], dataMat[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        for cent in range(k):
            pstInClust = dataMat[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(pstInClust, axis=0)
    return centroids, clusterAssment

def biKmeans(dataMat, k, distMeans=distEclud):
    m, n = shape(dataMat)
    clusterAssment = mat(zeros((m, 2)))     
    centroid0 = mean(dataMat, axis=0).tolist()[0]
    centList =[centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeans(mat(centroid0), dataMat[j, :]) ** 2
    
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataMat[nonzero(clusterAssment[:, 0].A == i)[0], :] 
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeans)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print('sseSplit, and notSplit: ', sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] =  bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment
              
def distSLC(vecA, vecB):
    '''
    返回地球表面两点间的距离,单位是英里
    给定两个点的经纬度,可以使用球面余弦定理来计算亮点的距离
    :param vecA:
    :param vecB:
    :return:
    '''
    # 经度和维度用角度作为单位,但是sin()和cos()以弧度为输入.
    # 可以将江都除以180度然后再诚意圆周率pi转换为弧度
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0


def clusterClubs(fileName, imgName, numClust=5):
    '''
    将文本文件的解析,聚类以及画图都封装在一起
    :param fileName: 文本数据路径
    :param imgName: 图片路径
    :param numClust: 希望得到的簇数目
    :return:
    '''
    # 创建一个空列表
    datList = []
    # 打开文本文件获取第4列和第5列,这两列分别对应维度和经度,然后将这些值封装到datList
    for line in open(fileName).readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    # 调用biKmeans并使用distSLC函数作为聚类中使用的距离计算方式
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    # 创建一幅图和一个举行,使用该矩形来决定绘制图的哪一部分
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    # 构建一个标记形状的列表用于绘制散点图
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    # 使用imread函数基于一幅图像来创建矩阵
    imgP = plt.imread(imgName)
    # 使用imshow绘制该矩阵
    ax0.imshow(imgP)
    # 再同一幅图上绘制一张新图,允许使用两套坐标系统并不做任何缩放或偏移
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    # 遍历每一个簇并将它们一一画出来,标记类型从前面创建的scatterMarkers列表中得到
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        # 使用索引i % len(scatterMarkers)来选择标记形状,这意味这当有更多簇时,可以循环使用这标记
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        # 使用十字标记来表示簇中心并在图中显示
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()

if __name__=='__main__':
    datMat=mat(loadDataSet('testSet.txt'))
#     print(randCent(datMat, 2))
#     print(distEclud(datMat[0], datMat[1]))
#     print(kMeans(datMat, 4))
    print(biKmeans(datMat, 3)[0])    