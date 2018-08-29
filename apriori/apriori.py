'''
Created on 2018年8月24日

@author: admin
'''
print(__doc__)
from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return C1
                
def scanD(D, Ck, minSupport):
    ssCnt,numItems = {},float(len(D))
    D = map(set, D)
    for tid in D:
        C = map(frozenset, Ck)
        for can in C:
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[: k-2]
            L2 = list(Lk[j])[: k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
#     D = map(set, dataSet)
    D = dataSet
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        if len(Lk) == 0:
            break
        L.append(Lk)
        k += 1
    return L, supportData

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] 
        if conf >= minConf:
            print (freqSet-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        print('Hmp1 = ', Hmp1)
        print('len(Hmp1) = ', len(Hmp1), ' len(freqSet) = ', len(freqSet))
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
  
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def testApriori():
    # 加载测试数据集
    dataSet = loadDataSet()
    print ('dataSet: ', dataSet)

    # Apriori 算法生成频繁项集以及它们的支持度
    L1, supportData1 = apriori(dataSet, minSupport=0.7)
    print ('L(0.7): ', L1)
    print ('supportData(0.7): ', supportData1)

    print ('->->->->->->->->->->->->->->->->->->->->->->->->->->->->')

    # Apriori 算法生成频繁项集以及它们的支持度
    L2, supportData2 = apriori(dataSet, minSupport=0.5)
    print ('L(0.5): ', L2)
    print ('supportData(0.5): ', supportData2)

def testGenerateRules():
    # 加载测试数据集
    dataSet = loadDataSet()
    print ('dataSet: ', dataSet)

    # Apriori 算法生成频繁项集以及它们的支持度
    L1, supportData1 = apriori(dataSet, minSupport=0.5)
    print ('L(0.7): ', L1)
    print ('supportData(0.7): ', supportData1)

    # 生成关联规则
    rules = generateRules(L1, supportData1, minConf=0.5)
    print ('rules: ', rules)

def main():
    # 测试 Apriori 算法
#     testApriori()

    # 生成关联规则
#     testGenerateRules()

    # 项目案例
    # 发现毒蘑菇的相似特性
    # 得到全集的数据
    dataSet = [line.split() for line in open("./mushroom.dat").readlines()]
    L, supportData = apriori(dataSet, minSupport=0.3)
    # 2表示毒蘑菇，1表示可食用的蘑菇
    # 找出关于2的频繁子项出来，就知道如果是毒蘑菇，那么出现频繁的也可能是毒蘑菇
    for item in L[1]:
        if item.intersection('2'):
            print (item)
#     
#     for item in L[2]:
#         if item.intersection('2'):
#             print (item)

if __name__ == "__main__":
    main()