'''
Created on 2018年8月27日

@author: admin
'''

print(__doc__)

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
        
    def inc(self, numOccur):
        self.count += numOccur
    
    def disp(self, ind=1):
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)
            

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
            #    ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat
    
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        if frozenset(trans) not in retDict.keys():
            retDict[frozenset(trans)] = 1
        else:
            retDict[frozenset(trans)] += 1
    return retDict

def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink is not None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode
    
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:    
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] is None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)   

def createTree(dataSet, minSup=1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del(headerTable[k])       
    
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
        
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable
    
def ascendTree(leafNode, prefixPath):
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)     

def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode is not None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats    

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[0])]
#     print('-----', sorted(headerTable.items(), key=lambda p: p[0]))
#     print('bigL=', bigL)
    
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
#         print('newFreqSet = ', newFreqSet, preFix)
        
        freqItemList.append(newFreqSet)
#         print('freqItemList=', freqItemList)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
#         print('condPattBases=', basePat, condPattBases)

        # 构建FP-tree
        myCondTree, myHead = createTree(condPattBases, minSup)
#         print('myHead=', myHead)
        # 挖掘条件 FP-tree, 如果myHead不为空，表示满足minSup {所有的元素+(value, treeNode)}
        if myHead is not None:
            print('conditional tree for: ', newFreqSet)
            myCondTree.disp(1)
#             print('\n\n\n')
            # 递归 myHead 找出频繁项集
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
#         print('\n\n\n')
        
    
if __name__ == "__main__":
    # rootNode = treeNode('pyramid', 9, None)
    # rootNode.children['eye'] = treeNode('eye', 13, None)
    # rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    # # 将树以文本形式显示
    # # print(rootNode.disp())

    # load样本数据
    simpDat = loadSimpDat()
    # print(simpDat, '\n')
    # frozen set 格式化 并 重新装载 样本数据，对所有的行进行统计求和，格式: {行：出现次数}
    initSet = createInitSet(simpDat)
#     print(initSet)

    # 创建FP树
    # 输入：dist{行：出现次数}的样本数据  和  最小的支持度
    # 输出：最终的PF-tree，通过循环获取第一层的节点，然后每一层的节点进行递归的获取每一行的字节点，也就是分支。然后所谓的指针，就是后来的指向已存在的
    myFPtree, myHeaderTab = createTree(initSet, 3)
#     myFPtree.disp()

    # 抽取条件模式基
    # 查询树节点的，频繁子项
#     print('x --->', findPrefixPath('x', myHeaderTab['x'][1]))
#     print('z --->', findPrefixPath('z', myHeaderTab['z'][1]))
#     print('r --->', findPrefixPath('r', myHeaderTab['r'][1]))

    # 创建条件模式基
    freqItemList = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItemList)
    print(freqItemList)

    # # 项目实战

    # # 新闻网站点击流中挖掘，例如：文章1阅读过的人，还阅读过什么？
    # parsedDat = [line.split() for line in open('input/12.FPGrowth/kosarak.dat').readlines()]
    # initSet = createInitSet(parsedDat)
    # myFPtree, myHeaderTab = createTree(initSet, 100000)

    # myFreList = []
    # mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreList)
    # print myFreList
 