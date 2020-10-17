#!/usr/bin/python
# forked and refined from todd9527

# from sets import Set
import random
import math
import numpy as np
import copy as cp

K_VALUE = 4

class Graph:
    def __init__(self, vertices, edges):  # 定义图类，包括节点，边
        self.vertices = vertices
        self.edges = edges

class Network:
    def __init__(self):  # 创建边集，点集
        self.edges = set()
        self.nodes = set()
    def addEdge(self, edge):  # 增加边
        self.edges.add(edge)
    def addNode(self, node):  # 增加点
        self.nodes.add(node)
    def removeEdge(self, edge):  # 移除边
        self.edges.remove(edge)
    def isEdge(self, edge):  # 判断图是否包含边
        return self.edges.contains(edge)
    def isNode(self, node):  # 判断图是否包含点
        return self.nodes.contains(node)
    def toGraph(self):  # 返回图：包括节点列表和边列表
        return Graph(list(self.nodes), list(self.edges))

# Takes in a pair of nodes and returns them in the order we standardize
# for edges.
def makeEdge(n1, n2):  # 构造边，数小在前，数大在后
    return (min(n1, n2), max(n1, n2))

def Getdegree(graph): # The DP Algorithm  利用动态规划算法实现图匿名并返回度序列

    print("Getdegree")

    # cost of setting degree of nodes startIndex to endIndex to d* (median val)

    # Retrieve degrees of graph
    degrees = [ 0 for _ in range(len(graph.vertices))]   # degree列表所有元素都设为0，长度为图节点个数
    for edge in graph.edges:  # 图中各边：两端节点度数各+1
        if graph.vertices[0] == 0:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
        else:
            degrees[edge[0] - 1] += 1
            degrees[edge[1] - 1] += 1
    print(sorted(degrees , reverse=True))  # 降序排列原图度序列
    return sorted(degrees , reverse=True)

# Calculation of the function I that calculates the cost of the operation
def Icalc(T):
    valMax = T[0]  # 序列是降序排列，所以T[0]是序列的第一个值，也是序列中的最大值
    res = 0
    for e in T:
        res += valMax - e  # 成本=Σ(每个值-匿名值)
    return res

# 在d1处，计算两种分组方法的成本值
# Calculation of the cost cmerge
def cMerge(T, d1, k):  # cMerge参数:cMerge(数组，初始位置值，扩展组长度)
    res = d1 - T[k] + Icalc(T[k + 1:min(len(T), 2 * k + 1)])  # T[k + 1:2 * k + 1)]有k个值为[k+1,2k]
    # T[i，j]，取出数组中的值时就会从数组下标i（包括）一直取到下标j（不包括j）
    # T[k + 1:min(len(T), 2 * k + 1)]，取值从k+1到T长度/2k+1两者中的较小值
    return res


# Calculation of the cost cnew
def cNew(T, k):  # cNew参数:cNew(数组，扩展组长度)
    t = T[k:min(len(T), 2 * k)]  # ×可能是2k-2，因为原文中是d[k+1,2k],2(k+1)-2=2k,所以是2k-2
    res = Icalc(t)
    return res


# Recursive function of the greedy algorithm that returns an anonymized sequence of degrees
# PARAMETERS
# arrayDegrees : array of the ordered degrees # 有序度的序列数组
# kdegree : the k of the k-anonymization # k匿名的k值
# posIni : initial position (by default 0) # 初始位置posIni
# extension : extension by default equal to k # 扩展值，默认扩展为k
def greedyRecAlgorithm(arrayDegrees, kdegree, posIni, extension):  # 贪婪算法参数：greedyRecAlgorithm(度数数组, k值, 初始位置, 初始为k的扩展值)
    if (posIni + extension >= len(arrayDegrees) - 1):  # 如果初始位置+扩展超出了总长度
        for i in range(posIni, len(arrayDegrees)):  # 从初始位置到数组尽头
            arrayDegrees[i] = arrayDegrees[posIni]  # 将中间所有的值设为初始位置值arrayDegrees[posIni]
        return 0
    else:                                       # 如果初始位置+扩展没有超出了总长度
        d1 = arrayDegrees[posIni]  # 将初始位置的值赋给d1
        cmerge = cMerge(arrayDegrees, d1, posIni + extension)  # 计算初始位置扩展k的合并成本
        cnew = cNew(arrayDegrees, posIni + extension)  # 计算初始位置作为新组第一个值的新建成本
        if (cmerge > cnew):  # new groups starts at index k+1 #  合并成本高，需要开新组
            for i in range(posIni, posIni + extension):  # 扩展新组中的每一个点度数都设为d1(初始位置的值)
                arrayDegrees[i] = d1
            greedyRecAlgorithm(arrayDegrees, kdegree, posIni + extension, kdegree)  # 位置设为k以后，重新递归
        else:  # we merge the k+1th and starts at k+2，合并k+1成本低，新组从k+2开始判断
               # 先不管posIni + extension位置上的数，当需要合并的时候，自然会将其k匿名化
            greedyRecAlgorithm(arrayDegrees, kdegree, posIni, extension + 1)

# Backtrack function to construct the correct array of degrees giving the array of couples that is return by the dynamic programming algorithm
# 根据由动态规划算法返回的数组对(即DP里面由成本和tag组成的couple)，回溯功能可构建正确的度数数组
# PARAMETERS
# arrayDegrees : array of the ordered degrees 降序数组
# array of couples that permit to reconstruct the correct sequence of degrees anonymized
def backtrackFunction(arrayDegrees, tabCouples):  # 回溯得到k匿名序列，tabCouples是(度序列成本，度序列位置)那个列表
    backtrackCurrent = tabCouples[-1]  # tabCouples[-1]取列表最后一个元素，同理tabCouples[-2]取列表倒数第二个元素
    maxRange = len(arrayDegrees)  # 取arrayDegrees的长度
    backtrack = True  # 回溯标记设为正
    while backtrack:  # 当回溯标记为正时(此时backtrackCurrent的值为)
        valueToApply = arrayDegrees[backtrackCurrent[1] + 1]  # 取tSave后面那个元素设为vTA
        for i in range(backtrackCurrent[1] + 1, maxRange):  # 从tSave后面那个位置到最后
            arrayDegrees[i] = valueToApply  # 将其中元素统统设为vTA
        maxRange = backtrackCurrent[1] + 1  # 将最后的位置改成tSave后面那个位置(循环时不包括)
        if (backtrackCurrent[1] == -1):  # 如果回溯标记到了度序列中的第一组
            backtrack = False  # 将标记设为False,结束匿名
        backtrackCurrent = tabCouples[backtrackCurrent[1]]  #bC设为前面一个组最后一个元素的的(成本，位置)对儿


# Dynammic Programming algorithm that returns an anonymized sequence of degrees
# PARAMETERS
# arrayDegrees : array of the ordered degrees
# kdegree : the k of the k-anonymization
# empty array that will be filled with couples that permit to reconstruct the correct sequence of degrees anonymized
def DPGraphAnonymization(arrayDegrees, k, array):  # 参数(数组，k值，空数组)
    for i in range(1, len(arrayDegrees) + 1):  # 列表从头到尾，列表索引超出范围，应该是(0,len(arrayDegrees))
        if i < 2 * k:  # 如果i的位置小于2k
            array.append((Icalc(arrayDegrees[0:i]), -1))  # 数组存放添加(0到i的成本,-1)作为一个元素存到数组中
        else:  # i的位置大于或等于2k
            minI = Icalc(arrayDegrees[0:i])  # 最小成本 = 0到i的成本
            tSave = -1  # tSave = -1 初始化记录分组最后一个元素的位置点
            for t in range(k, i - k + 1):  # 从k到最后留出k的地方(i-k+1)
                tmp = array[t - 1][0] + Icalc(arrayDegrees[t:i])  # array[][0]这个[0]是取元组中第一个位置上的元素
                minI = min(minI, tmp)
                if (minI == tmp):
                    tSave = t - 1
            array.append((minI, tSave))  #Python的if不用符号，只要对齐就有符号，在这里每次都后增
    return array  #得到一个(成本，位置)对

def constructGraph(degrees):
    # 该函数可用ig.Graph.Degree_Sequence(out=ListDegreesGreedy, method="vl")替代
    # 也不太行，Degree_Sequence输出的Graph格式并不是(vertices, edges)

    print ("constructGraph")

    edges = []
    if(sum(degrees) % 2 == 1):
        raise TypeError

    i = 1
    unseenIndices = set(range(len(degrees)))
    while True:
        # print("Adding node %d" % i)

        # Add edges to graph
        for degree in degrees:
            if degree < 0:
                raise RuntimeError

        index = random.choice(tuple(unseenIndices))

        d = degrees[index]
        degrees[index] = 0
        unseenIndices.remove(index)

        degDLargest = np.argsort(degrees)[::-1][:d]
        for vertex in degDLargest:
            edges.append(makeEdge(index, vertex))
            degrees[vertex] -= 1  # line 166 和 line 172 表明python函数会该边形参的实际值

        # See if graph is completed
        i += 1
        graphCompleted = not any(degrees)  # not any degrees != 0, all degree = 0
        if graphCompleted:
            return Graph(range(len(degrees)), edges) # Return completed graph

#  The greedyswap
def findBestSwap(inputGraph, anonymizedGraph):  # 输入分别为原始图与匿名图
#  匿名图已经经过MakeEdge的变换
    print("findBestSwap")

    # For computational purposes we only examien some subset of the edges
    # 算力原因进检验边的部分子集
    # in the graph.
    examineThreshold = math.log(len(anonymizedGraph.edges), 2)  # 计算以2为底，边个数的对数log2(len(anonymizedGraph.edges)
    numExamined = 0

    c = 0
    toAddEdges = ((-1, -1), (-1, -1))
    toRemoveEdges = ((-1, -1), (-1, -1))


    while numExamined < examineThreshold and c != 2:  # 检测数量 < 检测门限 & c = 2意味着找到了最好的交换
        [e1, e2] = random.sample(anonymizedGraph.edges, 2)  # 从边集中随机挑出两个组成边对[e1, e2]

        # if any of the nodes are the same, we won't consider this edge pair
        # 如果两条边中有重复节点，将不考虑这对边
        if any([i in e2 for i in e1]):  # e1，e2两条边中节点有重复
            print("Fail because of overlapped vertices")
            # print("HELLO MF")
            continue

    # 边交换，两条边本来是e1[0]e1[1]，e2[0]e2[1]，交换得到四条边e1[0]e2[0],e1[0]e2[1],e1[1]e2[0],e1[1]e2[1]，但只有0011和0110会真正交换
        swapSets = [[e1, e2, makeEdge(e1[0], e2[0]), makeEdge(e1[1], e2[1])],
                     [e1, e2, makeEdge(e1[0], e2[1]), makeEdge(e1[1], e2[0])]]
        for swap in swapSets:
            edge_already_in_anonymizedGraph = 0
            for edge in swap[2:]:  # 从每一个元组的第三个元素到最后
                if edge in anonymizedGraph.edges:  # 匿名边集中已包含该边
                    edge_already_in_anonymizedGraph = 1
                    print ("Fail because of Edges that are already exist in anonymized graph")
                    break
                    # continue
            if edge_already_in_anonymizedGraph == 1:
                continue

            def swapDifference(swaps, inputGraph):
                _c = 0  # 设_c = 0
                inputGraphEdges = list()
                for edge in inputGraph.edges:
                    if edge[0] < edge[1]:
                        inputGraphEdges.append(edge)
                    if edge[0] > edge[1]:
                        inputGraphEdges.append((edge[1], edge[0]))
                if swaps[0] in inputGraphEdges:  # swaps组合中第一条边：原边1在原图的边中，_c-1，去掉了原图中有的边，减分
                    _c -= 1
                if swaps[1] in inputGraphEdges:  # swaps组合中第二条边：原边2在原图的边中，_c-1，去掉了原图中有的边，减分
                    _c -= 1
                if swaps[2] in inputGraphEdges:  # swaps组合中第三条边：新边1在原图的边中，_c+1，增加了原图中有的边，加分
                    _c += 1
                if swaps[3] in inputGraphEdges:  # swaps组合中第四条边：新边2在原图的边中，_c+1，增加了原图中有的边，加分
                    _c += 1
                return _c

            if swapDifference(swap, inputGraph) > c:  # 换边后有加分，即换边后比换边前更接近原图中的边集
                c = swapDifference(swap, inputGraph)
                toAddEdges = tuple(swap[2:])  # 将元组(交换边1，交换边2)赋给toAddEdges
                toRemoveEdges = tuple(swap[:2])  # 将元组(去除边1，去除边2)赋给toRemoveEdges
                print("Success")
            if swapDifference(swap, inputGraph) == c:
                print("Fail because no improvement")
            if swapDifference(swap, inputGraph) < c:
                print("Fail because c is negative")


        print("numExamined %d" % numExamined)  # 检测次数+1，迟早达到算力规定的界限examineThreshold = log2(边个数)
        numExamined += 1

    print("Returning a swap improvement of %d" % c)
    print("Edges:")
    print("toRemoveEdges:", toRemoveEdges)
    print("toAddEdges:", toAddEdges)

    return c, toRemoveEdges, toAddEdges  # 返回c值，去除边元组，增加边元组


#  outputs graph with same degree sequence as initialGraph and high similarity to inputGraph
def greedySwap(initialGraph, inputGraph):  # 输入分别为initialGraph初始匿名图, inputGraph原始图

    print("greedySwap")

    resultGraph = initialGraph  # 将最开始的匿名图设为结果图
    c, toRemoveEdge, toAddEdge = findBestSwap(inputGraph, initialGraph)  # 寻找最佳交换：输入为(原始图，初始匿名图)

    while c > 0:  # 当c > 0，匿名图的边更接近原图时
        for edge in toAddEdge:
            resultGraph.edges.append(edge)
        for edge in toRemoveEdge:
            resultGraph.edges.remove(edge)
        c, toRemoveEdge, toAddEdge = findBestSwap(inputGraph, resultGraph)

    return resultGraph

# def anonymize(graph):
#
#     print("anonymize")
#
#     # get k-anonymous degrees for each node
#     degreeSequence = Getdegree(graph)  # 得到原图降序度序列
#     while True:
#         try:
#             # construct graph with desired degree sequence
#             kAnonymousGraph = constructGraph(degreeSequence)
#             return greedySwap(kAnonymousGraph, graph)

if __name__ == "__main__":
    # Construct network object
    network = Network()

    # Open file
    file = open("karate2.txt", "r")

    # Go through header
    for _ in range(4):
        file.readline()

    # Convert file into network object
    line = file.readline()
    while (line != ""):
        tokens = line.split()
        # edge = (int(tokens[0]) - 1, int(tokens[1]) - 1)  # Store as ints
        edge = (int(tokens[0]), int(tokens[1]))  # Store as ints
        network.addEdge(edge)
        network.addNode(edge[0])
        network.addNode(edge[1])
        line = file.readline()
    file.close()
    print("Done reading file.")

    # Code to anonymize data (I suggest switching to smaller dataset)
    OriginalGraph = network.toGraph()
    OriginalGraphEdges = list()
    for edge in OriginalGraph.edges:
        if edge[0] < edge[1]:
            OriginalGraphEdges.append(edge)
        if edge[0] > edge[1]:
            OriginalGraphEdges.append((edge[1], edge[0]))
    print(OriginalGraphEdges)
    RawDegree = Getdegree(network.toGraph())
    k_value = 4
    print("Greedy or DP?")
    pattern = input("I want：")
    if pattern == "Greedy":
        arrayDegrees = np.array(RawDegree)
        arrayDegreesGreedy= cp.deepcopy(arrayDegrees)
        greedyRecAlgorithm(arrayDegreesGreedy, k_value, 0, k_value)
        AnonimizedDegree = arrayDegreesGreedy.tolist()
    if pattern == "DP":
        arrayDegrees = np.array(RawDegree)
        arrayDegreesDP = cp.deepcopy(arrayDegrees)
        maxDegreeIndex = 0
        arrayDynamic = []
        arrayDegreesDP = np.copy(arrayDegrees)
        tabCouples = DPGraphAnonymization(arrayDegreesDP, k_value, arrayDynamic)
        backtrackFunction(arrayDegreesDP, tabCouples)
        AnonimizedDegree = arrayDegreesDP.tolist()
    print("AnonimizedDegree before construct", AnonimizedDegree)
    kAnonymousGraph = constructGraph(AnonimizedDegree)
    print("AnonimizedDegree after construct", AnonimizedDegree)
    Getdegree(kAnonymousGraph)
    kAnonymousGraphEdges = list()
    for edge in kAnonymousGraph.edges:
        if edge[0] < edge[1]:
            kAnonymousGraphEdges.append(edge)
        if edge[0] > edge[1]:
            kAnonymousGraphEdges.append((edge[1], edge[0]))
    print(kAnonymousGraphEdges)
    numx = 0
    for edge1 in kAnonymousGraphEdges:
        for edge2 in OriginalGraphEdges:
            if edge1 == edge2:
                numx += 1
                # print(edge1)
    print(numx)
    print("ratio of original", numx/len(OriginalGraphEdges))
    # print("ratio of k-ano", numx/len(kAnonymousGraphEdges))
    print("END SIMILARITY")
    FinalGraph = greedySwap(kAnonymousGraph, OriginalGraph)
    FinalGraphEdges = list()
    for edge in FinalGraph.edges:
        if edge[0] < edge[1]:
            FinalGraphEdges.append(edge)
        if edge[0] > edge[1]:
            FinalGraphEdges.append((edge[1], edge[0]))
    print(FinalGraphEdges)
    numy = 0
    for edge1 in FinalGraphEdges:
        for edge2 in OriginalGraphEdges:
            if edge1 == edge2:
                numy += 1
    print(numy)
    print("ratio of original", numy / len(OriginalGraphEdges))
    print("END PROGRAM")



