# -*- coding: utf-8 -*-
import numpy as np
import random
from sklearn.datasets import load_iris, load_boston, load_diabetes, load_breast_cancer, load_linnerud, load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve, roc_auc_score


class Node:
    def __init__(self, jsonNode):
        # state:0未开始 1等待左侧数据 2等待右侧数据 3结束
        self.state = 0
        self.id = jsonNode['id']
        self.type = jsonNode['type']
        # 节点的输入参数
        self.config = {}
        # 节点的运行结果
        self.result = {}

    @staticmethod
    def buildNode(jsonNode):
        return globals()[jsonNode['type']](jsonNode)

    # 返回若干元组，每个元组代表一种输出内容，元组由输出对象列表、输出内容、下一节点输入位置（左右中）组成
    def run(self, config):
        return [([], {})]


class DataNode(Node):
    '''
    Wine 红酒产地：load_wine()
    Iris 花数据集：load_iris()
    乳腺癌数据集：load_breast_cancer()：简单经典的用于二分类任务的数据集
    糖尿病数据集：load_diabetes()：经典的用于回归认为的数据集，值得注意的是，这10个特征中的每个特征都已经被处理成0均值，方差归一化的特征值，
    波士顿房价数据集：load_boston()：经典的用于回归任务的数据集
    体能训练数据集：load_linnerud()：经典的用于多变量回归任务的数据集，其内部包含两个小数据集：Excise是对3个训练变量的20次观测（体重，腰围，脉搏），physiological是对3个生理学变量的20次观测（引体向上，仰卧起坐，立定跳远）
    '''

    def __init__(self, jsonNode):
        Node.__init__(self, jsonNode)
        self.out = jsonNode['out1']
        self.dataset = jsonNode['config']['dataset']

    def run(self, config={}):
        self.state = 2
        dataset = self.getDataSet(self.dataset)
        self.data = dataset.data
        self.target = dataset.target
        return [(self.out, {'data': self.data, 'target': self.target})]

    def getDataSet(self, dataset):
        if self.dataset == 'iris':
            return load_iris()
        elif self.dataset == 'boston':
            return load_boston()
        elif self.dataset == 'cancer':
            return load_breast_cancer()
        elif self.dataset == 'diabetes':
            return load_diabetes()
        elif self.dataset == 'linnerud':
            return load_linnerud()
        elif self.dataset == 'wine':
            return load_wine()

    def preview(self):
        # TODO 预览信息都需要什么

        res = self.run()
        # print(res[0][1]['data'])
        slice = random.sample(res[0][1]['data'].tolist(), 3)
        for i in slice:
            print(i)


class SplitNode(Node):
    def __init__(self, jsonNode):
        Node.__init__(self, jsonNode)
        self.outLeft = jsonNode['out1']
        self.outRight = jsonNode['out2']

    def run(self, config):
        self.state = 2
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(config['data'], config['target'], test_size=0.2, random_state=0)

        return [(self.outLeft, {'data': self.x_train, 'target': self.y_train}),
                (self.outRight, {'data': self.x_test, 'target': self.y_test})]


class FeatureNode(Node):
    pass


class LogisticRegressionNode(Node):
    def __init__(self, jsonNode):
        Node.__init__(self, jsonNode)
        self.out = jsonNode['out1']

    def run(self, config):
        self.state = 2
        self.classifier = LogisticRegression()
        self.classifier.fit(config['data'], config['target'])
        return [(self.out, {'data': config['data'], 'target': config['target'], 'classifier': self.classifier})]


class DecisionTreeNode(Node):
    def __init__(self, jsonNode):
        Node.__init__(self, jsonNode)
        self.out = jsonNode['out1']

    def run(self, config):
        self.state = 2
        self.classifier = tree.DecisionTreeClassifier()
        self.classifier.fit(config['data'], config['target'])
        return [(self.out, {'data': config['data'], 'target': config['target'], 'classifier': self.classifier})]


class NaiveBayesNode(Node):
    def __init__(self, jsonNode):
        Node.__init__(self, jsonNode)
        self.out = jsonNode['out1']
        self.algorithm = jsonNode['config']['algorithm']

    def run(self, config):
        self.state = 2
        self.classifier = self.select_algorithm(self.algorithm)
        self.classifier.fit(config['data'], config['target'])
        return [(self.out, {'data': config['data'], 'target': config['target'], 'classifier': self.classifier})]

    def select_algorithm(self, name):
        # GaussianNB,MultinomialNB,BernoulliNB
        # 提供了3中朴素贝叶斯分类算法：GaussianNB(高斯朴素贝叶斯)、MultinomialNB(多项式朴素贝叶斯)、BernoulliNB(伯努利朴素贝叶斯)
        if name == 'gaussian':
            return GaussianNB()
        elif name == 'multinomial':
            return MultinomialNB()
        elif name == 'bernoulli':
            return BernoulliNB()


class DecisionTreeClassifierNode(Node):
    def __init__(self, jsonNode):
        Node.__init__(self, jsonNode)
        self.out = jsonNode['out1']

    def run(self, config):
        self.state = 2
        self.classifier = tree.DecisionTreeClassifier()
        self.classifier.fit(config['data'], config['target'])
        return [(self.out, {'data': config['data'], 'target': config['target'], 'classifier': self.classifier})]


class LinearRegressionNode(Node):
    def __init__(self, jsonNode):
        Node.__init__(self, jsonNode)
        self.out = jsonNode['out1']

    def run(self, config):
        self.state = 2
        self.classifier = LinearRegression()
        self.classifier.fit(config['data'], config['target'])
        return [(self.out, {'data': config['data'], 'target': config['target'], 'classifier': self.classifier})]


class PredictNode(Node):
    def __init__(self, jsonNode):
        Node.__init__(self, jsonNode)
        self.out = jsonNode['out1']

    def run(self, config):
        if config['position'] == 0:
            self.trainIn = config
        elif config['position'] == 1:
            self.testIn = config

        if self.state == 0:
            self.state = 1
            return []
        elif self.state == 1:
            self.state = 2
            self.test_target_predict = self.trainIn['classifier'].predict(self.testIn['data'])

            return [(self.out, {'predict': self.test_target_predict, 'classifier': self.trainIn['classifier'],
                                'data_train': self.trainIn['data'], 'target_train': self.trainIn['target'],
                                'data_test': self.testIn['data'], 'target_test': self.testIn['target']})]


class ScoreNode(Node):
    def __init__(self, jsonNode):
        Node.__init__(self, jsonNode)

    def run(self, config):
        print('target test:', config['target_test'])
        print('target predict: ', config['predict'])

        print("accuracy:", accuracy_score(config['target_test'], config['predict']))
        # 大多数方法是处理二分类的
        # 联系小方 处理此处问题
        print("precision:",precision_score(config['target_test'],config['predict'],average=None))
        print("recall:",recall_score(config['target_test'],config['predict'],average=None))
        print("f1:",f1_score(config['target_test'],config['predict'],average=None))

        # print("roc_auc",roc_auc_score(config['target_test'],config['predict'],average=None))

        # fpr, tpr, thresholds = roc_curve(config['target_test'],config['predict'], pos_label=1)
        # print(fpr)
        # print(tpr)
        # print(thresholds)
        # print(auc(fpr, tpr))
        # print("auc:",auc(config['target_test'],config['predict']))
        return []


class Model:
    def __init__(self, jsonData):
        # 数据作为起点
        self.startNodes = []
        # id->node
        self.hashNode = {}
        # list
        self.nodeList = []
        self.paraseNode(jsonData)

    # 检查模型是否存在问题
    def modelIsRight(self):
        # 检查模型节点的唯一性
        if len(self.hashNode) != len(self.nodeList):
            print('节点重复ID')
            return False
        # TODO 不能有环（自己输入到自己）

        return True

    # 从json数据构建DAG
    def paraseNode(self, model):
        self.nodes = model['dag']['nodes']
        for node in self.nodes:
            n = Node.buildNode(node)
            self.hashNode[n.id] = n
            self.nodeList.append(n)
            if n.type == 'DataNode':
                self.startNodes.append(n)

    def getNodeFromId(self, id):
        if id in self.hashNode:
            return self.hashNode[id]
        return None
