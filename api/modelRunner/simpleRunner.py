# -*- coding: utf-8 -*-
import json

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy

# boston=load_boston()
# x=boston.data
# y=boston.target

# state:0未开始 1等待部分数据 2运行结束
from .model import Model


class model_runner:
    def __init__(self, jsonData):
        self.model = Model(jsonData)

    def start(self):
        if (not self.model.modelIsRight()):
            return 'model wrong'

        for node in self.model.startNodes:
            self.run(node, {})

    # 节点 输入参数 下一节点输入位置（左右中 0 1 2）
    def run(self, node, config={}):
        res = node.run(config)
        # print(node)
        for outputs in res:
            nodes = outputs[0]
            config = outputs[1]
            for node in nodes:
                config['position'] = node['position']
                self.run(self.model.getNodeFromId(node['node']), config)


if __name__ == '__main__':
    # position 012 左右中 代表输入到下一个节点的具体位置
    # TODO feature节点还没有做

    model = {'dag': {'nodes': [
        {'id': 0, 'type': 'DataNode', 'out1': [{'node': 1, 'position': 2}], 'out2': [],
         'config': {'dataset': 'cancer'}},
        {'id': 1, 'type': 'SplitNode', 'out1': [{'node': 4, 'position': 2}], 'out2': [{'node': 5, 'position': 1}]},
        {'id': 4, 'type': 'NaiveBayesNode', 'out1': [{'node': 5, 'position': 0}], 'out2': [],
         'config': {'algorithm': 'gaussian'}},
        {'id': 5, 'type': 'PredictNode', 'out1': [{'node': 6, 'position': 2}], 'out2': []},
        {'id': 6, 'type': 'ScoreNode', 'out1': [], 'out2': []}
    ]
    }
    }
    runner = model_runner(model)
    runner.start()
