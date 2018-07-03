# -*- coding: utf-8 -*-
import json

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy


# boston=load_boston()
# x=boston.data
# y=boston.target

# state:0未开始 1等待部分数据 2运行结束


class model_runner:
    def __init__(self, model):
        self.nodes = model['dag']['nodes']
        self.hashNode = {}
        self.startNodes = []
        self.parase_node(model)
        if self.model_is_right(model):
            pass
        else:
            print('model wrong')

    def start(self):
        out = []

        for node in self.startNodes:
            out.append(self.run(node))
        return json.dumps(out)

    def parase_node(self, model):

        # 数据作为起点
        for node in self.nodes:
            node['state'] = 0
            self.hashNode[node['id']] = node
            if node['type'] == 'data':
                self.startNodes.append(node)

    def get_node_from_id(self, id):
        return self.hashNode[id]

    # 检查模型是否存在问题
    def model_is_right(self, model):
        # 检查模型节点的唯一性
        if len(self.hashNode) != len(self.nodes):
            print('节点重复ID')
            return False
        # TODO 不能有环（自己输入到自己）

        return True

    def run(self, node, config={}):
        if node['type'] == 'data':
            node['state'] = 1
            boston = load_boston()
            x = boston.data
            y = boston.target
            return self.run(self.get_node_from_id(node['out'][0]), {'data': x, 'target': y})
        elif node['type'] == 'split':
            node['state'] = 1
            x_train, x_test, y_train, y_test = train_test_split(config['data'], config['target'], test_size=0.2,
                                                                random_state=0)
            return self.run(self.get_node_from_id(node['out'][0]), {'data': x_train, 'target': y_train}) + ' ' + \
                   self.run(self.get_node_from_id(node['out'][1]), {'data': x_test, 'target': y_test})
        elif node['type'] == 'feature':
            return str(node['id']) + '-' + str(numpy.size(config['target']))
        elif node['type'] == 'output':
            return str(node['id']) + '-' + str(numpy.size(config['target']))


if __name__ == '__main__':
    model = {'dag': {'nodes': [
        {'id': 0, 'type': 'data', 'out': [1]},
        {'id': 1, 'type': 'split', 'out': [4, 5]},
        {'id': 2, 'type': 'split', 'out': [4, 5]},
        {'id': 3, 'type': 'output', 'out': []},
        {'id': 4, 'type': 'output', 'out': []},
        {'id': 5, 'type': 'output', 'out': []},
    ]
    }
    }
    runner = model_runner(model)
