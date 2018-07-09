import json

from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from django.views.decorators.csrf import csrf_exempt

# @csrf_exempt
from .modelRunner import model_runner


def index(request):
    if request.method == 'POST':
        json_str = (request.body.decode('utf-8'))
        # print(json_str)

        # 目前收到的json文件不是最新可用版本,用下面的json来代替

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

        data = json.loads(json_str)
        model = model_runner(model)
        res = model.start()
        print(json.dumps(res))

        return HttpResponse(res)
    else:
        return HttpResponse("hello word")
