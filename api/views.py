import json

from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from django.views.decorators.csrf import csrf_exempt

# @csrf_exempt
from api.modelRunner.simpleRunner import model_runner


def index(request):
    if request.method == 'POST':
        json_str = (request.body.decode('utf-8'))
        print(json_str)

        data = json.loads(json_str)
        model = model_runner(data);
        res = model.start()
        print(json.dumps(res))

        return HttpResponse(res)
    else:
        return HttpResponse("hello word")
