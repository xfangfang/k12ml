from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from django.template import loader


def index(request):
    node_num = 4
    context = {
        'title': 'DEMO',
        'nodeNum': node_num,
        'num': [i for i in range(node_num)],
    }
    return render(request, 'home/index.html', context)
