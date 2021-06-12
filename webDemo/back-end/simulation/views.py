import json
import os
import sys
import zipfile
from time import sleep

from django.http import JsonResponse, FileResponse
import numpy as np

from audio_simulation.simulation import createroom

from django.shortcuts import render


# Create your views here.


def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = os.path.join(path_data, i)  # 当前文件夹的下面的所有东西的绝对路径
        os.remove(file_data)
        print('remove:' + file_data)


def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()


def noise_upload(request):
    # dir_path = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'up_noise/')
    # del_file(dir_path)
    files = request.FILES.getlist('noise_file')
    sleep(3)
    # print(files)
    for file in files:
        file_path = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'up_noise/', file.name)
        f = open(file_path, mode='wb')
        for i in file.chunks():
            f.write(i)
        f.close()
    response = JsonResponse({"status": 200, "msg": "noise上传成功"})
    response["Access-Control-Allow-Origin"] = '*'
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response


def target_upload(request):
    # while True:
    #     dir_path = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'up_target/')
    #     del_file(dir_path)
    #     if not os.listdir(dir_path):
    #         break;
    files = request.FILES.getlist('target_file')
    sleep(3)
    for file in files:
        file_path = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'up_target/', file.name)
        # print('write'+file.name)
        f = open(file_path, mode='wb')
        for i in file.chunks():
            f.write(i)
        f.close()
    response = JsonResponse({"status": 200, "msg": "target上传成功"})
    response["Access-Control-Allow-Origin"] = '*'
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response


def simulate(request):
    data = json.loads(request.body)
    n_mic = data.get('n_mic')
    n_target = data.get('n_target')
    n_noise = data.get('n_noise')
    r_mic_locs = data.get('mic_locs')
    interferer_locs = data.get('interferer_locs')
    target_locs = data.get('target_locs')
    room_dim = np.array(data.get('room_dim'))
    absorption = data.get('absorption')
    max_order = data.get('max_order')
    noisemodle = not data.get('noisemodle')


    outputpath = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'output')  # 混音文件输出位置
    sourcepath = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'up_target')  # 源音文件位置
    noisepath = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'up_noise')  # 噪音文件位置
    noises = []
    if noisemodle:
        print("有噪音")
        noises = [os.path.join(noisepath, filename) for filename in os.listdir(noisepath)]
    else:
        interferer_locs, noises = [], []

    sourcepaths = [os.path.join(sourcepath, filename) for filename in os.listdir(sourcepath)]
    createroom(sourcepaths, noises, mic_locs, target_locs, interferer_locs, room_dim, absorption, max_order, n_mic,
               outputpath, noisemodle)
    response = JsonResponse({'results': data})
    response["Access-Control-Allow-Origin"] = '*'
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response


def download(request):
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = cwd + "/output"
    out_path = cwd + "/simulation.zip"
    zipDir(path, out_path)  # 压缩两个文件
    response = FileResponse(open(out_path, "rb"))
    dir_noise = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'up_noise/')
    del_file(dir_noise)
    dir_source = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'up_target/')
    del_file(dir_source)
    dir_output = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'output/')
    del_file(dir_output)
    response['Access-Control-Allow-Origin'] = '*'
    response["Access-Control-Expose-Headers"] = "Contetnt-Disposition"
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename = "simulaiton.zip"'  # 下载压缩包
    return response
