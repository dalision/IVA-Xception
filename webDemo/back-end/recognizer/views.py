import os
import sys
import zipfile
import json

from django.http import JsonResponse, FileResponse
from audio_recognizer.separate_recognizer import main
from django.shortcuts import render
from audio_simulation.separation.twobirdseparation import two_seperate
from audio_simulation.separation.threebirdseparation import three_seperate

# Create your views here.


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



def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = os.path.join(path_data, i)  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data):  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)


def upload(request):
    dir_path = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'source/')
    del_file(dir_path)
    files = request.FILES.getlist('file')
    for file in files:
        file_path = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'source', file.name)
        f = open(file_path, mode='wb')
        for i in file.chunks():
            f.write(i)
        f.close()
    response = JsonResponse({"status": 200, "msg": "上传成功"})
    response["Access-Control-Allow-Origin"] = '*'
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response


def recognize(request):
    num_results = int(request.GET.get('num'))
    print(num_results)
    data_path = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'source/')
    root_path = 'D:/Code/BirdSong/audio_recognizer'
    print(data_path)
    result = main(root_path, data_path, num_results)
    res = []
    for index, i in enumerate(result):
        res.append({'id': index, 'name': i[0], 'prob': i[1]})
    dir_path = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'source/')
    del_file(dir_path)
    response = JsonResponse({'results': res})
    response["Access-Control-Allow-Origin"] = '*'
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response


def seperate(request):
    source_num = int(request.GET.get('source_num'))
    msg = ''
    if source_num == 2:
        two_seperate()
        msg = 'two file seperate success'
        print(msg)
    if source_num == 3:
        three_seperate()
        msg = 'three file seperate success'
        print(msg)
    response = JsonResponse({'msg': msg})
    response["Access-Control-Allow-Origin"] = '*'
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response


def download(request):
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = cwd + "/seperate_result"
    out_path = cwd + "/seperate.zip"
    zipDir(path, out_path)  # 压缩两个文件
    response = FileResponse(open(out_path, "rb"))
    dir_seperate = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'seperate_result/')
    del_file(dir_seperate)
    dir_source = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'seperate_source/')
    del_file(dir_source)
    response['Access-Control-Allow-Origin'] = '*'
    response["Access-Control-Expose-Headers"] = "Contetnt-Disposition"
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename = "seperate.zip"'  # 下载压缩包
    return response


def upload_seperate(request):
    dir_path = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'seperate_source/')
    del_file(dir_path)
    files = request.FILES.getlist('seperate_file')
    print(files)
    for file in files:
        file_path = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), r'seperate_source', file.name)
        f = open(file_path, mode='wb')
        for i in file.chunks():
            f.write(i)
        f.close()
    response = JsonResponse({"status": 200, "msg": "上传成功"})
    response["Access-Control-Allow-Origin"] = '*'
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response

