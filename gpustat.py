import os
import json
import signal
import subprocess
import time

def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total, 2)
    used = round(meminfo.used, 2)
    free = round(meminfo.free, 2)
    return [total, used, free]


def get_GpuInfo(ip='127.0.0.1'):
    """
    :param ip: host
    :return: gpu利用率, gpu内存占用率, gpu温度, gpu数量
    """
    gpu_total_memory_list = []
    utilization_list = []
    timeout_seconds = 30

    gpu_cmd = 'ssh -o StrictHostKeyChecking=no %s gpustat --json' % ip  # 通过命令行执行gpustat --json
    gpu_info_dict = {}

    try:
        res = timeout_Popen(gpu_cmd, timeout_seconds)  # 超过30秒无返回信息,返回空值

        if res:
            res = res.stdout.read().decode()
            if not res:
                print('ssh %s 连接失败, 获取gpu信息失败' % ip)

            else:
                # gpu_info_dict = eval(res)
                gpu_info_dict = json.loads(res)  # str to json
                gpu_num = len(gpu_info_dict['gpus'])
    except:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    if gpu_info_dict:
        for i in gpu_info_dict['gpus']:
            gpu_total_memory_list.append(i["memory.total"])

            for p in i["processes"]:
                if p["username"] == "dlw":
                    utilization_gpu = float(p['gpu_memory_usage'])  # gpu利用率
                    utilization_list.append(str(utilization_gpu))

    else:
        print('{}: timeout > {}s, 获取gpu信息失败\n'.format(ip, timeout_seconds))
        utilization_list = ['-1']*4

    gpu_utilization = ','.join(utilization_list)

    return [gpu_utilization, gpu_total_memory_list]


# 处理popen等待超时:
def timeout_Popen(cmd, timeout=30):
    start = time.time()
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while process.poll() is None:  # 是否结束
        time.sleep(0.2)
        now = time.time()
        if now - start >= timeout:
            os.kill(process.pid, signal.SIGKILL)

            # pid=-1 等待当前进程的all子进程, os.WNOHANG 没有子进程退出,
            os.waitpid(-1, os.WNOHANG)
            return None

    return process

# [1659372581.371142, 1659372592.3839352, 1659372603.3841808, 1659372611.9927678, 1659372621.5426216, 1659372631.5114915, 1659372638.6062434, 1659372650.5401282, 1659372657.9112287, 1659372665.5063858, 1659372672.6043835]
# [[6442450944, 1713668096, 4728782848], [6442450944, 1913225216, 4529225728], [6442450944, 1920434176, 4522016768], [6442450944, 1911390208, 4531060736], [6442450944, 1913356288, 4529094656], [6442450944, 1913225216, 4529225728], [6442450944, 1913225216, 4529225728], [6442450944, 1913225216, 4529225728], [6442450944, 1913225216, 4529225728], [6442450944, 1913225216, 4529225728], [6442450944, 1913225216, 4529225728]]
