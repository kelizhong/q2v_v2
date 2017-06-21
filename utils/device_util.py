# coding=utf-8
"""util for devices. e.g. allocate GPU resource"""
import warnings
import subprocess
import math
import mxnet as mx
from utils.decorator_util import memoized


@memoized
def get_gpus_num():
    """
    return GPUs number
    """
    try:
        re = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
    except OSError:
        return 0
    return len([i for i in re.split('\n') if 'GPU' in i])


def get_devices(devices, device_mode='cpu', rank=0, hosts_num=1, workers_num=1):
    """ allocate GPU resource for training
    Parameters
    ----------
    devices: str
        devices id string, like '0,1,2,3' which represent the four devices
    device_mode: str
        device mode(cpu/gpu/gpu_auto), gpu_auto will divide the gpu devices(number: get_gpus_num())
        into int(math.ceil(workers_num / hosts_num)) part
    rank: int
        the rank of worker node
    hosts_num: int
        the number of hosts for local/distribute_stream training
    workers_num: int
        the number of workers in all hosts
    Returns
    -------
        the devices context list
    """
    assert device_mode in ['cpu', 'gpu', 'gpu_auto'], "device query2vec should in ['cpu', 'gpu', 'gpu_auto']"

    if device_mode == 'cpu':
        devs = mx.cpu() if devices is None or devices is '' else [
            mx.cpu(int(i)) for i in devices.split(',')]
    elif device_mode == 'gpu':
        devs = mx.gpu() if devices is None or devices is '' else [
            mx.gpu(int(i)) for i in devices.split(',')]
    else:
        if workers_num <= rank or workers_num < 1:
            raise ValueError("workers number must be larger than rank and 1")

        if hosts_num <= 0:
            raise ValueError("hosts number must be larger than one")

        if workers_num % hosts_num != 0:
            warnings.warn('the workers number is not divided by hosts number. Is it intended?')

        if workers_num > 8:
            warnings.warn(
                'the workers number larger than 8 which may make the training process slower. Is it intended?')

        gpus_num = get_gpus_num()
        workers_num_per_host = math.ceil(workers_num / hosts_num)

        workers_num_per_host = int(workers_num_per_host)

        gpus_num_per_work = gpus_num / workers_num_per_host

        start_index = (rank % workers_num_per_host) * gpus_num_per_work

        devs = [mx.gpu(int(i)) for i in xrange(start_index, start_index + gpus_num_per_work)]
    return devs

#print(get_devices(None, 'gpu_auto', 7, 2, 8))
