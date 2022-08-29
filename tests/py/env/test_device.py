import torch

def test_cpu_available():
    import multiprocessing
    num_device = multiprocessing.cpu_count()
    print(f'CPU count: {num_device}')

def test_cuda_available():
    assert torch.cuda.is_available() == True
    num_device = torch.cuda.device_count()
    print(f'GPU count: {num_device}')

def test_device():
    test_cpu_available()
    test_cuda_available()

if __name__ == '__main__':
    test_device()