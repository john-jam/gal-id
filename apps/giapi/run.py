import os
import uvicorn
import psutil
from tensorflow import config
from giapi.config import CONFIG


def main():
    print('GAL-iD api')

    if CONFIG.enable_gpu:
        print(f'Configure Tensorflow to run on GPU')
        gpu_devices = config.list_physical_devices('GPU')
        for device in gpu_devices:
            config.set_logical_device_configuration(device, [
                config.LogicalDeviceConfiguration(memory_limit=CONFIG.gpu_memory_limit)])
            print(f'GPU {device.name} enabled with memory limit {CONFIG.gpu_memory_limit}')
    else:
        print(f'Configure Tensorflow to run on CPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    num_workers = psutil.cpu_count()
    uvicorn.run(
        "giapi.app:app",
        host='0.0.0.0',
        reload=CONFIG.debug,
        workers=num_workers
    )


if __name__ == "__main__":
    main()
