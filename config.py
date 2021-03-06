from easydict import EasyDict
from pathlib import Path


def get_config(is_training=True):
    config = EasyDict()
    config.data_dir = '/2_data/share/workspace/ljw/dataset/cell_detection/cell_dataset/data/'
    config.log_dir = 'log/'
    config.log_train_dir = config.log_dir + 'train/'
    config.log_test_dir = config.log_dir + 'test/'
    config.log_validation_dir = config.log_dir + 'validation/'
    config.tfboard_dir = config.log_dir + 'tfboard/'
    config.weight_dir = 'weight/'
    config.so_path='post_process/post_process.so'
    config.num_classes=1

    if is_training:
        config.is_training=True
        config.save_dir=config.weight_dir
        config.lr=1e-3
        config.max_iter = 1000000
        config.batch_size=4
        config.max_to_keep=5
        config.optimizer='adam' #'optimizer', 'adam', 'adadelta', 'momentum', 'rmsprop'

    else:
        config.testmodel="test1"     #"test1"  "test2"
        config.is_training=False
        config.newckpt_path=config.weight_dir+'/pb/18_0.ckpt'
        config.pb_path = config.weight_dir + '/pb/18_0.pb'
    return config






