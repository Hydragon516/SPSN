DATA = {
    'data_root': "/SSD/minhyeok/dataset/RGBD-SOD",
    'val_dataset' : "NJU2K",
    'test_dataset' : "NJU2K, NLPR, RGBD135, STERE, SIP"
}

TRAIN = {
    'GPU': "0, 1",
    'epoch': 200,
    'learning_rate': 8e-5,
    'batch_size': 16,
    'num_superpixels': 100
}