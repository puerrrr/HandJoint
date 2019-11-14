from hand_dataset import Task1Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import utils
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys


class PuerModel(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, channel_3, channel_4, channel_5, channel_6,
                 input_dim, layer1, layer2, output_dim,):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channel_2, channel_3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channel_3, channel_4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channel_4, channel_5, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(channel_5, channel_6, kernel_size=3, padding=1)

        self.fc1_1 = nn.Linear(input_dim, layer1)
        self.fc1_2 = nn.Linear(layer1, layer2)
        self.fc1_3 = nn.Linear(layer2, output_dim)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):

        relu1 = F.relu(self.conv1(x))
        pool1 = self.pool(relu1)
        relu2 = F.relu(self.conv2(pool1))
        pool2 = self.pool(relu2)
        relu3 = F.relu(self.conv3(pool2))
        pool3 = self.pool(relu3)
        relu4 = F.relu(self.conv4(pool3))
        pool4 = self.pool(relu4)
        relu5 = F.relu(self.conv5(pool4))
        pool5 = self.pool(relu5)
        relu6 = F.relu(self.conv6(pool5))
        pool6 = self.pool(relu6)
        features = pool6
        features = self.flatten(features)
        layer1 = F.relu(self.fc1_1(features))
        layer2 = F.relu(self.fc1_2(layer1))
        layer3 = self.fc1_3(layer2)

        return layer3

    @staticmethod
    def flatten(x):
        N = x.shape[0]
        return x.contiguous().view(N, -1)


def train(config):
    """
    Train a Puer model on our datasets

    Inputs:
    - config, a dictionary containing necessary parameters
    'gpu_id', 'batch_size', 'in_channel', 'channel_1-7', 'input_dim', 'outpue_dim'
    'lr_start', 'lr_decay_step', 'lr_decay_rate', 'result_dir'
    Returns: Nothing, but prints model accuracies during training and testing.
    """
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % config['gpu_id'] if use_cuda else "cpu")
    dtype = torch.float32

    train_dataset = Task1Dataset(mode='train', padding=20, bbs_length=100)
    train_generator = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=16)

    test_dataset = Task1Dataset(mode='test', padding=20, bbs_length=100)
    test_generator = DataLoader(test_dataset, batch_size=config['batch_size']*4, shuffle=False, num_workers=16)

    # initialize model and optimizer
    # channels: 1 4 8 16 32 64 128
    # figure size: 200 100 50 25 12 6 3
    model = PuerModel(config['in_channel'], config['channel_1'], config['channel_2'],
                      config['channel_3'], config['channel_4'], config['channel_5'], config['channel_6'],
                      config['input_dim'], config['layer1'], config['layer2'], config['output_dim'])
    optimizer = optim.SGD(model.parameters(), lr=config['lr_start'], momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay_step'], gamma=config['lr_decay_rate'])
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    # save model and results
    model_dir = '%s/model.pkl' % config['result_dir']
    # writer = SummaryWriter(log_dir='%s/depth_96' % config['result_dir'])

    # training
    best_loss, iter = 100, 0
    for epoch in range(config['epoch']):
        print('==========================================Epoch %i============================================' % epoch)

        # test
        print('--------------------------------------- Testing --------------------------------------')
        start_time = time.time()
        test_loss, preds, pose_test = [], [], []
        model.eval()  # dropout layers will work in eval mode
        with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
            for batch_img, batch_pose, batch_flag in test_generator:
                batch_pose = batch_pose.to(device=device, dtype=dtype)
                batch_img = batch_img.to(device=device, dtype=dtype)
                pred_pose = model(batch_img)
                # loss = torch.mean(torch.sum((pred_pose - batch_anns_test).pow(2), (1, 2)), 0)
                loss = torch.sum((pred_pose - batch_pose).pow(2)*batch_flag) / torch.sum(batch_flag)
                # t = torch.abs(pred_ratio - batch_ratio)
                # loss = torch.mean(torch.sum(torch.where(t < 0.01, 50 * t ** 2, t - 0.005), 1))

                test_loss.append(loss)
                preds.append(pred_pose)
                pose_test.append(batch_pose)

        preds_pose = torch.cat(preds, dim=0)
        pose_test = torch.cat(pose_test, dim=0)
        avg_error = torch.sqrt(torch.sum((pose_test - preds_pose) ** 2, dim=2))
        test_loss = torch.stack(test_loss, dim=0)

        end_time = time.time()
        if best_loss > torch.mean(test_loss):
            torch.save(model.state_dict(), model_dir)
            best_loss = torch.mean(test_loss)
            print('>>> Model saved as {}... best loss {:.4f}'.format(model_dir, best_loss))
        print('>>> Testing loss: {:.4f}(best loss {:.4f}), time used: {:.2f}s.\n'
              '>>> Test average pixel difference: {:.6f} ({:.6f}) mm'
              .format(torch.mean(test_loss), best_loss, end_time - start_time,
                      torch.mean(avg_error), torch.std(avg_error)))

        # train
        print('--------------------------------------- Training --------------------------------------')
        model.train()
        start_time = time.time()
        train_loss, preds, pose_train = [], [], []
        for batch_img, batch_pose, batch_flag in train_generator:
            batch_pose = batch_pose.to(device=device, dtype=dtype)
            batch_img = batch_img.to(device=device, dtype=dtype)
            pred_pose = model(batch_img)
            loss = torch.sum((pred_pose - batch_pose).pow(2) * batch_flag) / torch.sum(batch_flag)
            # loss = F.mse_loss(pred_pose, batch_anns)
            # t = torch.abs(pred_pose - batch_anns)
            # loss = torch.mean(torch.sum(torch.where(t < 0.01, 50 * t ** 2, t - 0.005), 1), 0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            iter += 1

            preds.append(pred_pose)
            train_loss.append(loss)
            pose_train.append(batch_pose)
        #
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # scheduler.step()
        # iter += 1

        preds = torch.cat(preds, dim=0)  # N*2
        pose_train = torch.cat(pose_train, dim=0)  # N*2
        avg_error = torch.sqrt(torch.sum((pose_train - preds) ** 2, dim=2))
        train_loss = torch.stack(train_loss, dim=0)
        end_time = time.time()
        print(avg_error)
        print(preds)
        print('>>> [epoch {:2d}/ iter {:6d}] Training loss: {:.4f}, lr: {:.6f},\n'
              '    average pixel difference: {:.6f} ({:.6f}) mm, time used: {:.2f}s.'
              .format(epoch, iter, torch.mean(train_loss),
                      scheduler.get_lr()[0], torch.mean(avg_error), torch.std(avg_error), end_time - start_time))


def test(config):
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % config['gpu_id'] if use_cuda else "cpu")
    dtype = torch.float32

    test_dataset = Task1Dataset(mode='test', padding=20, bbs_length=100)
    test_generator = DataLoader(test_dataset, batch_size=4*config['batch_size'], shuffle=False, num_workers=4)

    # initialize model and optimizer
    model = PuerModel(config['in_channel'], config['channel_1'], config['channel_2'],
                      config['channel_3'], config['channel_4'], config['channel_5'],
                      config['channel_6'],  config['input_dim'], config['layer1'],
                      config['layer2'], config['output_dim'])
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    # load trained model
    model_dir = '%s/model.pkl' % config['result_dir']
    if os.path.exists(model_dir):
        model.load_state_dict(torch.load(model_dir))
        print('Model is loaded from %s' % model_dir)
    else:
        print('[Error] cannot find model file.')

    print('--------------------------------------- Testing --------------------------------------')

    start_time = time.time()
    preds, ratio_test = [], []
    model.eval()  # dropout layers will work in eval mode
    with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
        for batch_ratio, batch_img in test_generator:
            batch_ratio = batch_ratio.to(device=device, dtype=dtype)
            batch_img = batch_img.to(device=device, dtype=dtype)

            pred_ratio = model(batch_img)

            ratio_test.append(batch_ratio)
            preds.append(pred_ratio)

    preds = torch.cat(preds, dim=0)  # [N, 2]
    ratio_test = torch.cat(ratio_test, dim=0)

    avg_error = torch.abs(preds - ratio_test)
    end_time = time.time()
    print('>>> Test average pixel: {:.6f} ({:.6f}) mm, time used: {:.2f}s.'
          .format(torch.mean(avg_error), torch.std(avg_error), end_time - start_time))
    preds = preds.detach().cpu().numpy()
    with open('/home/data/hands2019_challenge/task1/predicted_instances.pkl', 'wb') as f:
        pickle.dump(preds, f)


def get_config():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', '-mode', type=str, default='train')
    parser.add_argument('--gpu_id', '-id', type=str, default='2')
    parser.add_argument('--store_dir', '-sd', type=str, default='/hand_pose_data/nyu')
    # parser.add_argument('--result_dir', '-rd', type=str, default='/home/caiyi/PycharmProjects/noisy_pose_result')
    parser.add_argument('--result_dir', '-rd', type=str, default='/home/data/hands2019_challenge/task1')
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--channel_1', type=int, default=4)
    parser.add_argument('--channel_2', type=int, default=8)
    parser.add_argument('--channel_3', type=int, default=16)
    parser.add_argument('--channel_4', type=int, default=32)
    parser.add_argument('--channel_5', type=int, default=64)
    parser.add_argument('--channel_6', type=int, default=128)
    parser.add_argument('--channel_7', type=int, default=256)
    parser.add_argument('--input_dim', type=int, default=1152)
    parser.add_argument('--layer1', type=int, default=256)
    parser.add_argument('--layer2', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--epoch', '-epoch', type=int, default=20)
    parser.add_argument('--lr_start', '-lr', help='learning rate', type=float, default=0.0005)
    parser.add_argument('--lr_decay_rate', type=float, default=0.9)
    parser.add_argument('--lr_decay_step', type=float, default=20000)
    args = vars(parser.parse_args())
    utils.print_args(args)
    return args


def test1():
    with open('/Users/pengyuehou/Desktop/test/predicted_instances.pkl', 'rb') as f:
        ratio = pickle.load(f)
    with open('/Users/pengyuehou/Desktop/test/train_instances.pkl', 'rb') as f:
        instances = pickle.load(f)
    instances = instances[:(len(instances)//10)]
    bbs = list(zip(*instances))[2]
    # width, height
    bbs = np.asarray(list(zip(*bbs)))
    bbs_center = [(bbs[2]-bbs[0])//2, (bbs[3]-bbs[1])//2]
    pose_center = np.transpose(bbs_center) - 200 * np.asarray(ratio)[:len(instances)]
    pose_center = pose_center.astype(int)
    for idx in range(5):
        instance = instances[idx]
        hand_center = pose_center[idx]
        img = cv2.imread('%s/train_images/%s' % ('/Users/pengyuehou/Downloads/datasets/Task 1',
                                                 instance[0]), cv2.IMREAD_UNCHANGED)
        cropped_img = img[(hand_center[0]-100):(hand_center[0]+100), (hand_center[1]-100):(hand_center[1]+100)]
        plt.imshow(cropped_img, cmap='gray')
        plt.show()


def main():
    config = get_config()
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["OMP_NUM_THREADS"] = "1"
    if config['mode'] == 'train':
        train(config)
    elif config['mode'] == 'test':
        test(config)
    else:
        raise ValueError('mode %s errors' % config['mode'])
    # test1()


if __name__ == '__main__':
    main()


