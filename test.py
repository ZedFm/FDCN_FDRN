import argparse
import os
import numpy as np
import torch.utils.data
from torch import nn
from tqdm import tqdm
from model import FDCN,FDRN
from multiprocessing import cpu_count
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import torch.optim as optim
import data as d
import logging
import time



parser = argparse.ArgumentParser(description='FWmodel_testing_parameters')
parser.add_argument('--snr',default=18,choices=[12,18,24,30])
parser.add_argument('--data_length',default=256,type=int, help='length of the output data')
parser.add_argument('--scale',default=4,type=int,choices=[2,4],help='SR ratio')
parser.add_argument('--NUM_EPOCHS',default=32,type=int,help='epochs of training')
parser.add_argument('--test_path',default='test')
# parser.add_argument('--gt_path',default='')
# parser.add_argument('--noise_path',default='')
# parser.add_argument('--info_path',default='')
parser.add_argument('--model_select',default='0',type=int,choices=[0,1])
parser.add_argument('--log_path',default='log')

parser.add_argument('--model_save',default='model_save')
parser.add_argument('--optimizer',default='Adamax',choices=['SGD','Adamax','RMSProp','AdaGrad'])


net1 = FDCN(1, 6, 6, 9)
net2 = FDRN()
if __name__=='__main__':
    ppa =parser.parse_args()
    print(ppa.log_path)
    loss_fun = nn.MSELoss()

    print('CPU cores',cpu_count())
    net1 = FDCN(1,6,6,9)
    net2 = FDRN()
    if ppa.model_select ==0:
        train_model = net1
        model_path = os.path.join(ppa.model_save, 'FDCN', 'SNR' + str(ppa.snr),
                                  str(ppa.optimizer), 'best_model.pth')
    else:
        train_model = net2
        model_path = os.path.join(ppa.model_save, 'FDRN','SNR' + str(ppa.snr),
                                  str(ppa.optimizer), 'best_model.pth')
  #  model_path = os.path.join(ppa.model_save,'model'+str(ppa.model_select),'SNR'+str(ppa.snr),str(ppa.optimizer),'best_model.pth')
    train_model.load_state_dict(torch.load(model_path))
    train_model.eval()

    train_model.cuda()
    test_path = os.path.join(ppa.test_path,'SNR'+str(ppa.snr),'noise')
    test_files = os.listdir(test_path)
    test_result_save = os.path.join(ppa.test_path,'SNR'+str(ppa.snr),'model'+str(ppa.model_select),str(ppa.optimizer))
    if os.path.exists(test_result_save):
        print('result saving dir already exists')
    else:
        os.makedirs(test_result_save)
    idx = 0
    num = len(test_files)
    for text in tqdm(test_files):
        idx = idx+1
        save_path = os.path.join(test_result_save,text)
        np_data = np.loadtxt(os.path.join(test_path,text))
        np_data = np_data.reshape(1,1,256)
        test_data = torch.from_numpy(np_data)
        test_data = test_data.type(torch.FloatTensor)
        test_data = test_data.cuda()
        if ppa.model_select == 1:
            result_data = train_model(test_data, 2, 256)
        else:
            result_data = train_model(test_data)
        result_data = result_data.cpu()
        result_data = result_data.detach().numpy()
        result_data = result_data.reshape(1024,1)
        np.savetxt(save_path,result_data)
        print('Test and save Done! (%d/%d)'%(idx,num))

    #x_test = d.load_dataset(path, ppa.data_length, ppa.scale)
    #x_test_t = torch.from_numpy(x_test)


  #  for i in
  #  if ppa.model_select == 1:

