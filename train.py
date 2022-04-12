import argparse
import os
import numpy as np
import torch.utils.data
from torch import nn
from tqdm import tqdm
from model import FDCN, FDRN
from multiprocessing import cpu_count
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import torch.optim as optim
import data as d
import logging
import time

parser = argparse.ArgumentParser(description='FWmodel_training_parameters')
parser.add_argument('--snr',default=12,choices=[12,18,24,30])
parser.add_argument('--data_length',default=1024,type=int, help='length of the output data')
parser.add_argument('--scale',default=4,type=int,choices=[2,4],help='SR ratio')
parser.add_argument('--NUM_EPOCHS',default=32,type=int,help='epochs of training')
parser.add_argument('--train_path',default='train')
# parser.add_argument('--gt_path',default='')
# parser.add_argument('--noise_path',default='')
# parser.add_argument('--info_path',default='')
parser.add_argument('--model_select',default='1',type=int,choices=[0,1])
parser.add_argument('--log_path',default='log')
parser.add_argument('--val_rate',default='0.01')
parser.add_argument('--model_save',default='model_save')
parser.add_argument('--optimizer',default='Adamax',choices=['SGD','Adamax','RMSProp','AdaGrad'])
if __name__=='__main__':
    ppa =parser.parse_args()
    print(ppa.log_path)
    loss_fun = nn.MSELoss()
    NUM_POCHS = ppa.NUM_EPOCHS
    print('CPU cores',cpu_count())
    net1 = FDCN(1,6,6,9)
    net2 = FDRN()
    if ppa.model_select ==0:
        train_model = net1
        if os.path.exists(os.path.join(ppa.model_save,'FDCN','SNR'+str(ppa.snr),ppa.optimizer)):
            print('save_dir already exits')
        else:
            #os.mkdir(os.path.join(ppa.model_save,'model'+ str(ppa.model_select),'SNR'+str(ppa.snr),ppa.optimizer))
            os.makedirs(os.path.join(ppa.model_save, 'FDCN', 'SNR' + str(ppa.snr), ppa.optimizer))

        if os.path.exists(os.path.join(ppa.model_save,'FDCN','SNR'+str(ppa.snr),ppa.optimizer,'best_model.pth')):
            train_model.load_state_dict(torch.load(os.path.join(ppa.model_save,'FDCN','SNR'+str(ppa.snr),ppa.optimizer,'best_model.pth')))
        save_path = os.path.join(ppa.model_save, 'FDCN', 'SNR' + str(ppa.snr), ppa.optimizer)
        modelspath = os.path.join(save_path,'models_10')
        if os.path.exists(modelspath):
            print('models_save dir already exists ')
            pths = os.listdir(modelspath)
            if pths != []:
                pths.sort(key = lambda x: int(x[6:-4]))
                pth = pths[-1]
                num = int(pth[6:-4])
            else:
                num = 0
            print('it alread trained for %d epoch' %(num))
        else:
            os.mkdir(modelspath)
            num = 0


    else:
        train_model = net2

        if os.path.exists(
                os.path.join(ppa.model_save, 'FDRN', 'SNR' + str(ppa.snr), ppa.optimizer)):
            print('save_dir already exits')
        else:
            # os.mkdir(os.path.join(ppa.model_save,'model'+ str(ppa.model_select),'SNR'+str(ppa.snr),ppa.optimizer))
            os.makedirs(
                os.path.join(ppa.model_save, 'FDRN', 'SNR' + str(ppa.snr), ppa.optimizer))

        if os.path.exists(
                os.path.join(ppa.model_save, 'FDRN', 'SNR' + str(ppa.snr), ppa.optimizer,
                             'best_model.pth')):
            train_model.load_state_dict(torch.load(
                os.path.join(ppa.model_save, 'FDRN', 'SNR' + str(ppa.snr), ppa.optimizer,
                             'best_model.pth')))
        save_path = os.path.join(ppa.model_save, 'FDRN', 'SNR' + str(ppa.snr), ppa.optimizer)
        modelspath = os.path.join(save_path, 'models_10')
        if os.path.exists(modelspath):
            print('models_save dir already exists ')
            pths = os.listdir(modelspath)
            if pths != []:
                pths.sort(key=lambda x: int(x[6:-4]))
                pth = pths[-1]
                num = int(pth[6:-4])
            else:
                num = 0
            print('it alread trained for %d epoch' % (num))
        else:
            os.mkdir(modelspath)
            num = 0

    val_rate =1- np.float(ppa.val_rate)
    if ppa.optimizer =='Adamax':
        optimizer = optim.Adamax(train_model.parameters())
    elif ppa.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(train_model.parameters())
    elif ppa.optimizer == 'SGD':
        optimizer = optim.SGD(train_model.parameters(),lr=3e-4,momentum=0.9)
    elif ppa.optimizer == 'RMSProp':
        optimizer = optim.RMSprop(train_model.parameters())


    if torch.cuda.is_available():
        print('CUDA is Available')
        train_model.cuda()

    def log_string(str):
        logger.info(str)
        print(str)


    print('# Total parameters in the model', sum(param.numel() for param in train_model.parameters()))
    logger = logging.getLogger('Model')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s2.txt' % (ppa.log_path, '/snr_'+str(ppa.snr) +'_model_'+str(ppa.model_select) ))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')

#############start loading data############
    standpath = os.path.join(ppa.train_path,'SNR'+str(ppa.snr),'stand')
    noisepath = os.path.join(ppa.train_path,'SNR'+str(ppa.snr),'stand_n')
    infopath = os.path.join(ppa.train_path,'SNR'+str(ppa.snr),'SNR'+str(ppa.snr)+'_info.txt')

    x_stand = d.load_dataset(standpath,ppa.data_length,ppa.scale)
    x_noise = d.load_dataset(noisepath,ppa.data_length,ppa.scale)
    x_sn = np.hstack((x_stand, x_noise))
    y_target = d.load_target(infopath)
    m,n = np.shape(y_target)
    y_target = y_target.reshape(n,m)
    data_concat = np.hstack((x_sn, y_target))
###########shuffle
    np.random.shuffle(data_concat)


    length,_ = np.shape(data_concat)
    C = length*val_rate
    train_set_data = data_concat[0:int(length*val_rate),0:1280]
    train_set_target = data_concat[0:int(length*val_rate),1280:1295]

    val_set_data = data_concat[int(length*val_rate):length,0:1280]
    val_set_target = data_concat[int(length*val_rate):length,1280:1295]
    train_data_t =torch.from_numpy(train_set_data)
    train_target_t = torch.from_numpy(train_set_target)
    val_data_t = torch.from_numpy(val_set_data)
    val_target_t = torch.from_numpy(val_set_target)

    train_data = TensorDataset(train_data_t,train_target_t)
    val_data = TensorDataset(val_data_t,val_target_t)#########prepare train and val data

    train_loader = DataLoader(dataset=train_data,
                              batch_size=64,
                              shuffle=True
    )

    val_loader = DataLoader(dataset=val_data,
                            batch_size=32,
                            shuffle=True)


    #time_save = {'epoch':[],'time':[]}
    epochs = []
    save_items = {'epoch':[],'time':[],'loss':[]}
  #  loss_val = []
    loss_tmp = 500
    time_start = time.time()
    for epoch in range(NUM_POCHS+1-num):
        time1 = time.time()
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0,  'g_loss': 0}
     #   val_bar = tqdm(val_loader)
        val_results = {'batch_sizes': 0,  'g_loss': 0, }

        train_model.train()
        for data, target in train_bar:
            train_model.train()
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ###########################
            real_param = Variable(target)
            if torch.cuda.is_available():
                real_param = real_param.cuda()
            z = Variable(data)  #######training data
            z = z.type(torch.FloatTensor)

            if torch.cuda.is_available():
                z = z.cuda()
            z_g_in = z[:, 1024:1280]  ######### noisy data (train in)
            z_real = z[:, 0:1024]  ########real data(gt)
            z_a, _ = z.size()
            z_g_in = torch.reshape(z_g_in, (z_a, 1, 256))
            real_data = torch.reshape(z_real, (z_a, 1, 1024))
            #  netG.load_state_dict(torch.load('g_pth_save/netG_epoch_4_24.pth'))

            train_model.zero_grad()
            if ppa.model_select == 1:
                fake_data = train_model(z_g_in, 2, 256)
            else:
                fake_data = train_model(z_g_in)
            # noisy data in
            fake_data = fake_data.cuda()
            G_loss_1 = loss_fun(real_data, fake_data)  ##########Glosss 1
            running_results['g_loss'] += G_loss_1.item() * batch_size  ######## save the gloss
           # log_string('**** Training Epoch %d (%d/%d) ****' % (epoch + 1, epoch + 1, NUM_POCHS))
            #print('**** G_loss  %6f   epoch  %d   pretrain epoch %d  Proccessing(%d/%d)  ****' % (G_loss_1, epoch+num,num, num,NUM_POCHS))

            # netG.zero_grad()
            G_loss_1.backward()
            optimizer.step()

        G_loss_2 = running_results['g_loss'] / running_results['batch_sizes']
        print('G_loss  %6f   epoch  %d   pretrain epoch %d  Proccessing(%d/%d) ' % (G_loss_2, 1+epoch+num,num, 1+epoch+num,NUM_POCHS))
        time2 = time.time()
        time_c = time2- time1

        if G_loss_2<loss_tmp:
            loss_tmp = G_loss_2
            torch.save(train_model.state_dict(), save_path + '/best_model.pth')
        save_items['time'].append(time_c)
        save_items['loss'].append(G_loss_2)
        save_items['epoch'].append(epoch)

        if epoch % 10 == 0:
            torch.save(train_model.state_dict(),modelspath + '/model_'+str(num+epoch) +'.pth' )

        torch.cuda.empty_cache()



    time_end = time.time()
    time_total = time_end - time_start
    save_items['time'].append(time_total)
    save_items['epoch'].append(NUM_POCHS)
    save_items['loss'].append(loss_tmp)
    save_np = np.vstack((save_items['epoch'],save_items['time'],save_items['loss']))
    np.savetxt(save_path+'/time_loss.txt',save_np,fmt='%f',delimiter=' ')
   # np.savetxt(save_path+'loss.txt',loss,)
    #np.savetxt(save_path+'lossval.txt',loss_val)
    print('total time cost {:.3f} s'.format(time_total))
    print('final loss is {:.6}'.format(loss_tmp))
    print(1)
