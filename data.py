import numpy as np
import os


def load_dataset(data_dir,data_length,scale):

    #gloabal numbers

    txt_files = os.listdir(data_dir)
    numbers = len(txt_files)
    data_list = []
    if str(data_dir[-1]) == 'n':
        txt_files.sort(key=lambda x: int(x[7:-6]))
        for txt in txt_files:
            data_tmp = np.loadtxt(data_dir + '/' + txt)
            data_list.append(data_tmp)

        data_list = np.array(data_list).reshape(numbers, data_length//scale)
    else:
        txt_files.sort(key= lambda x : int(x[7:-4]))
        for txt in txt_files:
            data_tmp = np.loadtxt(data_dir + '/' + txt)
            data_list.append(data_tmp)

        data_list = np.array(data_list).reshape(numbers, data_length)


    return data_list


def load_target(info_file):
    tmp = np.loadtxt(info_file)
    # target = tmp.reshape(3,n_files)

    return tmp