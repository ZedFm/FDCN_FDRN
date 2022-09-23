## Full-waveform LiDAR echo decomposition based on dense and residual neural networks
### Paper and Details:
https://doi.org/10.1364/AO.444910
### Architecture: 
![image](https://github.com/ZedFm/FDCN_FDRN/blob/d5955a1222ff2ebe4439e64a251379336400bd26/pics/pic_1.png)
### Dataset and Pre-trained Models:
Google Drive: https://drive.google.com/file/d/1jI9kVC4dHZewRxjOZIttKfcPJo1Dgd53/view?usp=sharing 
or
Google Drive2:https://drive.google.com/file/d/1xV7iCuQxGroAsB3Cxdwi-IqgbHVC2aXI/view?usp=sharing

### Training Details:
![image](https://github.com/ZedFm/FDCN_FDRN/blob/d5955a1222ff2ebe4439e64a251379336400bd26/pics/pic_2_train.png)

### Usage:
Train: `<python train.py --snr 30/24/18/12 --model_select 0/1  ### 0: FDCN 1: FDRN >`  
Test: `<python test.py >` 

### Some Result:
#### FDCN:

![image](https://github.com/ZedFm/FDCN_FDRN/blob/d5955a1222ff2ebe4439e64a251379336400bd26/pics/pic_3_d.png)
  
  
#### FDRN:
![image](https://github.com/ZedFm/FDCN_FDRN/blob/d5955a1222ff2ebe4439e64a251379336400bd26/pics/pic_3_r.png)


#### Real World Experiment:
![image](https://github.com/ZedFm/FDCN_FDRN/blob/d5955a1222ff2ebe4439e64a251379336400bd26/pics/pic_3_total.png)

### Cite
if you find this work useful, please cite :
@article{liu2022full,
  title={Full-waveform LiDAR echo decomposition based on dense and residual neural networks},
  author={Liu, Gangping and Ke, Jun},
  journal={Applied Optics},
  volume={61},
  number={9},
  pages={F15--F24},
  year={2022},
  publisher={Optical Society of America}
}

Thanks a lot!
