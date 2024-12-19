import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os
import glob
import xarray as xr
import time
import keras
from keras import layers

import scipy.stats
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

import keras.callbacks as callbacks
from keras.models import load_model
import tensorflow as tf
from dask.distributed import Client, LocalCluster
from keras.utils import Sequence
import dask as da
import seaborn as sns
from  matplotlib.colors import LinearSegmentedColormap
from random import shuffle, randint

new_p = np.append([1, 5, 20], np.arange(50, 1005, 25))

#new_p = np.arange(1, 1005, 10)

dens_lu = {
    "SU": 1600,
    "AMM": 1600,
    "DU": 1700,
    "SS": 2200,
    "BC": 1600,
    "SOA": 900,
    "POM": 900
    }
    
f_dpg = {
    "ACC": 0.739,
    "AIT": 0.891,
    "CDU": 0.739,
    "CSS": 0.604,
    "FDU": 0.739,
    "FSS": 0.604,
    "PCM": 0.891
	} 
#fill_value =  -9999.
fill_value=1.e+15

min_mass = -20.
min_num =  -3. #log10(cm-3)

def int_p(W, p):
    Wint =  np.interp(new_p, p*0.01, W)#, left= fill_value, right=fill_value)
    return Wint
    
def int_lev2p(w, p):
    Wp= xr.apply_ufunc(int_p, w, p, input_core_dims=[["lev"], ["lev"]], output_core_dims=[["new_p"]], 
        dask_gufunc_kwargs=dict(output_sizes={"new_p":len(new_p)}),
        exclude_dims=set(("lev",)), vectorize=True, dask="parallelized", 
        output_dtypes=[w.dtype]) 
    lat  = Wp["lat"].values
    lon  = Wp["lon"].values
    time  = Wp["time"].values
    Wp =  Wp.assign_coords({"lon": lon, "lat":lat, "time":time, "new_p":new_p})
    Wp =  Wp.rename({'new_p':'p'})
    #Wp =  Wp.to_dataset(name="Wstd")
    return Wp.load()


MASS_vnms = ['SU_A_ACC', 'SOA_A_ACC', 'SS_A_ACC', 'POM_A_ACC', 'BC_A_ACC', 'AMM_A_ACC',
        'SU_A_AIT', 'SOA_A_AIT', 'SS_A_AIT', 'AMM_A_AIT',
        'SU_A_CDU', 'DU_A_CDU', 'AMM_A_CDU',
        'SU_A_CSS', 'SS_A_CSS', 'AMM_A_CSS',
        'SU_A_FDU', 'DU_A_FDU', 'AMM_A_FDU',
        'SU_A_FSS', 'SS_A_FSS', 'AMM_A_FSS',
        'POM_A_PCM', 'BC_A_PCM']
MASS_labels =   [item.replace("A_", "", 1) for item in MASS_vnms]
NCONC_vnms = ['NUM_A_ACC', 'NUM_A_AIT', 'NUM_A_CDU', 'NUM_A_CSS', 'NUM_A_FDU', 'NUM_A_FSS', 'NUM_A_PCM']

NUM_labels =  [item.replace("A_", "", 1) for item in NCONC_vnms]

DPG_vnms =["DPG_ACC", "DPG_AIT", "DPG_CDU", "DPG_CSS","DPG_FDU", "DPG_FSS", "DPG_PCM"] 

all_vnms =  MASS_vnms + NCONC_vnms + DPG_vnms

all_labels = MASS_labels + NUM_labels + DPG_vnms
        
def correlation(x1, x2):	
    x1 = np.nan_to_num(x1, nan = -40.  )
    x2 = np.nan_to_num(x2, nan = -40.  )    
    return np.corrcoef(x1, x2)[0,1] # to return a single correlation index, instead of a matriz

def bias(x1, x2):
    x1 = np.nan_to_num(x1, nan = -40.  )
    x2 = np.nan_to_num(x2, nan = -40.  )
    return np.mean(x1-x2)

def NMB(pred, true):
    pred  = pow10(pred)
    true =  pow10(true) 
    b  = np.mean((pred-true))/np.mean(true)       
    return b
 
def MRB(pred, true):
    pred  = pow10(pred)
    true =  pow10(true) 
    b  = (pred-true)/true       
    return np.mean(b)
        
def metric(fun, y, x, coord='s'):
    #'Finds the "fun" along a given dimension of a dataarray.'
    return xr.apply_ufunc(fun, y, x, input_core_dims=[[coord],[coord]] , 
    output_core_dims=[[]], vectorize=True, output_dtypes=[float], dask="parallelized")

def pow10(d):
    func = lambda x: np.power(10, x)
    return xr.apply_ufunc(func, d, dask="parallelized")

def powN(d, n):
    func = lambda d, n: np.power(d, n)
    return xr.apply_ufunc(func, d, n, dask="parallelized")
        
def log_10(d):
    func = lambda x: np.log10(x)
    return xr.apply_ufunc(func, d, dask="parallelized")
    
    
def stdz_input(ds, mu_lst, sigma_lst):   
    for i,var in enumerate(ds.data_vars):
        #print('=====STDZ IN',var)
        dsv = ds[var]
        #print('mean', dsv.mean().values)
        #print('std', dsv.std().values)
        mu = mu_lst[i]
        sigma = sigma_lst[i]
        dsv = (dsv - mu)/sigma
        ds[var] = dsv
    '''    
    mu = ds.mean(axis=1)
    sigma = 1.
    
    ds = (ds - mu)/sigma
    '''
    return ds

def get_random_files(pth, nts, name):

    print(pth)
    lall = glob.glob(pth)
    #print(lall)
    f_ind = np.random.randint(0, len(lall)-1, nts)
    print(f_ind)
    fils = [lall[i] for i in f_ind] 
    print("==========fils")
    print(fils)
    return fils
 
def dens (ds): 
    d = ds.PL/287.0/ds.T
    ds.PL.data = d
    return ds
    
def get_pres(ds):
    p =  287.0*ds.T*ds.AIRDENS # in Pa 
    return p 


def get_Veq(dat_MAS, datNCONC):
	
    Veq = datNCONC*0.0
    #ACC
    Veq[NCONC_vnms[0]] =   dat_MAS[MASS_vnms[0]]/dens_lu['SU'] + dat_MAS[MASS_vnms[1]]/dens_lu['SOA'] + dat_MAS[MASS_vnms[2]]/dens_lu['SS'] + \
    dat_MAS[MASS_vnms[3]]/dens_lu['POM'] + dat_MAS[MASS_vnms[4]]/dens_lu['BC'] + dat_MAS[MASS_vnms[5]]/dens_lu['AMM']    
    #Aitken
    Veq[NCONC_vnms[1]]= dat_MAS[MASS_vnms[6]]/dens_lu['SU'] + dat_MAS[MASS_vnms[7]]/dens_lu['SOA'] + dat_MAS[MASS_vnms[8]]/dens_lu['SS'] + \
    dat_MAS[MASS_vnms[9]]/dens_lu['AMM'] 
    #coarse dust
    Veq[NCONC_vnms[2]] =   dat_MAS[MASS_vnms[10]]/dens_lu['SU'] + dat_MAS[MASS_vnms[11]]/dens_lu['DU'] + dat_MAS[MASS_vnms[12]]/dens_lu['AMM'] 
    #coarse sea salt
    Veq[NCONC_vnms[3]]  = dat_MAS[MASS_vnms[13]]/dens_lu['SU'] + dat_MAS[MASS_vnms[14]]/dens_lu['SS'] + dat_MAS[MASS_vnms[15]]/dens_lu['AMM'] 
    #fine dust
    Veq[NCONC_vnms[4]] = dat_MAS[MASS_vnms[16]]/dens_lu['SU'] + dat_MAS[MASS_vnms[17]]/dens_lu['DU'] + dat_MAS[MASS_vnms[18]]/dens_lu['AMM'] 
    #fine sea salt
    Veq[NCONC_vnms[5]] = dat_MAS[MASS_vnms[19]]/dens_lu['SU'] + dat_MAS[MASS_vnms[20]]/dens_lu['SS'] + dat_MAS[MASS_vnms[21]]/dens_lu['AMM'] 
    #Primary carbon matter
    Veq[NCONC_vnms[6]] = dat_MAS[MASS_vnms[22]]/dens_lu['POM'] + dat_MAS[MASS_vnms[23]]/dens_lu['BC']

    return Veq

        
def get_DPG(datMASS, datNCONC, pred=False): #This has to be done before taking logs
        
        # f = rv/reff (ratio of volumetric radius to effective radius)
            
        FDPG = [ 0.739, 0.891, 0.739, 0.604, 0.739, 0.604, 0.891]    
        
        datMASS  = pow10(datMASS)*1.0e-9
        datNCONC =  pow10(datNCONC)*1.e6

        # calculate modal mass scaled by density
        Veq = get_Veq(datMASS, datNCONC)         
        
        datDPG = Veq*0
        i =  0

        for v in datDPG.data_vars:        
            datDPG[v]  = 1.0e6*FDPG[i]*powN(Veq[v]/datNCONC[v], 1/3) 
            i =  i+1
        aux =  np.power(10., min_num)
        datDPG = datDPG.where(datNCONC > aux)
       
        i =  0
        for v in datDPG.data_vars: 
            datDPG = datDPG.rename({v:DPG_vnms[i]})
            i = i+1  
        
            
        #datDPG = datDPG.fillna(fill_value)
        def dpg_stats(name, dmicrons):          
          #dmicrons = d.squeeze()*1e6
          #m  =  dmicrons.mean()
          #s =  dmicrons.std()
          m = np.nanmean(dmicrons)
          s = np.nanstd(dmicrons)
          
          print('===== microns ', name, m, s)
        

        return datDPG.load()

def con_loss(loss_train, loss_val):
    return np.abs(np.subtract(loss_train, loss_val))

 
class get_dts():
    def __init__(self, exp_out=1, ndts =  1, nam ="def", batch_size = 32000, write_test=False, levup=1):  #creates a class that will handle nts files

        #self.filepath = "/home/dbarahon/nobck/RESMOD/2020_basemodels/mam_base/mam_NN/mam_base.mam_NN.monthly.2*.nc4"
        #self.filepath = "/home/dbarahon/nobck/RESMOD/2020_basemodels/mam_base/holding/mam_NN/*/*.nc4"
        self.filepath = "/discover/nobackup/khbreen/ML_models/MAMnet/data/*.nc4"
        self.chk = { "lat": -1, "lon": -1, "lev":  -1, "time": 1} # needed so we can regrid

        #get nts files 
        self.fls = get_random_files(self.filepath,ndts,name=nam)
        self.levup = levup # cut the stratosphere    
        self.levs = 72
        self.var_in = 7
        self.var_out = 31
        self.name = nam


        # [SU, SS, OG, BC, DU, T, P]
        self.input_mu = [-1.95, -2.32, -3.01, -4.12, -4.72, 243.8, 0.39] #calculated from 100 random time steps  
        self.input_sigma = [3.11, 3.46, 3.34, 3.24, 5.61, 27.35, 0.44 ]

        self.batch_size = batch_size
        dat_MASS =  xr.open_mfdataset(self.fls, chunks=self.chk, parallel=True).sel(lev=slice(self.levup,72))[MASS_vnms]
        dat_NCONC =  xr.open_mfdataset(self.fls, chunks=self.chk, parallel=True).sel(lev=slice(self.levup,72))[NCONC_vnms]
        
        dat_MASS = dat_MASS*1e9 ##To microg/Kg #####################
        dat_NCONC = dat_NCONC*1e-6 ##To 1/mg #####################


        #print('GET INPUTS')
        ######################################
        #### INPUTS ####
        # sum mass mixing ratios across modes per species
        # T, P
        ######################################
        # make sulphates the sum of sulphate and ammonia
        SU_vnms = ['SU_A_ACC', 'SU_A_AIT', 'SU_A_CDU', 'SU_A_CSS', 'SU_A_FDU', 'SU_A_FSS',
        'AMM_A_ACC', 'AMM_A_AIT', 'AMM_A_CDU', 'AMM_A_CSS', 'AMM_A_FDU', 'AMM_A_FSS'
        ]
        SS_vnms = ['SS_A_ACC', 'SS_A_AIT', 'SS_A_CSS', 'SS_A_FSS']
        OG_vnms = ['POM_A_ACC', 'POM_A_PCM', 'SOA_A_ACC', 'SOA_A_AIT']
        BC_vnms = ['BC_A_ACC', 'BC_A_PCM']  
        DU_vnms = ['DU_A_CDU', 'DU_A_FDU']
        OTH_vnms = ['T', 'AIRDENS']
        
        #input
        dat_lst=  []
        def get_mass(vnms, name):
            mass_sum =  dat_MASS[vnms].to_array().sum("variable")
            mass_sum = xr.where(mass_sum>0., np.log10(mass_sum), min_mass)
            mass_sum.name = name
            return mass_sum

        SUsum = get_mass(SU_vnms, 'SUsum')
        SSsum = get_mass(SS_vnms, 'SSsum')
        OGsum = get_mass(OG_vnms, 'OGsum')
        BCsum = get_mass(BC_vnms, 'BCsum')
        DUsum = get_mass(DU_vnms, 'DUsum')


        dat_lst.append(SUsum)
        dat_lst.append(SSsum)
        dat_lst.append(OGsum)
        dat_lst.append(BCsum)
        dat_lst.append(DUsum)

        # temp, air density
        dat_tmp = xr.open_mfdataset(self.fls, chunks=self.chk, parallel=True).sel(lev=slice(self.levup,72))[OTH_vnms]
        dat_lst.append(dat_tmp)

        MAMin = xr.merge(dat_lst) # use this for the mass conservation - we want the mass IN to be the raw values so they match the preprocessing done on output mass
        
       # print('====in====', MAMin.load())
        MAMin = stdz_input(MAMin, self.input_mu, self.input_sigma)
       # print('====in==std==', MAMin.load())
        self.MAMin = MAMin        
        #we didn't save pressure!! 
        self.press =  get_pres(dat_tmp)
        
  
        #==========================output===============================
          
        m= np.power(10., min_mass)
        dat_MASS = xr.where(dat_MASS<1e3, dat_MASS, 0.0)
        dat_MASS = xr.where(dat_MASS> m, np.log10(dat_MASS), min_mass) 
        self.Mout = dat_MASS             

        m= np.power(10., min_num)
        dat_NCONC = xr.where(dat_NCONC<1e14, dat_NCONC, 0.0)
        dat_NCONC = xr.where(dat_NCONC>m, np.log10(dat_NCONC), min_num)        
        self.Nout =  dat_NCONC
         
        dat_MASS =  dat_MASS.load()
        dat_NCONC =  dat_NCONC.load() 
        self.MAMout =  xr.merge([dat_MASS, dat_NCONC]) 
        self.DPG =  get_DPG(dat_MASS, dat_NCONC)
        
       # print('===============out=======================================')
       # print(self.MAMout)


       
        
         
######################################
## mass
######################################


    
    def get_Xy(self, batch_size=5120, test=False):
        self.batch_size = batch_size
        self.feat_in = self.var_in
        self.feat_out = self.var_out

        Xall = self.MAMin
        yall_mass = self.MAMout

        Xall = Xall.to_array()
        Xall = Xall.stack(s = ('time',  'lev', 'lat', 'lon'))
        Xall =  Xall.rename({"variable":"ft"})  
        Xall = Xall.squeeze()
        Xall = Xall.transpose()
        Xall = Xall.chunk({"ft":self.feat_in, "s": 34000}) #chunked this way aligns the blocks/chunks with the samples      

        yall_mass = yall_mass.to_array()
        yall_mass = yall_mass.rename({"variable":"ft"})
        self.mass_feat = len(yall_mass['ft'].values)  
        yall_mass = yall_mass.stack(s = ('time', 'lev', 'lat', 'lon'))
        
        yall_mass = yall_mass.squeeze()
        yall_mass = yall_mass.transpose()   
        yall_mass = yall_mass.chunk({"ft":self.feat_out, "s": 34000})

        Xall = Xall.chunk({"ft":self.feat_in, "s": batch_size})      
        self.Nsamples = len(yall_mass['s'].values)
        self.X_feat = len(Xall['ft'].values)

        return Xall.load(), yall_mass.load()
           
   
#==============set callbacks===========================
# try a custom callback 
class epoch_cllbck(callbacks.Callback):
    def __init__(self, mod_name):
        self.mass_min_loss = np.inf
        self.min_con_loss = np.inf
        self.mod_name = mod_name
        
    def on_epoch_end(self, epoch, logs=None):
        
        # the log keys are the dict keys in the return statement in the training function for the custom model
        #keys = list(logs.keys())
        #print("End epoch {} of training; got log keys: {}".format(epoch,keys))
        
        # SAVE CHECKPOINT IF VAL LOSS IS DECREASING
        monitor = "val_loss"
        current_loss = logs.get(monitor)
        print("min val_loss: {}    current val_loss: {}   min>current: {}".format(self.mass_min_loss,current_loss,self.mass_min_loss > current_loss))
        
        if self.mass_min_loss > current_loss:
            self.mass_min_loss = current_loss
            filename = self.mod_name + "_best_checkpoint.h5"
            message = monitor + " improved from " + str(self.mass_min_loss) + " to " + str(current_loss) + " - saving to " + filename
            print(message, '--current epoch: ', epoch)
            # save the current state of the generator/discriminator
            MASS_mod.save(filename)
        
        # SAVE CHECKPOINT IF TRAIN AND VAL LOSSES ARE CONVERGING
        current_train_loss = logs.get("loss")
        current_val_loss = logs.get("val_loss")
        current_con_loss = con_loss(current_train_loss, current_val_loss)
        print("min con_loss: {}    current con_loss: {}   min>current: {}".format(self.min_con_loss,current_con_loss,self.min_con_loss > current_con_loss))
        
        if self.min_con_loss > current_con_loss:
            self.min_con_loss = current_con_loss
            filename = self.mod_name + "_best_checkpoint.h5"
            message = "con_loss improved from " + str(self.min_con_loss) + " to " + str(current_con_loss) + " - saving to " + filename
            print(message, '--current epoch: ', epoch)
            # save the current state of the generator/discriminator
            MASS_mod.save(filename)
        
        
def set_callbacks(mod_name):
    # SET CALLBACKS

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=20,
        verbose=1,
        mode='min'
    )

    csv_logger = callbacks.CSVLogger(mod_name+'.csv', append=False)

    # checkpoint for lowest val loss overall
    best_set_chkpt = callbacks.ModelCheckpoint(
                        mod_name + '_checkpoint.h5',
                        monitor='val_loss', 
                        verbose=1, 
                        save_best_only=True, 
                        save_weights_only=True,
                        mode='min') 
                        
    #callbcks = [csv_logger, best_set_chkpt]
    #callbcks = [csv_logger, best_set_chkpt, TerminateOnNaN()]
    callbcks = [early_stop, csv_logger, callbacks.TerminateOnNaN(), epoch_cllbck(mod_name)]
     
    return callbcks  
    

def build_model(hp, feats_in, feats_out, name):
    
    Nfeat_in = feats_in
    Nfeat_out = feats_out
    
    # 
    mass_in = layers.Input(shape=(Nfeat_in,), name=name+"_input")
    x = layers.Dense(hp['Nnodes'], name=name+"_dense0")(mass_in)
    x = LeakyReLU(alpha=hp['alpha'], name=name+"_lr0")(x)
    #x = layers.BatchNormalization(name=name+"_bn0")(x)
    x = layers.Dropout(hp['do'], name=name+"_do0")(x)
    for i in range(hp['Nlayers']-1):
        x = layers.Dense(hp['Nnodes'], name=name+"_dense"+str(i+1))(x)
        x = LeakyReLU(alpha=hp['alpha'], name=name+"_lr"+str(i+1))(x)
        #x = layers.BatchNormalization(name=name+"_bn"+str(i+1))(x)
        x = layers.Dropout(hp['do'], name=name+"_do"+str(i+1))(x)
    N_out = layers.Dense(Nfeat_out, name=name+"_out")(x) 

    # define the model I/O - this lets the model know where to calculate loss (each output)
    model = Model(mass_in, N_out)
    
    opt = tf.keras.optimizers.Adam(learning_rate=hp['lr'])
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt)
    
    return model


#=========================================
#=========================================
#=========================================
if __name__ == '__main__':

    model_name = "MAMnet"
    
    # set2D, trial 032
    hp = {
        'Nlayers': 7,
        'Nnodes': 256,
        'lr': 1e-5,
        'alpha': 0.3,
        'do': 0.1,
        'batch_size' : 4608
    }
    
    strategy = tf.distribute.MirroredStrategy()
    batch_size = hp['batch_size'] #actual batch size
    

    nepochs =  250
    ndts_train = 25 #   # time steps for training 
    ndts_val  = 5      # time steps for validation
    ndts_test =  10     # time steps for testing
    do_train =  False
    do_plot = False
    do_test = True
    read_nc = True  # instead of loading the model/new data, read in prev *.nc files to plot heatmaps
    do_heat  =  True #only works if do_test=True
    do_pressure =  True #generate file and heat map on pressure levels
    
    if do_train:
        #######################################################
        # GET DATA
        #######################################################
        #print('====GET DATA====')
        #=========create model and data streams==========#
        train_data =  get_dts(ndts=ndts_train, nam = 'train', batch_size = hp['batch_size'])
        val_data =  get_dts(ndts=ndts_val, nam = 'val', batch_size =hp['batch_size'])

        X_train,  y_train = train_data.get_Xy(batch_size = hp['batch_size']) 
        X_val,  y_val = val_data.get_Xy(batch_size = hp['batch_size']) 

        #print('=========Xtrain============', X_train)
        #print('=========Ytrain=============', y_train)
        
        Xtrain = tf.cast(X_train.values,tf.float32)
        Xval = tf.cast(X_val.values,tf.float32)
        ytrain = tf.cast(y_train.values,tf.float32)
        yval = tf.cast(y_val.values,tf.float32)


        #######################################################
        # DEFINE and train MODEL
        #######################################################

        # use mass summed per species (across modes) to predict mass per species per mode
        # def build_model(hp, feats_in, feats_out, levs, out_name):
        mass_in = train_data.X_feat
        mass_out = train_data.mass_feat

        with strategy.scope():
            print('========', mass_in, mass_out)
            MASS_mod =  build_model(hp, mass_in, mass_out, 'mass') 
        MASS_mod._name = 'MAMnet'
        MASS_mod.summary()


        history =MASS_mod.fit(
                     Xtrain, ytrain,
                     validation_data =(Xval, yval),
                     epochs=nepochs, 
                     batch_size  = hp['batch_size'],
                     callbacks=set_callbacks(model_name), 
                     verbose=2, 
                     use_multiprocessing=True, 
                     workers=10
                     ) 

        '''
        print('####### PLOT LOSS')

        #plot mass loss
        plt.switch_backend('agg')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(model_name + '_loss.png')


        print('####DONE####')
        '''
    if do_plot:  # plot the loss without re-training the model


        print('####### PLOT LOSS')
        
        history = pd.read_csv("MAMnet.csv")
        
        #plot mass loss
        plt.switch_backend('agg')
        #plt.rcParams['text.usetex'] = True  # use latex to render text
        #from matplotlib import rcParams
        #plt.rcParams.update({"text.usetex": True})
        plt.plot(history['loss'], linewidth=2)
        plt.plot(history['val_loss'], linewidth=2, linestyle="--")
        #plt.title('model loss')
        plt.ylabel(r'$\mathcal{L}_{\mathrm{prior}}$', fontsize=16)
        plt.xlabel(r'epoch', fontsize=16)
        plt.legend(['training loss', 'validation loss'], loc='upper right', frameon=False, framealpha=0., fontsize='x-large', borderpad=0)
        plt.savefig(model_name + '_loss.png')


        print('####DONE PLOTTING####')
    
    if do_test:
      if read_nc:  # read in data to plot
      
        all_vnms_pred = []
        for vnm in all_vnms:
          all_vnms_pred.append(vnm+"_pred")
           
        test_data = xr.open_dataset(model_name+"_test.nc")
        #print('====TEST DATA',test_data)
        
        ytrue = test_data[all_vnms]
        yp = test_data[all_vnms_pred]
        
        for var in yp.data_vars:
          yp = yp.rename({var:var[:-5]})
        
        if do_pressure:
          test_data_pres = xr.open_dataset(model_name+"_test_pres.nc")
          print('====TEST DATA PRES',test_data_pres)
          print(test_data_pres.coords['lev'].values)
          
          ytrue_p = test_data_pres[all_vnms]
          yp_p = test_data_pres[all_vnms_pred]
        
          for var in yp_p.data_vars:
            yp_p = yp_p.rename({var:var[:-5]})
      else:  # predict using checkpoint
        #=========================================
        #====================test=====================
        #=========================================
        #note: datasets instead of arrays are easier for netcdf so we tell it not to make y into an array yet

        test_data =  get_dts(ndts=ndts_test, nam = 'test', batch_size = hp['batch_size'], levup=1)

        pth  =  model_name + '_best_checkpoint.h5'
        print(pth)
        if os.path.exists(pth): #get the best model
            
            # Load model:
            print('==============test set comparison============')
            model = load_model(pth, compile=True)

        test_x, test_y =  test_data.get_Xy(batch_size = 512*72*5, test = True)
     
        #==error calculation
        #y_t =  test_y.to_array(dim="W").squeeze().persist() #need array for error calculation
        X =  test_x.load()
        y_hat = model.predict(X, batch_size=32768) 
        y_hat =  np.squeeze(y_hat) 

        print ("===yhat==")
        #print(y_hat)
        print(y_hat.shape)

        #==========save netcdf ================"
        y_pred=  test_y.copy(data=y_hat) #short cut to make y_pred a data set and inherit attributes and coordinates 

        ytrue = test_y.transpose().unstack()
        
        ytrue =  ytrue.to_dataset('ft').set_coords(['time', 'lev', 'lat', 'lon'])#.rename({"W":"Wvar"})        
        ypred = y_pred.transpose().unstack()        
        ypred = ypred.to_dataset('ft').set_coords(['time', 'lev', 'lat', 'lon'])#.rename({"W":"Wvar_pred"})
        ytrue = ytrue.transpose('time', 'lev', 'lat', 'lon').load()
        ypred = ypred.transpose('time', 'lev', 'lat', 'lon').load()
        
        #get ================ DPG
        DPG_true  = log_10(test_data.DPG.load())
        DPG_pred = get_DPG(ypred[MASS_vnms], ypred[NCONC_vnms], pred = True)
        DPG_pred = log_10(DPG_pred)
        
        ytrue =  xr.merge([ytrue, DPG_true])
        ypred =  xr.merge([ypred, DPG_pred])
        
        
        yp =  ypred #save for heat map
        for var in ypred.data_vars:
        #print(var)
        	ypred = ypred.rename({var:var+"_pred"})

        print("=====ytrue======")
        print(ytrue)
        print("====ypred=======")
        print(ypred)

        encx = dict(dtype= 'float32', _FillValue= fill_value)
        enc = {var: encx for var in ytrue.data_vars}
        ytrue.to_netcdf(model_name+"_test.nc", mode = "w", encoding=enc)

        enc = {var: encx for var in ypred.data_vars}
        ypred.to_netcdf(model_name+"_test.nc", mode = "a", encoding=enc)
        
        
        if do_pressure:
            pres = test_data.press 
            ytrue_p = int_lev2p(ytrue.to_array(), pres)
            ytrue_p = ytrue_p.to_dataset('variable').rename({"p":"lev"}).load()
            ytrue_p = ytrue_p.set_coords(['time', 'lev', 'lat', 'lon']).transpose('time', 'lev', 'lat', 'lon')

            ypred_p = int_lev2p(yp.to_array(), pres)
            ypred_p = ypred_p.to_dataset('variable').rename({"p":"lev"}).load()
            ypred_p = ypred_p.set_coords(['time', 'lev', 'lat', 'lon']).transpose('time', 'lev', 'lat', 'lon')
            
        
            yp_p =  ypred_p #save for heat map
            for var in ypred_p.data_vars:
            	ypred_p = ypred_p.rename({var:var+"_pred"})

            print("=====ytrue_p======")
            print(ytrue_p)
            print("====ypred_p=======")
            print(ypred_p)

            enc = {var: encx for var in ytrue_p.data_vars}
            ytrue_p.to_netcdf(model_name+"_test_pres.nc", mode = "w", encoding=enc)

            enc = {var: encx for var in ypred_p.data_vars}
            ypred_p.to_netcdf(model_name+"_test_pres.nc", mode = "a", encoding=enc)
        
         #=========================================
        #====================heatmap=====================
      if do_heat:
            min_heat = -100. 
            
            #heat map options
            plt.switch_backend('agg')
            #sns.set_theme()
            plt.rcParams.update({'font.size': 4.2})
            cmap_log=LinearSegmentedColormap.from_list('byr',["b", "y", "r"], N=256)
            cmap_cr=LinearSegmentedColormap.from_list('rb',["r", "b"], N=256)
            linewidths = 0.05
            linecolor = "gray"   
            
            '''
            fig, axes = plt.subplots(nrows=2, ncols=1)#, edgecolor='k', linewidth=1.5)
            axes =  axes.flatten()                   
    
            yt =  ytrue.stack(s = ('time',  'lat', 'lon')).squeeze()
            yp =  yp.stack(s = ('time',  'lat', 'lon')).squeeze()            
            yt = yt.where(yt > min_heat)
            yp = yp.where(yp > min_heat)
            
            mt =  metric(correlation, yp, yt)
            print('-------correlation-----', mt) 
            data_c  =  mt.to_array()          
            s=sns.heatmap(data=data_c,
            vmin=0.5,
            vmax=1, cmap =cmap_cr, center=0.75, 
            linewidths=linewidths,
            linecolor=linecolor,  yticklabels=all_vnms, ax=axes[0])
            axes[0].set_title('Correlation')
            
            mt =  metric(bias, yp, yt)      
            print('-------bias-----', mt) 
            data_c  =  mt.to_array()                      
            s=sns.heatmap(data=data_c,
            vmin=-0.5,
            vmax=0.5, cmap =cmap_log, center=0.0, 
            linewidths=linewidths,
            linecolor=linecolor,  yticklabels=all_vnms, ax=axes[1])
            axes[1].set_title('Log10 mean bias')
  
            #axes[1].set_yticklabels([])
            #s.set(ylabel = 'var')
            #s.set(xlabel = 'level')
            #plt.rc('ytick', labelsize=3) 
            fig.tight_layout()
            tit = model_name + '_heat_lev.png'
            plt.savefig(tit, dpi =  200) 
            '''
                        
            if do_pressure: 
                print('####PLOT PRESSURE HEAT####')
                yt =  ytrue_p.stack(s = ('time',  'lat', 'lon')).squeeze()
                yp =  yp_p.stack(s = ('time',  'lat', 'lon')).squeeze()
                yt = yt.where(yt > min_heat)
                yp = yp.where(yp > min_heat)

                fig, axes = plt.subplots(nrows=2, ncols=1)#, edgecolor='k', linewidth=1.5)
                fig.set_size_inches(6,5.5)  # (w,h) 
                axes =  axes.flatten()
                mt =  metric(correlation, yp, yt)
                            
                data_c  =  mt.to_array().load() 
                print('-------correlation (press)-----', data_c)          
                s=sns.heatmap(data=data_c,
                vmin=0.5,
                vmax=1.0, cmap =cmap_cr, center=0.75, 
                linewidths=linewidths,
                linecolor=linecolor,  yticklabels=all_labels, ax=axes[0], xticklabels=new_p)
                #axes[0].set_title(r'Correlation')
                s.collections[0].colorbar.set_label(r"$R$", fontsize=6)

                mt =  metric(bias, yp, yt)  # np.mean(x1-x2)
                print('-------bias (press)------', mt) 
                data_c  =  mt.to_array()                      
                s=sns.heatmap(
                    data=data_c,
                    vmin=-0.5,
                    vmax=0.5, 
                    cmap =cmap_log, 
                    center=0.0, 
                    linewidths=linewidths,
                    linecolor=linecolor,  
                    yticklabels=all_labels, 
                    ax=axes[1], 
                    xticklabels=new_p
                    )
                #axes[1].set_title(r'$log_{10}$ Mean Bias')
                #s.collections[0].colorbar.set_label(r"$\frac{\Sigma (log_{10}(\mathbf{Y}) - log_{10}(\mathbf{\hat{Y}}))}{N_{\mathrm{time,lat,lon}}}$", fontsize=6)
                s.collections[0].colorbar.set_label("MLB", fontsize=6)
                '''
                mt =  metric(NMB, yp, yt)      
                print('-------NMB-----', mt) 
                data_c  =  mt.to_array()                      
                s=sns.heatmap(data=data_c,
                vmin=-0.25,
                vmax=0.25, cmap =cmap_log, center=0.0, 
                linewidths=linewidths,
                linecolor=linecolor,  yticklabels=all_vnms, ax=axes[2])
                axes[2].set_title('NMB')
                '''
                #axes[1].set_yticklabels([])
                #s.set(ylabel = 'var')
                s.set(xlabel = 'Pressure (hPa)')
                #plt.rc('ytick', labelsize=3) 
                fig.tight_layout()
                tit = model_name + '_heat_pres.png'
                plt.savefig(tit, dpi =  200) 
                
                print("DONE!!!!!")
                
               
            
            
            
            

    
    
 
