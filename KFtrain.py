
import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import pipeline
from sklearn import impute
from sklearn import compose
from sklearn import neural_network
import joblib

import matplotlib as mpl
from matplotlib import pyplot as plt

import torch.nn as nn
import torch.optim as optim

import neural_filter
import torchNN

import test_functions
import createDataLive
import kfnn
import OrbitDetermination as OD


dataseting = 4
datalegnths = {0:5,1:10,2:9,3:18,4:10,5:20,6:30}
datalength = datalegnths[dataseting]
print("Running with input size",datalength)
dnntype = 0
dnntypes = {0:'dnn',1:'rnn',2:'encoder',3:'transformer'}
save_append = dnntypes[dnntype]#+'-200'

maxIters = 1000

# training parameters
n_epochs = 5   # number of epochs to run
batch_size = 100  # max size of each batch
num_tles = 10000
obs_noise = .0003 #0.02 # .02 is 1.2 arcminutes, .004 is ~15 arcseconds, .0003 ~1.1 arcseconds
batch_start = torch.arange(0, maxIters, batch_size)
test_size = 100
eval_size = 500
max_obs = 15
num_test_tles = 10000
# min_obs = 5
tmin = 5
tmax = 5

dt = 10

rangenorm = 72000
d2r = np.pi/180

TLEtest = createDataLive.TLEdata()
TLE = createDataLive.TLEdata()
Dmanip = torchNN.MLdataManipulation()
orbdet = OD.orbitDetermination()
TF = test_functions.testML()
uccs = neural_filter.locations.Location('UCCS',38.89588,-104.80232,1950)

nnpred = kfnn.KFNN_v4(dimin=datalength,dnntype=dnntype,covarianceStrength=0)
nnkf = kfnn.KFNN_v0(dimin=datalength,dnntype=dnntype)#,covarianceStrength=1e-6)
nnkfp = kfnn.KFNN_v3(dimin=datalength,dnntype=dnntype,covarianceStrength=0,usezerr=True)
nnkfz = kfnn.KFNN_v2(dimin=datalength,dnntype=dnntype,covarianceStrength=0,usezdiff=True)
nnkfr = kfnn.KFNN_v1(dimin=datalength,dnntype=dnntype)
nnkfo = kfnn.KFNN_v5(dimin=datalength,dnntype=dnntype,covarianceStrength=0,usez=True)
print("Running",save_append,"with datalength",str(datalength))
try:
    nnkf.loadNN('models/KFNN_v0_dnn_'+str(dataseting)+save_append+'bestweights')
    print("Loaded previous weights")
except:
    print("Could not load previous weights")
try:
    nnpred.loadNN('models/KFNN_v4_dnn_'+str(dataseting)+save_append+'bestweights')
    print("Loaded previous weights")
except:
    print("Could not load previous weights")
try:
    nnkfz.loadNN('models/KFNN_v2_dnn_'+str(dataseting)+save_append+'bestweights')
    print("Loaded previous weights")
except:
    print("Could not load previous weights")
try:
    nnkfr.loadNN('models/KFNN_v1_dnn_'+str(dataseting)+save_append+'bestweights')
    print("Loaded previous weights")
except:
    print("Could not load previous weights")
try:
    nnkfp.loadNN('models/KFNN_v3_dnn_'+str(dataseting)+save_append+'bestweights')
    print("Loaded previous weights")
except:
    print("Could not load previous weights")
try:
    nnkfo.loadNN('models/KFNN_v5_dnn_'+str(dataseting)+save_append+'bestweights')
    print("Loaded previous weights")
except:
    print("Could not load previous weights")


dataSaveX = []
dataSaveY = []
history = []
XpredSave = []
CovarianceSave = []
positionErrSave = []
gaussErr = []
errGauss = []
bestErr1 = 1e2
bestErr2 = 1e2
bestErr3 = 1e2
bestErr4 = 1e2
bestErr5 = 1e2
bestErr6 = 1e2

n_epochs += 1
tletest = TLEtest.createRandomOrbit(num_test_tles)#,maxAltSeed=rangenorm)
tle = TLE.createRandomOrbit(num_tles)#,maxAltSeed=rangenorm)
for epoch in range(n_epochs):
    saveX = []
    saveY = []
    X_batch = []
    orbit_batch = []
    sv_batch = []
    R_batch = []
    y_batch = []
    nnkf.dnn.train()
    nnpred.dnn.train()
    nnkfp.dnn.train()
    nnkfr.dnn.train()
    nnkfz.dnn.train()
    nnkfo.dnn.train()
    for iter in range(batch_size):
        if epoch==0:
            break
        dt = np.random.randint(tmin,tmax+1)
        obsrvr, reality, orbit = TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs)
        observer = np.transpose(obsrvr)
        gausoln3 = orbdet.GaussOrbitDetermination(observer[0],[observer[6].astype(float),observer[7].astype(float)],uccs,radecOrazel=0)
        nnkf.filter.x = np.append(gausoln3[1],gausoln3[2])
        realtle = reality[1]
        reals = np.transpose(reality[0])
        siderealtime = TLE.OD.siderealTime(observer[0],TLE.location)
        # consideration - try setting data so input is cosines and sines of angles instead of direct angular measurements?
        Dtrain,Rtrain = Dmanip.grabInputNoNorm(observer,siderealtime,dataseting)
        nnkf.setStartTime(observer[0][0])
        nnpred.setStartTime(observer[0][0])
        nnkfz.setStartTime(observer[0][0])
        nnkfr.setStartTime(observer[0][0])
        nnkfp.setStartTime(observer[0][0])
        nnkfo.setStartTime(observer[0][0])
        nnkf.filter.P = np.diag([1e6,1e6,1e6,1e2,1e2,1e2])
        orbits = []
        for ii in range(len(Dtrain)):
            orbit = orbdet.OrbitElement2StateVector4(reality[ii]).flatten()
            X_batch.append(Dtrain[ii])
            R_batch.append(Rtrain[ii])
            sv_batch.append(reality[ii])
            # Train KFNN - expects each input individually
            # nnkf.forward(Dtrain[ii])
            # nnkf.trainstep(Dtrain[ii],Rtrain[ii],dt=dt)
            # nnpred.trainstep(Dtrain[ii],reality[ii],dt=dt)
            # nnkfz.trainstep(Dtrain[ii],Rtrain[ii],dt=dt)
            # nnkfr.trainstep(Dtrain[ii],Rtrain[ii],dt=dt)
            # nnkfp.trainstep(Dtrain[ii],reality[ii],dt=dt)
            # Save data (for some reason?)
            orbit_batch.append(np.array(orbit))
        #     X_batch.append(torch.tensor(Dtrain[ii].astype(float),dtype=torch.float32))
        #     y_batch.append(torch.tensor(Rtrain[ii].astype(float),dtype=torch.float32).reshape(1))
        #     saveX.append(Dtrain[ii])
        #     saveY.append(Rtrain[ii])
        # X_batch = np.array(X_batch)
        # y_batch = torch.tensor(y_batch).reshape(len(y_batch),1)
        orbitmod = [orbit[0],orbit[1],orbit[2]*d2r,orbit[3]*d2r,orbit[4]*d2r,orbit[5]*d2r]
        # forward pass
        # y_pred = model(X_batch)
        # loss = loss_fn(y_pred1, y_batch)
        # # backward pass
        # # optimizer1.zero_grad() # should this be optimizer or model?
        # # model.zero_grad()
        # loss.backward()
        # # update weights
        # optimizer.step()
    if epoch > 0:
        X_batch = np.array(X_batch)
        R_batch = np.array(R_batch)
        sv_batch = np.array(sv_batch)
        orbit_batch = np.array(orbit_batch)
        nnkf.trainBatch(X_batch,R_batch,dt=dt)
        nnpred.trainBatch(X_batch,sv_batch,dt=dt)
        nnkfr.trainBatch(X_batch,R_batch,dt=dt)
        nnkfz.trainBatch(X_batch,R_batch,dt=dt)
        nnkfp.trainBatch(X_batch,sv_batch,dt=dt)
        nnkfo.trainBatch(X_batch,orbit_batch,dt=dt)
    nnkf.dnn.eval()
    orbitsave = []
    for jj in range(test_size):
        dt = np.random.randint(5,61)
        obsrvr,reality,orbit = TLEtest.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs)
        orbitmod = [orbit[0],orbit[1],orbit[2]*d2r,orbit[3]*d2r,orbit[4]*d2r,orbit[5]*d2r]
        observer = np.transpose(obsrvr)
        nnkf.setStartTime(observer[0][0])
        nnpred.setStartTime(observer[0][0])
        nnkfz.setStartTime(observer[0][0])
        nnkfr.setStartTime(observer[0][0])
        nnkfp.setStartTime(observer[0][0])
        nnkfo.setStartTime(observer[0][0])
        realtle = reality[1]
        reals = np.transpose(reality[0])
        siderealtime = TLEtest.OD.siderealTime(observer[0],TLE.location)
        Dtrain,Rtrain = Dmanip.grabInputNoNorm(observer,siderealtime,dataseting)
        gausoln3 = orbdet.GaussOrbitDetermination(observer[0],[observer[6].astype(float),observer[7].astype(float)],uccs,radecOrazel=0)
        nnkf.filter.x = np.append(gausoln3[1],gausoln3[2])
        nnkf.setStartTime(observer[0][0])
        # Xrft = rft.predict(Dtrain)
        # RftErr = np.sqrt(np.mean(np.square(Rtrain - Xrft)))
        # historyRFT.append(RftErr)
        # Errrft = abs(Xrft - Rtrain)
        relerr = []
        relerr2 = []
        relerrz = []
        relerrr = []
        relerrp = []
        relerro = []
        errRatio1 = []
        errRatio2 = []
        for ii in range(len(Dtrain)):
            xpred = nnkf.forward(Dtrain[ii],dt=dt)
            xpred2 = nnpred.nn_only(Dtrain[ii])
            xpredz = nnkfz.forward(Dtrain[ii],dt=dt)
            xpredr = nnkfr.forward(Dtrain[ii],dt=dt)
            xpredp = nnkfp.forward(Dtrain[ii],dt=dt)
            xpredo = nnkfo.forward(Dtrain[ii],dt=dt)
            err = abs(np.dot(xpred,reality[ii])/np.linalg.norm(reality[ii])**2-1)
            err2 = abs(np.dot(xpred2,reality[ii])/np.linalg.norm(reality[ii])**2-1)
            err3 = abs(np.dot(xpredz,reality[ii])/np.linalg.norm(reality[ii])**2-1)
            err4 = abs(np.dot(xpredr,reality[ii])/np.linalg.norm(reality[ii])**2-1)
            err5 = abs(np.dot(xpredp,reality[ii])/np.linalg.norm(reality[ii])**2-1)
            erro = abs(np.dot(xpredo,reality[ii])/np.linalg.norm(reality[ii])**2-1)
            relerr.append(err)
            relerr2.append(err2)
            relerrz.append(err3)
            relerrr.append(err4)
            relerrp.append(err5)
            relerro.append(erro)
            positionErrSave.append(err)
            errg = abs(np.dot(np.append(gausoln3[1],gausoln3[2]),reality[ii])/np.linalg.norm(reality[ii])**2-1)
            gaussErr.append(errg)
            errGauss.append(np.dot(np.append(gausoln3[1],gausoln3[2]),reality[ii])/(np.linalg.norm(reality[ii])**2))
            CovarianceSave.append(np.linalg.norm(nnkf.filter.P))
            saveX.append(torch.tensor(Dtrain[ii].astype(float),dtype=torch.float32))
            saveY.append(torch.tensor(Rtrain[ii].astype(float),dtype=torch.float32).reshape(1))
            orbitsave.append(orbitmod)
            XpredSave.append(xpred)
        # positionErrSave.append(rms)
    if epoch == 0:
        bestErr1 = np.mean(relerr)
        bestErr2 = np.mean(relerr2)
        bestErr3 = np.mean(relerrz)
        bestErr4 = np.mean(relerrr)
        bestErr5 = np.mean(relerrp)
        bestErr6 = np.mean(relerro)
    print('Relative errors:',np.mean(relerr),np.mean(relerr2),np.mean(relerrz),np.mean(relerrr),np.mean(relerrp),np.mean(relerro))
    if np.mean(relerr) < bestErr1:
        best_weights = copy.deepcopy(nnkf.dnn.state_dict())
        bestErr1 = np.mean(relerr)
        print("Saving NNKF weights",bestErr1)
    if np.mean(relerr2) < bestErr2:
        best_weights2 = copy.deepcopy(nnpred.dnn.state_dict())
        bestErr2 = np.mean(relerr2)
        print("Saving NNOE weights",bestErr2)
    if np.mean(relerrz) < bestErr3:
        best_weights3 = copy.deepcopy(nnkfz.dnn.state_dict())
        bestErr3 = np.mean(relerrz)
        print("Saving NNKFz weights",bestErr3)
    if np.mean(relerrr) < bestErr4:
        best_weights4 = copy.deepcopy(nnkfr.dnn.state_dict())
        bestErr4 = np.mean(relerrr)
        print("Saving NNKFr weights",bestErr4)
    if np.mean(relerrp) < bestErr5:
        best_weights5 = copy.deepcopy(nnkfp.dnn.state_dict())
        bestErr5 = np.mean(relerrp)
        print("Saving NNKFp weights",bestErr5)
    if np.mean(relerro) < bestErr6:
        best_weights6 = copy.deepcopy(nnkfo.dnn.state_dict())
        bestErr6 = np.mean(relerro)
        print("Saving NNKFo weights",bestErr6)
    print('Finished epoch',epoch,'of',n_epochs-1)
if n_epochs > 0: # some training was accomplished
    try:
        torchNN.torch.save(best_weights,'models/KFNN_v0_dnn_'+str(dataseting)+save_append+'bestweights')
        print('Saved KFNN')
        nnkf.loadNN('models/KFNN_v0_dnn_'+str(dataseting)+save_append+'bestweights')
    except:
        print("KFNN best weights faulty.  Proceed with caution.")
    try:
        torchNN.torch.save(best_weights3,'models/KFNN_v2_dnn_'+str(dataseting)+save_append+'bestweights')
        print('Saved KFNNz')
        nnkfz.loadNN('models/KFNN_v2_dnn_'+str(dataseting)+save_append+'bestweights')
    except:
        print("KFNNz best weights faulty.  Proceed with caution.")
    try:
        torchNN.torch.save(best_weights2,'models/KFNN_v4_dnn_'+str(dataseting)+save_append+'bestweights')
        print('Saved OENN')
        nnpred.loadNN('models/KFNN_v4_dnn_'+str(dataseting)+save_append+'bestweights')
    except:
        print("OENN best weights faulty.  Proceed with caution.")
    try:
        torchNN.torch.save(best_weights5,'models/KFNN_v3_dnn_'+str(dataseting)+save_append+'bestweights')
        print('Saved KFNNp')
        nnkfp.loadNN('models/KFNN_v3_dnn_'+str(dataseting)+save_append+'bestweights')
    except:
        print("KFNNp best weights faulty.  Proceed with caution.")
    try:
        torchNN.torch.save(best_weights4,'models/KFNN_v1_dnn_'+str(dataseting)+save_append+'bestweights')
        print('Saved KFNNr')
        nnkfr.loadNN('models/KFNN_v1_dnn_'+str(dataseting)+save_append+'bestweights')
    except:
        print("KFNNr best weights faulty.  Proceed with caution.")
    try:
        torchNN.torch.save(best_weights6,'models/KFNN_v5_dnn_'+str(dataseting)+save_append+'bestweights')
        print('Saved KFNNo')
        nnkfo.loadNN('models/KFNN_v5_dnn_'+str(dataseting)+save_append+'bestweights')
    except:
        print("KFNNo best weights faulty.  Proceed with caution.")
    nnkf.saveNN('models/KFNN_v0_dnn_'+str(dataseting)+save_append)
    nnpred.saveNN('models/KFNN_v4_dnn_'+str(dataseting)+save_append)
    nnkfz.saveNN('models/KFNN_v2_dnn_'+str(dataseting)+save_append)
    nnkfr.saveNN('models/KFNN_v1_dnn_'+str(dataseting)+save_append)
    nnkfp.saveNN('models/KFNN_v3_dnn_'+str(dataseting)+save_append)
    nnkfo.saveNN('models/KFNN_v5_dnn_'+str(dataseting)+save_append)


print('Testing')
nnkf.dnn.eval()
nnpred.dnn.eval()
nnkfz.dnn.eval()
nnkfr.dnn.eval()
dataEvalX = []
dataEvalY = []
XpredEval = []
CovarianceEval = []
positionErrEval = []
positionErrEval2 = []
nnkZEval = []
nnkREval = []
gaussEval = []
orbiteval = []
KFNNeval = []
NNOEeval = []
NNKReval = []
NNKZeval = []
NNKPeval = []
NNKOeval = []
SVeval = []
Rtrue = []
gauseval = []
dataSaveX = []
dataSaveY = []
history = []
errGauss = []
errNNpred = []
errKFNN = []
errKNNR = []
errKNNZ = []
errKNNP = []
errKNNO = []
Reval = []
NNKFrange = []
NNKZrange = []
NNKRrange = []
positionErrEvalR = []
positionErrEvalZ = []
positionErrEvalP = []
positionErrEvalO = []
ranges = []
for jj in range(eval_size):
    dt = np.random.randint(5,61)
    obsrvr,reality,orbit = TLEtest.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs)
    orbitmod = [orbit[0],orbit[1],orbit[2]*d2r,orbit[3]*d2r,orbit[4]*d2r,orbit[5]*d2r]
    observer = np.transpose(obsrvr)
    ranges.append(observer[3].astype(np.float32))
    realtle = reality[1]
    reals = np.transpose(reality[0])
    siderealtime = TLEtest.OD.siderealTime(observer[0],TLE.location)
    Dtrain,Rtrain = Dmanip.grabInputNoNorm(observer,siderealtime,dataseting)
    gausoln3 = orbdet.GaussOrbitDetermination(observer[0],[observer[6].astype(float),observer[7].astype(float)],uccs,radecOrazel=0)
    gausoln3_2 = orbdet.GaussOrbitDetermination(observer[0][-4:],[observer[6][-4:].astype(float),observer[7][-4:].astype(float)],uccs,radecOrazel=0)
    nnkf.filter.x = np.append(gausoln3[1],gausoln3[2])
    nnkf.setStartTime(observer[0][0])
    # Xrft = rft.predict(Dtrain)
    # RftErr = np.sqrt(np.mean(np.square(Rtrain - Xrft)))
    # historyRFT.append(RftErr)
    # Errrft = abs(Xrft - Rtrain)
    Xbatch = []
    relerr = []
    relerr2 = []
    rmsR = []
    rmsZ = []
    # errRatio1 = []
    # errRatio2 = []
    # errRatioR = []
    # errRatioZ = []
    gaussRatio = []
    y_batch = []
    nnkfPred = []
    nnoePred = []
    nnkZpred = []
    nnkRpred = []
    nnkPpred = []
    nnkOpred = []
    rpred = []
    rpredr = []
    rpredz = []
    for ii in range(len(Dtrain)):
        xpred = nnkf.forward(Dtrain[ii],dt=dt)
        xpred2 = nnpred.forward(Dtrain[ii],dt=dt)
        xpredr = nnkfr.forward(Dtrain[ii],dt=dt)
        xpredz = nnkfz.forward(Dtrain[ii],dt=dt)
        xpredp = nnkfp.forward(Dtrain[ii],dt=dt)
        xpredo = nnkfo.forward(Dtrain[ii],dt=dt)
        nnkfPred.append(xpred)
        nnoePred.append(xpred2)
        nnkZpred.append(xpredz)
        nnkRpred.append(xpredr)
        nnkPpred.append(xpredp)
        nnkOpred.append(xpredo)
        rpred.append(nnkf.nn_only(Dtrain[ii],dt=dt))
        rpredr.append(nnkfr.nn_only(Dtrain[ii],dt=dt))
        rpredz.append(nnkfz.nn_only(Dtrain[ii],dt=dt))
        err = np.linalg.norm(xpred-reality[ii])
        err2 = np.linalg.norm(xpred2-reality[ii])
        relerr1 = np.dot(xpred,reality[ii])/(np.linalg.norm(reality[ii])**2)-1
        relerr2 = np.dot(xpred2,reality[ii])/(np.linalg.norm(reality[ii])**2)-1
        relerrR = np.dot(xpredr,reality[ii])/(np.linalg.norm(reality[ii])**2)-1
        relerrZ = np.dot(xpredz,reality[ii])/(np.linalg.norm(reality[ii])**2)-1
        relerrP = np.dot(xpredp,reality[ii])/(np.linalg.norm(reality[ii])**2)-1
        relerrO = np.dot(xpredo,reality[ii])/(np.linalg.norm(reality[ii])**2)-1
        positionErrEvalR.append(np.linalg.norm(xpredr-reality[ii]))
        positionErrEvalZ.append(np.linalg.norm(xpredz-reality[ii]))
        positionErrEvalP.append(np.linalg.norm(xpredp-reality[ii]))
        positionErrEvalO.append(np.linalg.norm(xpredo-reality[ii]))
        # errRatio1.append(relerr1)
        # errRatio2.append(relerr2)
        # errRatioR.append(relerrR)
        # errRatioZ.append(relerrZ)
        errKFNN.append(relerr1)
        errNNpred.append(relerr2)
        errKNNR.append(relerrR)
        errKNNZ.append(relerrZ)
        errKNNP.append(relerrP)
        errKNNO.append(relerrO)
        positionErrEval.append(err)
        positionErrEval2.append(err2)
        gaussEval.append(np.linalg.norm(np.append(gausoln3[1],gausoln3[2])-reality[1]))
        gauserr = np.dot(np.append(gausoln3[1],gausoln3[2]),reality[ii])/(np.linalg.norm(reality[ii])**2)
        errGauss.append(gauserr)
        gaussRatio.append(gauserr)
        CovarianceEval.append(np.linalg.norm(nnkf.filter.P))
        Xbatch.append(torch.tensor(Dtrain[ii].astype(float),dtype=torch.float32))
        y_batch.append(torch.tensor(Rtrain[ii].astype(float),dtype=torch.float32).reshape(1))
        orbitsave.append(orbitmod)
        XpredEval.append(xpred)
    SVeval.append(Xbatch)
    Reval.append(y_batch)
    Rtrue.append(reality)
    KFNNeval.append(nnkfPred)
    NNOEeval.append(nnoePred)
    NNKReval.append(nnkRpred)
    NNKZeval.append(nnkZpred)
    NNKPeval.append(nnkPpred)
    NNKOeval.append(nnkOpred)
    # gauseval.append([gausoln3[1],gausoln3_2[1]])
    gauseval.append([np.append(gausoln3[1],gausoln3[2])])
    orbiteval.append(orbitmod)
    NNKFrange.append(rpred)
    NNKRrange.append(rpredr)
    NNKZrange.append(rpredz)


if n_epochs > 0:
#     plt.figure()
#     plt.plot(gaussErr)
#     plt.plot(positionErrEval,alpha=0.7)
#     # plt.title('RMS error')
#     # plt.axis([0,len(positionErrSave),0,1e6])
#     plt.legend(['Guass\'s','NN-KF'])
#     plt.ylabel('Root-mean-square error (km)')
#     plt.axis([0,len(gaussErr),0,np.max([np.mean(positionErrSave) + np.std(positionErrSave)*3,np.max(gaussErr)])])
#     plt.savefig('plots/kfnn_position_rms_history_'+str(dataseting)+save_append+'.png')
    
#     plt.figure()
#     plt.plot(gaussErr)
#     plt.plot(positionErrEvalR,alpha=0.7)
#     # plt.title('RMS error')
#     # plt.axis([0,len(positionErrSave),0,1e6])
#     plt.legend(['Guass\'s','NN-KF'])
#     plt.ylabel('Root-mean-square error (km)')
#     plt.axis([0,len(gaussErr),0,np.max([np.mean(positionErrEvalR) + np.std(positionErrEvalR)*3,np.max(gaussErr)])])
#     plt.savefig('plots/kfnn_R_rms_history_'+str(dataseting)+save_append+'.png')
    
#     plt.figure()
#     plt.plot(gaussErr)
#     plt.plot(positionErrEvalZ,alpha=0.7)
#     # plt.title('RMS error')
#     # plt.axis([0,len(positionErrSave),0,1e6])
#     plt.legend(['Guass\'s','NN-KF'])
#     plt.ylabel('Root-mean-square error (km)')
#     plt.axis([0,len(gaussErr),0,np.max([np.mean(positionErrEvalZ) + np.std(positionErrEvalZ)*3,np.max(gaussErr)])])
#     plt.savefig('plots/kfnn_Z_rms_history_'+str(dataseting)+save_append+'.png')

#     plt.figure()
#     plt.plot(gaussErr)
#     plt.plot(positionErrEvalP,alpha=0.7)
#     # plt.title('RMS error')
#     # plt.axis([0,len(positionErrSave),0,1e6])
#     plt.legend(['Guass\'s','NN-KF'])
#     plt.ylabel('Root-mean-square error (km)')
#     plt.axis([0,len(gaussErr),0,np.max([np.mean(positionErrEvalP) + np.std(positionErrEvalP)*3,np.max(gaussErr)])])
#     plt.savefig('plots/kfnn_P_rms_history_'+str(dataseting)+save_append+'.png')

    plt.figure()
    plt.plot(CovarianceEval)
    plt.title('Covariance magnitude')
    plt.ylabel('Root-mean-square Covariance')
    plt.savefig('plots/kfnn_covariance_history_'+str(dataseting)+save_append+'.png')

ranges = np.array(ranges)
NNKFrange = np.array(NNKFrange)
NNKZrange = np.array(NNKZrange)
NNKRrange = np.array(NNKRrange)
SV1 = np.transpose(KFNNeval[-1])
SV2 = np.transpose(NNOEeval[-1])
SVR = np.transpose(NNKReval[-1])
SVZ = np.transpose(NNKZeval[-1])
SVP = np.transpose(NNKPeval[-1])
SVO = np.transpose(NNKOeval[-1])
R1 = np.transpose(Rtrue[-1])
G3 = np.transpose(gauseval[-1])
plt.figure()
plt.plot(R1[0],'r.')
plt.plot(G3[0],'r*')
plt.plot(SVR[0],'r:')
plt.plot(SVZ[0],'r--')
plt.plot(SV1[0],'r-')
plt.plot(SV2[0],'r-.')
plt.plot(SVP[0],'r',linestyle=(0, (5, 5)))
plt.plot(SVO[0],'r',linestyle=(0, (3, 5, 1, 5)))
plt.plot(R1[1],'g.')
plt.plot(G3[1],'g*')
plt.plot(SVR[1],'g:')
plt.plot(SVZ[1],'g--')
plt.plot(SV1[1],'g')
plt.plot(SV2[1],'g-.')
plt.plot(SVP[1],'g',linestyle=(0, (5, 5)))
plt.plot(SVO[1],'g',linestyle=(0, (3, 5, 1, 5)))
plt.plot(R1[2],'b.')
plt.plot(G3[2],'b*')
plt.plot(SVR[2],'b:')
plt.plot(SVZ[2],'b--')
plt.plot(SV1[2],'b')
plt.plot(SV2[2],'b-.')
plt.plot(SVP[2],'b',linestyle=(0, (5, 5)))
plt.plot(SVO[2],'b',linestyle=(0, (3, 5, 1, 5)))
plt.ylabel('Magnitude (km)')
plt.xlabel('Observation number')
plt.legend(['Truth','Gauss\'s','NNKF-R','NNKF-Z','NN-KF','NN-OE','NNKF-P','NNKF-O'])
# plt.title('XYZ position comparison')
# plt.plot(R1[0]*0+G3[0],'r*')
# plt.plot(R1[1]*0+G3[1],'g*')
# plt.plot(R1[2]*0+G3[2],'b*')
plt.savefig('plots/kfnn_track_sample_'+str(dataseting)+save_append+'.png')

plt.figure()
plt.plot(ranges.flatten(),'k:')
plt.plot(NNKFrange.flatten(),'r-',alpha=.9)
plt.plot(NNKRrange.flatten(),'g-.',alpha=.7)
plt.plot(NNKZrange.flatten(),'b--',alpha=.5)
plt.legend(['Truth','NNKF','NNKFr','NNKF-Z'])
plt.ylabel('Range estimate (km)')
plt.title('Range estimation')
plt.savefig('plots/kfnn_range_sample_'+str(dataseting)+save_append+'.png')

plt.figure()
plt.plot(NNKFrange.flatten()-ranges.flatten(),'r-')
plt.plot(NNKRrange.flatten()-ranges.flatten(),'g-.',alpha=.8)
plt.plot(NNKZrange.flatten()-ranges.flatten(),'b--',alpha=.6)
plt.legend(['NNKF','NNKFr','NNKF-Z'])
plt.ylabel('Range error (km)')
plt.title('Range error')
plt.savefig('plots/kfnn_range_error_'+str(dataseting)+save_append+'.png')

plt.figure()
plt.subplot(3,1,1)
plt.plot(R1[0],'k.')
plt.plot(G3[0],'g*')
plt.plot(SV1[0],'b')
plt.plot(SVR[0],'r:')
plt.plot(SVZ[0],'r--')
plt.plot(SVP[0],'r-.')
# plt.plot(SV2[0],'b-.')
plt.subplot(3,1,2)
plt.plot(R1[1],'k.')
plt.plot(G3[1],'g*')
plt.plot(SV1[1],'b')
plt.plot(SVR[1],'r:')
plt.plot(SVZ[1],'r--')
plt.plot(SVP[1],'r-.')
# plt.plot(SV2[1],'b-.')
plt.subplot(3,1,3)
plt.plot(R1[2],'k.')
plt.plot(G3[2],'g*')
plt.plot(SV1[2],'b')
plt.plot(SVR[2],'r:')
plt.plot(SVZ[2],'r--')
plt.plot(SVP[2],'r-.')
# plt.plot(SV2[2],'b-.')
plt.ylabel('Magnitude (km)')
plt.xlabel('Observation number')
plt.legend(['Truth','Gauss\'s','NN-KF','NNKF-R','NNKF-Z','NNKF-P'])#,'NN-OE'])
# plt.title('XYZ position comparison')
# plt.plot(R1[0]*0+G3[0],'r*')
# plt.plot(R1[1]*0+G3[1],'g*')
# plt.plot(R1[2]*0+G3[2],'b*')
plt.savefig('plots/kfnn_track_3view_'+str(dataseting)+save_append+'.png')

errKFNN = np.array(errKFNN)
errNNpred = np.array(errNNpred)
errGauss = np.array(errGauss)
errKNNZ = np.array(errKNNZ)
errKNNR = np.array(errKNNR)
errKNNP = np.array(errKNNP)
bin = np.linspace(-3e0,3e0,100)
plt.figure()
hG = plt.hist(errGauss,bins=bin,alpha=0.5)
hKFNN = plt.hist(errKFNN,bins=bin,alpha=0.5)
hKNNR = plt.hist(errKNNR,bins=bin,alpha=0.5)
hKNNZ = plt.hist(errKNNZ,bins=bin,alpha=0.5)
hNNOE = plt.hist(errNNpred,bins=bin,alpha=0.5)
hKNOP = plt.hist(errKNNP,bins=bin,alpha=0.5)
diffadd = np.mean(hG[1][1:] - hG[1][0:-1])/2
plt.plot(hG[1][0:-1]+diffadd,hG[0],'sk')
plt.plot(hKFNN[1][0:-1]+diffadd,hKFNN[0],'oC0')
plt.plot(hKNNR[1][0:-1]+diffadd,hKNNR[0],'vC2')
plt.plot(hKNNZ[1][0:-1]+diffadd,hKNNZ[0],'dC3')
plt.plot(hNNOE[1][0:-1]+diffadd,hNNOE[0],'*C4')
plt.plot(hKNOP[1][0:-1]+diffadd,hKNOP[0],'.C5')
plt.xlabel('Relative error')
plt.legend(['Gauss','NN-KF','NNKF-R','NNKF-Z','NN-OE','NNKF-P'])
plt.savefig('plots/gauss_vs_kfnn_vs_oenn_histogram'+str(dataseting)+save_append+'.png')

plt.figure()
# plt.plot(np.ones(len(gaussRatio)),alpha=0.5)
plt.plot(errGauss-1,alpha=0.7)
plt.plot(errKFNN-1,alpha=0.5)
plt.plot(errKNNR-1,alpha=0.4)
plt.plot(errKNNZ-1,alpha=0.4)
plt.plot(errNNpred-1,alpha=0.4)
plt.plot(errKNNP-1,alpha=0.4)
plt.xlabel('Relative error')
plt.legend(['Gauss','NN-KF','NNKF-R','NNKF-Z','NNKF-P'])#,'NN-OE'])
plt.axis([0,len(errGauss),np.min([errGauss-2,errKFNN-2]),np.max([errGauss,errKFNN])])
plt.savefig('plots/gauss_vs_kfnn_vs_oenn'+str(dataseting)+save_append+'.png')

print('Gauss\'s error:',np.mean(np.abs(gaussEval)),'NN+KF error:',np.mean(np.abs(positionErrEval)),'NN-OE error:',np.mean(np.abs(positionErrEval2)),
      'NNKF-R error',np.mean(np.abs(positionErrEvalR)),'NNKF-Z',np.mean(np.abs(positionErrEvalZ)),'NNKF-P',np.mean(np.abs(positionErrEvalP)),
      'NNKF-O',np.mean(np.abs(positionErrEvalO)))

# b = np.linspace(0,1e5,101)
# plt.figure()
# # hnnkf = plt.hist(positionErrEval,bins=b,alpha=0.5)
# # hg = plt.hist(gaussEval,bins=b,alpha=0.5)
# plt.hist(gaussEval,bins=b,alpha=0.5)
# plt.hist(positionErrEval,bins=b,alpha=0.5)
# plt.hist(positionErrEval2,bins=b,alpha=0.5)
# # plt.bar(b[0:-1],hnnkf[0],width=0.5)
# # plt.bar(b[0:-1],hg[0],width=0.5,align="edge")
# plt.xlabel('Error RMS (km)')
# plt.ylabel('Instances')
# plt.legend(['Gauss\'s','NN-KF','NN'])
# plt.savefig('plots/kfnn_vs_gauss_vs_nn_err_hist_'+str(dataseting)+save_append+'.png')

# plt.figure()
# # hnnkf = plt.hist(positionErrEval,bins=b,alpha=0.5)
# # hg = plt.hist(gaussEval,bins=b,alpha=0.5)
# plt.hist(gaussEval,bins=b,alpha=0.5)
# plt.hist(positionErrEval,bins=b,alpha=0.5)
# # plt.bar(b[0:-1],hnnkf[0],width=0.5)
# # plt.bar(b[0:-1],hg[0],width=0.5,align="edge")
# plt.xlabel('Error RMS (km)')
# plt.ylabel('Instances')
# plt.legend(['Gauss\'s','NN-KF'])
# plt.savefig('plots/kfnn_vs_gauss_err_hist_'+str(dataseting)+save_append+'.png')

# plt.figure()
# plt.plot(gaussEval)
# plt.plot(positionErrEval,alpha=0.7)
# plt.title('RMS error')
# # plt.axis([0,len(positionErrEval),0,1e6])
# plt.legend(['Guass\'s','NN-KF'])
# plt.ylabel('Root-mean-square error (km)')
# plt.savefig('plots/kfnn_position_rms_eval_'+str(dataseting)+save_append+'.png')

plt.figure()
plt.plot(CovarianceEval)
plt.title('Covariance magnitude')
plt.ylabel('Root-mean-square Covariance')
plt.savefig('plots/kfnn_covariance_eval_'+str(dataseting)+save_append+'.png')

plt.show()

