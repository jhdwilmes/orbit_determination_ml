import numpy as np
import torch
# import tqdm
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import pipeline
from sklearn import impute
from sklearn import compose
from sklearn import neural_network
import joblib
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import sigma_points
import time

import torch.nn as nn
import torch.optim as optim

# import neural_filter
import torchNN
import OrbitDetermination as OD
import orbitFunctions
import kfnn
import test_functions

import createDataLive

import matplotlib as mpl
from matplotlib import pyplot as plt


dataseting = 6
dataseting1 = 4
dataseting2 = 5
dnntype = 0
dnntype1 = 0
dnntype2 = 2
test_size = 1000
dt = 10
num_tles = 10000
obs_noise = .004 # .02 is 1.2 arcminutes, .004 ~15 arcseconds, .0003 ~1.1 arcseconds
max_obs = 30
appender = ''#'5second'#'-1'
create_additional_tles = False
dohist = 0
tmin = 1
tmax = 10
label = '1-10second'
tarray = []#[1,5,10,30,60]
usemean = False

datalegnths = {0:5,1:10,2:9,3:18,4:10,5:20,6:30}
datasetings = {0:'dnnbestweights',2:'encoderbestweights'}
datalength = datalegnths[dataseting]
datalength1 = datalegnths[dataseting1]
datalength2 = datalegnths[dataseting2]
print("Running with input sizes",datalength,"and",datalength1,'plots saved as',label)
d2r = np.pi/180

nn_append = 'lownoise' #'lownoise'

modelO = torchNN.Dense91(datalength,6,(100,100)) #torchNN.Trnsfrmr1(datalength,6,(datalength,datalength))
modelO.load_state_dict(torch.load('models/Orbit_Predict_bestweights_inputsize_'+str(dataseting),weights_only=True))
nnkf0 = kfnn.KFNN_v0(dimin=datalength1,dnntype=dnntype1)
nnkf0.loadNN('models/KFNN_v0_dnn_'+str(dataseting1)+datasetings[dnntype1])
nnkf02 = kfnn.KFNN_v0(dimin=datalength1,dnntype=dnntype2) # NN predicts range
nnkf02.loadNN('models/KFNN_v0_dnn_'+str(dataseting1)+datasetings[dnntype2])
# nnkf = kfnn.KFNN_v1(dimin=datalength1,dnntype=dnntype1)
# nnkf.loadNN('models/KFNN_v1_dnn_'+str(dataseting1)+datasetings[dnntype1])
nnkf = kfnn.KFNN_v2(dimin=datalength1,dnntype=dnntype1,usezdiff=True)
nnkf2 = kfnn.KFNN_v2(dimin=datalength,dnntype=dnntype)
nnkf1 = kfnn.KFNN_v2(dimin=datalength2,dnntype=dnntype) # NN predicts range, includes KF Z in input
nokf = kfnn.KFNN_v3(dimin=datalength1,dnntype=dnntype1,usezerr=True) # NN predicts SV, includes KF Z in input
oenn = kfnn.KFNN_v4(dimin=datalength,dnntype=dnntype) # NN predicts SV
nnkfo = kfnn.KFNN_v5(dimin=datalength1,dnntype=dnntype1,usez=True) # NN predicts OE, may include KF Z in input
nnkf.loadNN('models/KFNN_v2_dnn_'+str(dataseting1)+datasetings[dnntype1])
nnkf2.loadNN('models/KFNN_v2_dnn_'+str(dataseting)+datasetings[dnntype])
nnkf1.loadNN('models/KFNN_v2_dnn_'+str(dataseting2)+datasetings[dnntype])
nokf.loadNN('models/KFNN_v3_dnn_'+str(dataseting1)+datasetings[dnntype1])
oenn.loadNN('models/KFNN_v4_dnn_'+str(dataseting)+datasetings[dnntype1])
nnkfo.loadNN('models/KFNN_v5_dnn_'+str(dataseting1)+datasetings[dnntype1])

modelR = torchNN.Dense91(datalength1,1,(100,100))
modelR.load_state_dict(torch.load('models/range_nn_'+str(dataseting1)+'bestweights'+nn_append,weights_only=True))

orbitForests = [
    joblib.load('models/gradboost_MM_predictor_'+str(dataseting)+appender),
    joblib.load('models/gradboost_Ecc_predictor_'+str(dataseting)+appender),
    joblib.load('models/gradboost_Inc_predictor_'+str(dataseting)+appender),
    joblib.load('models/gradboost_RAAN_predictor_'+str(dataseting)+appender),
    joblib.load('models/gradboost_AP_predictor_'+str(dataseting)+appender),
    joblib.load('models/gradboost_MA_predictor_'+str(dataseting)+appender)
]

rangeForest = joblib.load('models/gradientboost_save_'+str(dataseting)+appender)

TF = test_functions.testML()
# TF.TLE.createRandomOrbit(num_tles)
TF.TLE.readTLEfile('tles/tle_11052024.txt')
TF.TLE.prunTLEs([.001,15])
num_tles = len(TF.TLE.tles)

points = sigma_points.MerweScaledSigmaPoints(6, alpha=.01, beta=2., kappa=-3)
UKF  = UnscentedKalmanFilter(dim_x=6, dim_z=5, dt=dt, fx=TF.OF.orbitFunction6, hx=TF.OF.orbitSensorModelRt, points=points)
UKF.P = np.diag([1e6,1e6,1e6,1e2,1e2,1e2])
UKF.R = np.diag([obs_noise,obs_noise,1e2,obs_noise*2,obs_noise*2])
datakf = []
datann = []
# for ii in range(len(kfmodels)):
#     datakf.append([])
# for ii in range(len(nnmodels)):
#     datann.append([])
GaussData = []
GaussDots = []
KFNN0data = []
KFNN0dots = []
KFNNdata = []
KFNNdots = []
KFNN2data = []
KFNN2dots = []
KFNN3data = []
KFNN3dots = []
NNOEdata = []
NNOEdots = []
NOKFdata = []
NOKFdots = []
KFNOdata = []
KFNOdots = []
orbitNNdata = []
orbitNNdots = []
rangeforestData = []
rangeforestDots = []
orbitforestData = []
orbitforestDots = []
rangeNNdata = []
rangeNNdots = []
dtsave = []
realdata = []
dataKFNN = []
dataKFNNZ = []
dataRFR = []
dataRange = []
gaussRange = []
encoderData = []
encoderDots = []
encoderRange = []
RFdirectData = []
RFdirectDots = []
t1 = time.time()
for ii in range(test_size):
    if create_additional_tles:
        TF.TLE.createRandomOrbit()
    if len(tarray) == 0:
        dt = np.random.randint(tmin,tmax+1)
    else:
        dt = tarray[np.random.randint(0,len(tarray))]
    observer,reality,orbit = TF.TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,minObs=max_obs,maxObs=max_obs)
    observer = np.transpose(observer)
    TF.OF.time = observer[0][0]
    dtsave.append(dt)
    x,dx,O,dO,xp,Op = TF.organizeTestData(reality,observer,reality)
    realdata.append(np.transpose([x,dx,O,dO]))
    dataRange.append(np.copy(observer[3]))
    # 
    gausoln = TF.orbdet.GaussOrbitDetermination(observer[0],[observer[6].astype(float),observer[7].astype(float)],TF.location,radecOrazel=0)
    # gausoln = TF.orbdet.GaussOrbitDetermination(observer[0],[observer[4].astype(float),observer[5].astype(float)],TF.location,radecOrazel=0)
    # gausoln = TF.orbdet.GaussOrbitDetermination(observer[0],[observer[1].astype(float),observer[2].astype(float)],TF.location,radecOrazel=1)
    guasses = []
    gr = []
    for jj in range(3,len(observer[0])):
        iterlist = [0,int(np.floor(jj/2)),jj-1]
        gausoln = TF.orbdet.GaussOrbitDetermination(observer[0],[observer[6].astype(float),observer[7].astype(float)],TF.location,radecOrazel=0,iters=iterlist)
        gausrange = TF.orbdet.GaussOrbitDetermination(observer[0],[observer[6].astype(float),observer[7].astype(float)],TF.location,radecOrazel=0,iters=iterlist,onlyRange=True)
        guasses.append(np.append(gausoln[1],gausoln[2]))
        gr.append(gausrange)
    gaussRange.append(gr)
    x,dx,O,dO,xp,Op = TF.organizeTestData(guasses,observer,reality)
    GaussData.append(np.transpose([x,dx,O,dO]))
    GaussDots.append(np.transpose([xp,Op]))
    # 
    UKF.x = np.append(gausoln[1],gausoln[2])
    datasetkf = []
    # for jj in range(len(kfmodels)):
    output = TF.testKFML(nnkf,dt=dt,xstart=np.append(gausoln[1],gausoln[2]),observer=observer,dataseting=dataseting1)
    x,dx,O,dO,xp,Op = TF.organizeTestData(output,observer,reality)
    KFNNdata.append(np.transpose([x,dx,O,dO]))
    KFNNdots.append(np.transpose([xp,Op]))
    dataKFNNZ.append(TF.testKF_NNonly(nnkf,dt,observer,dataseting1))
    #
    output = TF.testKFML(nnkfo,dt=dt,xstart=np.append(gausoln[1],gausoln[2]),observer=observer,dataseting=dataseting1)
    x,dx,O,dO,xp,Op = TF.organizeTestData(output,observer,reality)
    KFNOdata.append(np.transpose([x,dx,O,dO]))
    KFNOdots.append(np.transpose([xp,Op]))
    # 
    output = TF.testKFML(nnkf0,dt=dt,xstart=np.append(gausoln[1],gausoln[2]),observer=observer,dataseting=dataseting1)
    x,dx,O,dO,xp,Op = TF.organizeTestData(output,observer,reality)
    KFNN0data.append(np.transpose([x,dx,O,dO]))
    KFNN0dots.append(np.transpose([xp,Op]))
    dataKFNN.append(TF.testKF_NNonly(nnkf0,dt,observer,dataseting1))
    #
    output = TF.testKFML(nnkf02,dt=dt,xstart=np.append(gausoln[1],gausoln[2]),observer=observer,dataseting=dataseting1)
    x,dx,O,dO,xp,Op = TF.organizeTestData(output,observer,reality)
    encoderData.append(np.transpose([x,dx,O,dO]))
    encoderDots.append(np.transpose([xp,Op]))
    encoderRange.append(TF.testKF_NNonly(nnkf02,dt,observer,dataseting1))
    #
    output = TF.testKFML(nnkf2,dt=dt,xstart=np.append(gausoln[1],gausoln[2]),observer=observer,dataseting=dataseting)
    x,dx,O,dO,xp,Op = TF.organizeTestData(output,observer,reality)
    KFNN2data.append(np.transpose([x,dx,O,dO]))
    KFNN2dots.append(np.transpose([xp,Op]))
    # 
    output = TF.testKFML(nokf,dt=dt,xstart=np.append(gausoln[1],gausoln[2]),observer=observer,dataseting=dataseting1)
    x,dx,O,dO,xp,Op = TF.organizeTestData(output,observer,reality)
    NOKFdata.append(np.transpose([x,dx,O,dO]))
    NOKFdots.append(np.transpose([xp,Op]))
    # 
    output = TF.testKFML(nnkf1,dt=dt,xstart=np.append(gausoln[1],gausoln[2]),observer=observer,dataseting=dataseting2)
    x,dx,O,dO,xp,Op = TF.organizeTestData(output,observer,reality)
    KFNN3data.append(np.transpose([x,dx,O,dO]))
    KFNN3dots.append(np.transpose([xp,Op]))
    # for jj in range(len(nnmodels)):
    output = TF.testNN(oenn.dnn,dt=dt,observer=observer,dataseting=dataseting)
    output = oenn.outputTranslation(output)
    x,dx,O,dO,xp,Op = TF.organizeTestData(output,observer,reality)
    NNOEdata.append(np.transpose([x,dx,O,dO]))
    NNOEdots.append(np.transpose([xp,Op]))
    UKF.x = np.append(gausoln[1],gausoln[2])
    # for jj in range(len(forestRangeModels)):
    datasetnnrange = []
    output = TF.testNN(modelR,dt=dt,observer=observer,dataseting=dataseting1).flatten()
    z = observer[1:6]
    output = TF.rangetranslation(output,direction=1)
    z1 = np.transpose(z)
    for point in range(len(output)):
        # z2 = np.insert(z1[point],2,output[point])
        z2 = z1[point]
        z2[2] = output[point]
        try:
            UKF.predict(dt=dt)
        except:
            print("Prediction failed, state",UKF.x.flatten(),gausoln[1],gausoln[2])
        try:
            UKF.update(z2)
        except:
            print("Could not update NNR-KF, range",output[point])
        datasetnnrange.append(UKF.x)
    x,dx,O,dO,xp,Op = TF.organizeTestData(datasetnnrange,observer,reality)
    rangeNNdata.append(np.transpose([x,dx,O,dO]))
    rangeNNdots.append(np.transpose([xp,Op]))
    #
    output = TF.testNN(modelO,dt=dt,observer=observer,dataseting=dataseting)
    # output[:,2:] = output[:,2:]/d2r
    x,dx,O,dO,xp,Op = TF.organizeTestData(output,observer,reality,dataType=1)
    orbitNNdata.append(np.transpose([x,dx,O,dO]))
    orbitNNdots.append(np.transpose([xp,Op]))
    # Test forest range prediction
    datasetforestrange = []
    forestranges = []
    UKF.x = np.append(gausoln[1],gausoln[2])
    output = TF.testForest(rangeForest,dt=dt,observer=observer,dataseting=dataseting) # need to set dataseting?
    z = observer[1:6]
    output = TF.rangetranslation(output,direction=1)
    dataRFR.append(output)
    z1 = np.transpose(z)
    rfgResult = []
    for point in range(len(output)):
        # z2 = np.insert(z1[point],2,output[point])
        iterlist = [0,int(np.floor(jj/2)),jj-1]
        rfgOutput = TF.orbdet.range2orbit(output,observer[0],[observer[6].astype(float),observer[7].astype(float)],TF.location,radecOrazel=0,iters=iterlist)
        rfgResult.append(np.array(rfgOutput[1:]).flatten())
        # rfgOrbit = TF.orbdet.StateVector2OrbitalElements4(rfgOutput)
        z2 = z1[point]
        z2[2] = output[point]
        forestranges.append(output[point])
        try:
            UKF.predict(dt=dt)
        except:
            print("Prediction failed, state",UKF.x.flatten(),gausoln[1],gausoln[2])
        try:
            UKF.update(z2)
        except:
            print("Could not update RF-R-KF, range",output[point])
        datasetforestrange.append(UKF.x)
    x,dx,O,dO,xp,Op = TF.organizeTestData(rfgResult,observer,reality)
    RFdirectData.append(np.transpose([x,dx,O,dO]))
    RFdirectDots.append(np.transpose([xp,Op]))
    x,dx,O,dO,xp,Op = TF.organizeTestData(datasetforestrange,observer,reality)
    rangeforestData.append(np.transpose([x,dx,O,dO]))
    rangeforestDots.append(np.transpose([xp,Op]))
    # for jj in range(len(forestOrbitModels)):
    output = TF.testForests(orbitForests,observer=observer,dataseting=dataseting)
    x,dx,O,dO,xp,Op = TF.organizeTestData(output,observer,reality,dataType=1)
    orbitforestData.append(np.transpose([x,dx,O,dO]))
    orbitforestDots.append(np.transpose([xp,Op]))
    if ii % 100 == 0:
        print("Finished iteration",ii,"of",test_size)
dtsave = np.array(dtsave).astype(np.float64)
KFNN0data = np.array(KFNN0data).transpose().astype(np.float64)
KFNN0dots = np.array(KFNN0dots).transpose().astype(np.float64)
KFNNdata = np.array(KFNNdata).transpose().astype(np.float64)
KFNNdots = np.array(KFNNdots).transpose().astype(np.float64)
KFNOdata = np.array(KFNOdata).transpose().astype(np.float64)
KFNOdots = np.array(KFNOdots).transpose().astype(np.float64)
KFNN2data = np.array(KFNN2data).transpose().astype(np.float64)
KFNN2dots = np.array(KFNN2dots).transpose().astype(np.float64)
KFNN3data = np.array(KFNN3data).transpose().astype(np.float64)
KFNN3dots = np.array(KFNN3dots).transpose().astype(np.float64)
NNOEdata = np.array(NNOEdata).transpose().astype(np.float64)
NNOEdots = np.array(NNOEdots).transpose().astype(np.float64)
NOKFdata = np.array(NOKFdata).transpose().astype(np.float64)
NOKFdots = np.array(NOKFdots).transpose().astype(np.float64)
rangeforestData = np.array(rangeforestData).transpose().astype(np.float64)
rangeforestDots = np.array(rangeforestDots).transpose().astype(np.float64)
orbitforestData = np.array(orbitforestData).transpose().astype(np.float64)
orbitforestDots = np.array(orbitforestDots).transpose().astype(np.float64)
GaussData = np.array(GaussData).transpose().astype(np.float64)
GaussDots = np.array(GaussDots).transpose().astype(np.float64)
rangeNNdata = np.array(rangeNNdata).transpose().astype(np.float64)
rangeNNdots = np.array(rangeNNdots).transpose().astype(np.float64)
orbitNNdata = np.array(orbitNNdata).transpose().astype(np.float64)
orbitNNdots = np.array(orbitNNdots).transpose().astype(np.float64)
realdata = np.array(realdata).transpose().astype(np.float64)
dataRFR = np.array(dataRFR).astype(np.float32)
dataKFNN = np.array(dataKFNN).astype(np.float32)
dataKFNNZ = np.array(dataKFNNZ).astype(np.float32)
dataRange = np.array(dataRange).astype(np.float32)
gaussRange = np.array(gaussRange).astype(np.float32)
encoderRange = np.array(encoderRange).astype(np.float32)
encoderData =  np.array(encoderData).astype(np.float32)
encoderDots =  np.array(encoderDots).astype(np.float32)
RFdirectData =  np.array(RFdirectData).transpose().astype(np.float32)
RFdirectDots =  np.array(RFdirectDots).transpose().astype(np.float32)
t2 = time.time()
print("Completed testing, ellapsed time:",t2-t1)

# TF.writeData(KFNNdata,'KFNNdata.csv')
# TF.writeData(NNOEdata,'NNOEdata.csv')
# TF.writeData(rangeforestData,'rangeRFdata.csv')
# TF.writeData(orbitforestData,'orbitRFdata.csv')
# TF.writeData(GaussData,'GaussData.csv')
# TF.writeData(rangeNNdata,'rangeNNdata.csv')

# next - finish producing plots
plt.figure()
plt.plot(dataRange[:,0:max_obs-3].flatten(),'k:')
plt.plot(gaussRange[:,0:max_obs-3].flatten(),'C0-',alpha=0.8)
plt.plot(dataRFR[:,0:max_obs-3].flatten(),'C1--',alpha=0.5)
plt.plot(dataKFNN[:,0:max_obs-3].flatten(),'C2-.',alpha=0.7)
plt.plot(dataKFNNZ[:,0:max_obs-3].flatten(),'C3-',alpha=0.6)
# plt.plot(encoderRange[:,0:max_obs-3].flatten(),'C4-',alpha=0.5)
plt.legend(['Truth','Guass','RF','NN','NN-Z'])#,'Encdr'])
b,t = plt.ylim()
plt.ylim(np.max([b,0]),np.min([t,90000]))
plt.ylabel('Range prediction (km)')
plt.ylabel('Range prediction comparison')
plt.savefig('plots/RangePredictionComparison_'+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'_'+label+'.png')

f = open('plots/range_results.csv','w+')
f.write('method,mean,median,stand-dev,5th,95th\n')
f.write('Gauss,'+str(np.mean(gaussRange[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.median(gaussRange[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.std(gaussRange[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.percentile(gaussRange[:,0:max_obs-3]-dataRange[:,0:max_obs-3],5))+','+str(np.percentile(gaussRange[:,0:max_obs-3]-dataRange[:,0:max_obs-3],95))+'\n')
f.write('RandomForest,'+str(np.mean(dataRFR[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.median(dataRFR[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.std(dataRFR[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.percentile(dataRFR[:,0:max_obs-3]-dataRange[:,0:max_obs-3],5))+','+str(np.percentile(dataRFR[:,0:max_obs-3]-dataRange[:,0:max_obs-3],95))+'\n')
f.write('NeuralNetwork,'+str(np.mean(dataKFNN[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.median(dataKFNN[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.std(dataKFNN[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.percentile(dataKFNN[:,0:max_obs-3]-dataRange[:,0:max_obs-3],5))+','+str(np.percentile(dataKFNN[:,0:max_obs-3]-dataRange[:,0:max_obs-3],95))+'\n')
f.write('NeuralNetworkKFfeedback,'+str(np.mean(dataKFNNZ[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.median(dataKFNNZ[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.std(dataKFNNZ[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.percentile(dataKFNNZ[:,0:max_obs-3]-dataRange[:,0:max_obs-3],5))+','+str(np.percentile(dataKFNNZ[:,0:max_obs-3]-dataRange[:,0:max_obs-3],95))+'\n')
f.write('Encoder,'+str(np.mean(encoderRange[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.median(encoderRange[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.std(encoderRange[:,0:max_obs-3]-dataRange[:,0:max_obs-3]))+','+str(np.percentile(encoderRange[:,0:max_obs-3]-dataRange[:,0:max_obs-3],5))+','+str(np.percentile(encoderRange[:,0:max_obs-3]-dataRange[:,0:max_obs-3],95))+'\n')
f.close()

randselect = np.random.randint(0,len(dataRange),10)

plt.figure()
plt.plot(dataRange[randselect,0:max_obs-3].flatten(),'k:')
plt.plot(gaussRange[randselect,0:max_obs-3].flatten(),'C0-',alpha=0.8)
plt.plot(dataRFR[randselect,0:max_obs-3].flatten(),'C1--',alpha=0.7)
plt.plot(dataKFNN[randselect,0:max_obs-3].flatten(),'C2-.',alpha=0.5)
plt.plot(dataKFNNZ[randselect,0:max_obs-3].flatten(),'C3-',alpha=0.6)
# plt.plot(encoderRange[randselect,0:max_obs-3].flatten(),'C4-',alpha=0.5)
plt.legend(['Truth','Gauss','RF','NN','NN-Z'])#,"Encdr"])
b,t = plt.ylim()
plt.ylim(np.max([b,0]),np.min([t,90000]))
plt.ylabel('Range prediction (km)')
plt.ylabel('Range prediction comparison')
plt.savefig('plots/RangePredictionComparisonPartial_'+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'_'+label+'.png')

bin = np.linspace(-30000,30000,600)
plt.figure()
h1 = plt.hist(gaussRange[:,0:max_obs-3].flatten()-dataRange[:,0:max_obs-3].flatten(),bins=bin,alpha=.7)#,'r-')
h2 = plt.hist(dataRFR[:,0:max_obs-3].flatten()-dataRange[:,0:max_obs-3].flatten(),bins=bin,alpha=.7)#,'g-.',alpha=0.6)
h3 = plt.hist(dataKFNN[:,0:max_obs-3].flatten()-dataRange[:,0:max_obs-3].flatten(),bins=bin,alpha=.7)#,'b--',alpha=0.8)
h4 = plt.hist(dataKFNNZ[:,0:max_obs-3].flatten()-dataRange[:,0:max_obs-3].flatten(),bins=bin,alpha=.6)#,'g-.',alpha=0.6)
h5 = plt.hist(encoderRange[:,0:max_obs-3].flatten()-dataRange[:,0:max_obs-3].flatten(),bins=bin,alpha=.6)#,'g-.',alpha=0.6)
plt.legend(['Gauss','RF','NN','NN-Z','Encdr'])
plt.xlabel('Range error (km)')
plt.title('Range prediction error')
plt.savefig('plots/RangeErrorComparison_'+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'_'+label+'.png')


bin = np.linspace(-1,1,100)
plt.figure()
plt.hist(GaussDots[0].flatten(),bin,alpha=0.5)
plt.hist(KFNNdots[0].flatten(),bin,alpha=0.5)
plt.hist(rangeforestDots[0].flatten(),bin,alpha=0.5)
plt.hist(orbitforestDots[0].flatten(),bin,alpha=0.5)
plt.hist(NNOEdots[0].flatten(),bin,alpha=0.5)
plt.hist(KFNOdots[0].flatten(),bin,alpha=0.5)
plt.legend(['Gauss','NN-KF','RF-KF','RF','NN','NN-O-KF'])
plt.title('SV relative error histogram')
plt.savefig('plots/SVrelativeErrorHist'+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'_'+label+'.png')

bin = np.linspace(-10,10,100)
plt.figure()
plt.hist(GaussDots[1].flatten(),bin,alpha=0.5)
plt.hist(KFNNdots[1].flatten(),bin,alpha=0.5)
plt.hist(rangeforestDots[1].flatten(),bin,alpha=0.5)
plt.hist(orbitforestDots[1].flatten(),bin,alpha=0.5)
plt.hist(NNOEdots[1].flatten(),bin,alpha=0.5)
plt.hist(KFNOdots[1].flatten(),bin,alpha=0.5)
plt.legend(['Gauss','NN-KF','RF-KF','RF','NN','NN-O-KF'])
plt.title('Orbit relative error histogram')
plt.savefig('plots/OrbitRelativeErrorHist'+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'_'+label+'.png')

# Orbit error changes over time
saves = ['Mean Motion','Eccentricity','Inclination','Right Ascension','Argument of Perigee','Mean Anomaly']
saves1 = ['MM','E','I','RAAN','AP','MA']
x0 = np.linspace(0,len(np.mean(GaussData[3,:,0],1)),len(np.mean(GaussData[3,:,0],1)))
x1 = np.linspace(0,len(np.mean(KFNNdata[3,:,0],1)),len(np.mean(KFNNdata[3,:,0],1)))
x2 = np.linspace(0,len(np.mean(rangeforestData[3,:,0],1)),len(np.mean(rangeforestData[3,:,0],1)))
for num in range(6):
    correction = 1
    if num >= 2:
        correction = 180/np.pi
    fig = plt.figure()
    if usemean:
        plt.plot(x0,np.mean(GaussData[3,:,num],1)*correction,'C0')
        plt.plot(x2,np.mean(RFdirectData[3,:,num],1)*correction,'C1')
        plt.plot(x2,np.mean(rangeforestData[3,:,num],1)*correction,'C2')
        plt.plot(x2,np.mean(orbitforestData[3,:,num],1)*correction,'C3')
        plt.plot(0*x1,'k:',alpha=0.5)
        plt.fill_between(x0,np.mean(GaussData[3,:,num],1)*correction-np.std(GaussData[3,:,num],1)*correction,np.mean(GaussData[3,:,num],1)*correction+np.std(GaussData[3,:,num],1)*correction,color='C0',alpha=0.2)
        plt.fill_between(x2,np.mean(RFdirectData[3,:,num],1)*correction-np.std(RFdirectData[3,:,num],1)*correction,np.mean(RFdirectData[3,:,num],1)*correction+np.std(RFdirectData[3,:,num],1)*correction,color='C1',alpha=0.2)
        plt.fill_between(x2,np.mean(rangeforestData[3,:,num],1)*correction-np.std(rangeforestData[3,:,num],1)*correction,np.mean(rangeforestData[3,:,num],1)*correction+np.std(rangeforestData[3,:,num],1)*correction,color='C2',alpha=0.2)
        plt.fill_between(x2,np.mean(orbitforestData[3,:,num],1)*correction-np.std(orbitforestData[3,:,num],1)*correction,np.mean(orbitforestData[3,:,num],1)*correction+np.std(orbitforestData[3,:,num],1)*correction,color='C3',alpha=0.2)
    else:
        plt.plot(x0,np.median(GaussData[3,:,num],1)*correction,'C0')
        plt.plot(x2,np.median(RFdirectData[3,:,num],1)*correction,'C1')
        plt.plot(x2,np.median(rangeforestData[3,:,num],1)*correction,'C2')
        plt.plot(x2,np.median(orbitforestData[3,:,num],1)*correction,'C3')
        plt.plot(0*x1,'k:',alpha=0.5)
        plt.fill_between(x0,np.percentile(GaussData[3,:,num],5,axis=1)*correction,np.percentile(GaussData[3,:,num],95,axis=1)*correction,color='C0',alpha=0.2)
        plt.fill_between(x2,np.percentile(RFdirectData[3,:,num],5,1)*correction,np.percentile(RFdirectData[3,:,num],95,1)*correction,color='C1',alpha=0.2)
        plt.fill_between(x2,np.percentile(rangeforestData[3,:,num],5,1)*correction,np.percentile(rangeforestData[3,:,num],95,1)*correction,color='C2',alpha=0.2)
        plt.fill_between(x2,np.percentile(orbitforestData[3,:,num],5,1)*correction,np.percentile(orbitforestData[3,:,num],95,1)*correction,color='C3',alpha=0.2)
    plt.legend(['Gauss','RF-range','RF-KF','RF-orbit'])
    if num == 0:
        plt.ylabel('Error (1/day)')
        b,t = plt.ylim()
        plt.ylim(-np.min([15,abs(b)]),np.min([15,abs(t)]))
    elif num == 1:
        plt.ylabel("Error")
        b,t = plt.ylim()
        plt.ylim(-np.min([3,abs(b)]),np.min([3,abs(t)]))
    elif num > 1:
        plt.ylabel('Error (degrees)')
    plt.xlabel('Number of observations')
    plt.title('Orbit '+saves[num]+' error over time')
    plt.savefig('plots/gauss_vs_rf_'+saves1[num]+'_overtime_'+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'_'+label+'.png')
plt.close()

for num in range(6):
    correction = 1
    if num >= 2:
        correction = 180/np.pi
    fig = plt.figure()
    if usemean:
        plt.plot(x0,np.mean(GaussData[3,:,num],1)*correction,'C0')
        plt.plot(x1,np.mean(KFNNdata[3,:,num],1)*correction,'C1')
        plt.plot(x2,np.mean(rangeforestData[3,:,num],1)*correction,'C2')
        plt.plot(x2,np.mean(orbitforestData[3,:,num],1)*correction,'C3')
        # plt.plot(x2,np.mean(NNOEdata[3,:,num],1)*correction,'C4')
        plt.plot(x2,np.mean(orbitNNdata[3,:,num],1)*correction,'C4')
        plt.plot(x1,np.mean(rangeNNdata[3,:,num],1)*correction,'C5')
        plt.plot(0*x1,'k:',alpha=0.5)
        plt.fill_between(x0,np.mean(GaussData[3,:,num],1)*correction-np.std(GaussData[3,:,num],1)*correction,np.mean(GaussData[3,:,num],1)*correction+np.std(GaussData[3,:,num],1)*correction,color='C0',alpha=0.2)
        plt.fill_between(x1,np.mean(KFNNdata[3,:,num],1)*correction-np.std(KFNNdata[3,:,num],1)*correction,np.mean(KFNNdata[3,:,num],1)*correction+np.std(KFNNdata[3,:,num],1)*correction,color='C1',alpha=0.2)
        plt.fill_between(x2,np.mean(rangeforestData[3,:,num],1)*correction-np.std(rangeforestData[3,:,num],1)*correction,np.mean(rangeforestData[3,:,num],1)*correction+np.std(rangeforestData[3,:,num],1)*correction,color='C2',alpha=0.2)
        plt.fill_between(x2,np.mean(orbitforestData[3,:,num],1)*correction-np.std(orbitforestData[3,:,num],1)*correction,np.mean(orbitforestData[3,:,num],1)*correction+np.std(orbitforestData[3,:,num],1)*correction,color='C3',alpha=0.2)
        # plt.fill_between(x2,np.mean(NNOEdata[3,:,num],1)-np.std(NNOEdata[3,:,num],1),np.mean(NNOEdata[3,:,num],1)*correction+np.std(NNOEdata[3,:,num],1)*correction,color='C4',alpha=0.2)
        plt.fill_between(x2,np.mean(orbitNNdata[3,:,num],1)*correction-np.std(orbitNNdata[3,:,num],1)*correction,np.mean(orbitNNdata[3,:,num],1)*correction+np.std(orbitNNdata[3,:,num],1)*correction,color='C4',alpha=0.2)
        plt.fill_between(x1,np.mean(rangeNNdata[3,:,num],1)*correction-np.std(rangeNNdata[3,:,num],1)*correction,np.mean(rangeNNdata[3,:,num],1)*correction+np.std(rangeNNdata[3,:,num],1)*correction,color='C5',alpha=0.2)
    else:
        plt.plot(x0,np.median(GaussData[3,:,num],1)*correction,'C0')
        plt.plot(x1,np.median(KFNNdata[3,:,num],1)*correction,'C1')
        plt.plot(x2,np.median(rangeforestData[3,:,num],1)*correction,'C2')
        plt.plot(x2,np.median(orbitforestData[3,:,num],1)*correction,'C3')
        # plt.plot(x2,np.median(NNOEdata[3,:,num],1)*correction,'C4')
        plt.plot(x2,np.median(orbitNNdata[3,:,num],1)*correction,'C4')
        plt.plot(x1,np.median(rangeNNdata[3,:,num],1)*correction,'C5')
        plt.plot(0*x1,'k:',alpha=0.5)
        plt.fill_between(x0,np.percentile(GaussData[3,:,num],5,axis=1)*correction,np.percentile(GaussData[3,:,num],95,axis=1)*correction,color='C0',alpha=0.2)
        plt.fill_between(x1,np.percentile(KFNNdata[3,:,num],5,1)*correction,np.percentile(KFNNdata[3,:,num],95,1)*correction,color='C1',alpha=0.2)
        plt.fill_between(x2,np.percentile(rangeforestData[3,:,num],5,1)*correction,np.percentile(rangeforestData[3,:,num],95,1)*correction,color='C2',alpha=0.2)
        plt.fill_between(x2,np.percentile(orbitforestData[3,:,num],5,1)*correction,np.percentile(orbitforestData[3,:,num],95,1)*correction,color='C3',alpha=0.2)
        # plt.fill_between(x2,np.percentile(NNOEdata[3,:,num],5,1),np.percentile(NNOEdata[3,:,num],95,1)*correction,color='C4',alpha=0.2)
        plt.fill_between(x2,np.percentile(orbitNNdata[3,:,num],5,1)*correction,np.percentile(orbitNNdata[3,:,num],95,1)*correction,color='C4',alpha=0.2)
        plt.fill_between(x1,np.percentile(rangeNNdata[3,:,num],5,1)*correction,np.percentile(rangeNNdata[3,:,num],95,1)*correction,color='C5',alpha=0.2)
    plt.legend(['Gauss','NN-KF','RF-KF','RF','NN','NN-range'])
    if num == 0:
        plt.ylabel('Error (1/day)')
        b,t = plt.ylim()
        plt.ylim(-np.min([15,abs(b)]),np.min([15,abs(t)]))
    elif num == 1:
        plt.ylabel("Error")
        b,t = plt.ylim()
        plt.ylim(-np.min([3,abs(b)]),np.min([3,abs(t)]))
    elif num > 1:
        plt.ylabel('Error (degrees)')
    plt.xlabel('Number of observations')
    plt.title('Orbit '+saves[num]+' error over time')
    plt.savefig('plots/elset'+saves1[num]+'overtime'+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'_'+label+'.png')
plt.close()

for num in range(6):
    correction = 1
    if num >= 2:
        correction = 180/np.pi
    plt.figure()
    if usemean:
        plt.subplot(3,2,1)
        plt.plot(np.mean(NNOEdata[3,:,num],1)*correction,'C1')
        plt.plot(np.mean(orbitNNdata[3,:,num],1)*correction,'C3')
        plt.plot(0*x1,'k:',alpha=0.5)
        plt.fill_between(x2,np.mean(NNOEdata[3,:,num],1)*correction-np.std(NNOEdata[3,:,num],1)*correction,np.mean(NNOEdata[3,:,num],1)*correction+np.std(NNOEdata[3,:,num],1)*correction,color='C1',alpha=0.2)
        plt.fill_between(x2,np.mean(orbitNNdata[3,:,num],1)*correction-np.std(orbitNNdata[3,:,num],1)*correction,np.mean(orbitNNdata[3,:,num],1)*correction+np.std(orbitNNdata[3,:,num],1)*correction,color='C4',alpha=0.2)
    else:
        plt.plot(np.median(NNOEdata[3,:,num],1)*correction,'C1')
        plt.plot(np.median(orbitNNdata[3,:,num],1)*correction,'C3')
        plt.plot(0*x1,'k:',alpha=0.5)
        plt.fill_between(x2,np.percentile(NNOEdata[3,:,num],5,1)*correction,np.percentile(NNOEdata[3,:,num],95,1)*correction,color='C1',alpha=0.2)
        plt.fill_between(x2,np.percentile(orbitNNdata[3,:,num],5,1)*correction,np.percentile(orbitNNdata[3,:,num],95,1)*correction,color='C4',alpha=0.2)
    plt.ylabel(saves[num])
    plt.title('NN-KF Input size comparison')
    plt.legend(['KF trained','traditional'])
    if num == 0:
        plt.ylabel('Error (1/day)')
        b,t = plt.ylim()
        plt.ylim(-np.min([100,abs(b)]),np.min([100,abs(t)]))
    elif num == 1:
        plt.ylabel("Error")
        b,t = plt.ylim()
        plt.ylim(-np.min([100,abs(b)]),np.min([100,abs(t)]))
    elif num > 1:
        plt.ylabel('Error (degrees)')
    plt.savefig('plots/kfnnelset'+str(datalength)+"_NNOE_"+saves1[num]+"_"+str(dataseting)+'_'+str(test_size)+'_'+label+'.png')
plt.close()

x3 = np.linspace(0,len(np.mean(KFNN3data[3,:,0],1)),len(np.mean(KFNN3data[3,:,0],1)))
for num in range(6):
    correction = 1
    if num >= 2:
        correction = 180/np.pi
    plt.figure()
    if usemean:
        plt.subplot(3,2,num)
        plt.plot(np.mean(KFNNdata[3,:,num]*correction,1),'C1')
        plt.plot(np.mean(KFNN3data[3,:,num]*correction,1),'C3')
        plt.plot(np.mean(KFNN2data[3,:,num]*correction,1),'C4')
        plt.plot(0*x1,'k:',alpha=0.5)
        plt.fill_between(x1,np.mean(KFNNdata[3,:,num],1)*correction-np.std(KFNNdata[3,:,num],1)*correction,np.mean(KFNNdata[3,:,num],1)*correction+np.std(KFNNdata[3,:,num],1)*correction,color='C1',alpha=0.2)
        plt.fill_between(x3,np.mean(KFNN3data[3,:,num],1)*correction-np.std(KFNN3data[3,:,num],1)*correction,np.mean(KFNN3data[3,:,num],1)*correction+np.std(KFNN3data[3,:,num],1)*correction,color='C3',alpha=0.2)
        plt.fill_between(x2,np.mean(KFNN2data[3,:,num],1)*correction-np.std(KFNN2data[3,:,num],1)*correction,np.mean(KFNN2data[3,:,num],1)*correction+np.std(KFNN2data[3,:,num],1)*correction,color='C4',alpha=0.2)
    else:
        plt.plot(np.median(KFNNdata[3,:,num]*correction,1),'C1')
        plt.plot(np.median(KFNN3data[3,:,num]*correction,1),'C3')
        plt.plot(np.median(KFNN2data[3,:,num]*correction,1),'C4')
        plt.plot(0*x1,'k:',alpha=0.5)
        plt.fill_between(x1,np.percentile(KFNNdata[3,:,num],5,1)*correction,np.percentile(KFNNdata[3,:,num],95,1)*correction,color='C1',alpha=0.2)
        plt.fill_between(x3,np.percentile(KFNN3data[3,:,num],5,1)*correction,np.percentile(KFNN3data[3,:,num],95,1)*correction,color='C3',alpha=0.2)
        plt.fill_between(x2,np.percentile(KFNN2data[3,:,num],5,1)*correction,np.percentile(KFNN2data[3,:,num],95,1)*correction,color='C4',alpha=0.2)
    plt.ylabel(saves[num])
    plt.title('NN-KF Input size comparison')
    if num == 0:
        plt.ylabel('Error (1/day)')
        b,t = plt.ylim()
        plt.ylim(-np.min([900,abs(b)]),np.min([900,abs(t)]))
    elif num == 1:
        plt.ylabel("Error")
        b,t = plt.ylim()
        plt.ylim(-np.min([2000,abs(b)]),np.min([2000,abs(t)]))
    elif num > 1:
        plt.ylabel('Error (degrees)')
    plt.legend(['inputsize '+str(datalength),'inputsize '+str(datalength2),'inputsize '+str(datalength1)])
    plt.savefig('plots/kfnnelset'+str(datalength)+"vs"+str(datalength1)+saves1[num]+"Comparison"+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'_'+label+'.png')
plt.close()

for num in range(6):
    correction = 1
    if num >= 2:
        correction = 180/np.pi
    plt.figure()
    if usemean:
        plt.plot(np.mean(KFNNdata[3,:,num],1)*correction,'C1')
        plt.plot(np.mean(KFNN0data[3,:,num],1)*correction,'C2')
        plt.plot(0*x1,'k:',alpha=0.5)
        plt.fill_between(x1,np.mean(KFNNdata[3,:,num],1)*correction-np.std(KFNNdata[3,:,num],1)*correction,np.mean(KFNNdata[3,:,num],1)*correction+np.std(KFNNdata[3,:,num],1)*correction,color='C1',alpha=0.2)
        plt.fill_between(x1,np.mean(KFNN0data[3,:,num],1)*correction-np.std(KFNN0data[3,:,num],1)*correction,np.mean(KFNN0data[3,:,num],1)*correction+np.std(KFNN0data[3,:,num],1)*correction,color='C2',alpha=0.2)
    else:
        plt.plot(np.median(KFNNdata[3,:,num],1)*correction,'C1')
        plt.plot(np.median(KFNN0data[3,:,num],1)*correction,'C2')
        plt.plot(0*x1,'k:',alpha=0.5)
        plt.fill_between(x1,np.percentile(KFNNdata[3,:,num],5,1)*correction,np.percentile(KFNNdata[3,:,num],95,1)*correction,color='C1',alpha=0.2)
        plt.fill_between(x1,np.percentile(KFNN0data[3,:,num],5,1)*correction,np.percentile(KFNN0data[3,:,num],95,1)*correction,color='C2',alpha=0.2)
    plt.ylabel(saves[num])
    # plt.title('NN-KF')
    plt.legend(['KF-Z','KF'])
    if num == 0:
        plt.ylabel('Error (1/day)')
        b,t = plt.ylim()
        plt.ylim(-np.min([500,abs(b)]),np.min([500,abs(t)]))
    elif num == 1:
        plt.ylabel("Error")
        b,t = plt.ylim()
        plt.ylim(-np.min([1000,abs(b)]),np.min([1000,abs(t)]))
    elif num > 1:
        plt.ylabel('Error (degrees)')
    plt.title('Orbit '+saves[num]+' error over time')
    plt.savefig('plots/kfnnComp'+str(datalength)+"vs"+str(datalength1)+saves1[num]+"Comparison"+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'_'+label+'.png')
plt.close()

for num in range(6):
    correction = 1
    if num >= 2:
        correction = 180/np.pi
    plt.figure()
    if usemean:
        plt.plot(np.mean(NOKFdata[3,:,num],1)*correction,'C1')
        plt.plot(np.mean(NNOEdata[3,:,num],1)*correction,'C2')
        plt.plot(np.mean(KFNNdata[3,:,num],1)*correction,'C3')
        plt.plot(np.mean(KFNOdata[3,:,num],1)*correction,'C4')
        plt.plot(0*x1,'k:',alpha=0.5)
        plt.fill_between(x1,np.mean(NOKFdata[3,:,num],1)*correction-np.std(NOKFdata[3,:,num],1)*correction,np.mean(NOKFdata[3,:,num],1)*correction+np.std(NOKFdata[3,:,num],1)*correction,color='C1',alpha=0.2)
        plt.fill_between(x2,np.mean(NNOEdata[3,:,num],1)*correction-np.std(NNOEdata[3,:,num],1)*correction,np.mean(NNOEdata[3,:,num],1)*correction+np.std(NNOEdata[3,:,num],1)*correction,color='C2',alpha=0.2)
        plt.fill_between(x1,np.mean(KFNNdata[3,:,num],1)*correction-np.std(KFNNdata[3,:,num],1)*correction,np.mean(KFNNdata[3,:,num],1)*correction+np.std(KFNNdata[3,:,num],1)*correction,color='C3',alpha=0.2)
        plt.fill_between(x1,np.mean(KFNOdata[3,:,num],1)*correction-np.std(KFNOdata[3,:,num],1)*correction,np.mean(KFNOdata[3,:,num],1)*correction+np.std(KFNOdata[3,:,num],1)*correction,color='C4',alpha=0.2)
    else:
        plt.plot(np.median(NOKFdata[3,:,num],1)*correction,'C1')
        plt.plot(np.median(NNOEdata[3,:,num],1)*correction,'C2')
        plt.plot(np.median(KFNNdata[3,:,num],1)*correction,'C3')
        plt.plot(np.median(KFNOdata[3,:,num],1)*correction,'C4')
        plt.plot(0*x1,'k:',alpha=0.5)
        plt.fill_between(x1,np.percentile(NOKFdata[3,:,num],5,1)*correction,np.percentile(NOKFdata[3,:,num],95,1)*correction,color='C1',alpha=0.2)
        plt.fill_between(x2,np.percentile(NNOEdata[3,:,num],5,1)*correction,np.percentile(NNOEdata[3,:,num],95,1)*correction,color='C2',alpha=0.2)
        plt.fill_between(x1,np.percentile(KFNNdata[3,:,num],5,1)*correction,np.percentile(KFNNdata[3,:,num],95,1)*correction,color='C3',alpha=0.2)
        plt.fill_between(x1,np.percentile(KFNOdata[3,:,num],5,1)*correction,np.percentile(KFNOdata[3,:,num],95,1)*correction,color='C4',alpha=0.2)
    plt.ylabel(saves[num])
    if num == 0:
        plt.ylabel('Error (1/day)')
        b,t = plt.ylim()
        plt.ylim(-np.min([50,abs(b)]),np.min([50,abs(t)]))
    elif num == 1:
        plt.ylabel("Error")
        b,t = plt.ylim()
        plt.ylim(-np.min([10,abs(b)]),np.min([10,abs(t)]))
    elif num > 1:
        plt.ylabel('Error (degrees)')
    # plt.title('NN-KF')
    plt.legend(['NN-OE-KF','NN-OE','NN-R-KF','NN-O-KF'])
    plt.title('Orbit '+saves[num]+' error over time')
    plt.savefig('plots/NNOEcomparison_'+saves1[num]+'_'+str(datalength)+'_'+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'_'+label+'.png')
plt.close()


TF.logData(GaussData,'Gauss',firstline=1)
TF.logData(rangeforestData,'forest-range')
TF.logData(orbitforestData,'forest-orbit')
TF.logData(RFdirectData,'forest-gauss')
TF.logData(orbitNNdata,'NN-orbit')
# TF.logData(rangeNNdata,'KFNN')
TF.logData(KFNNdata,'NN-R-KF')
TF.logData(NOKFdata,'NN-OE-KF')
TF.logData(NNOEdata,'NN-orbit')
TF.logData(KFNOdata,'NN-SV-KF')

f = open('plots/orbit_data_stats.csv','w+')
f.write('Element,min,median,max\n')
f.write('Mean motion,'+str(np.min(realdata[2,:,0]))+','+str(np.median(realdata[2,:,0]))+','+str(np.max(realdata[2,:,0]))+'\n')
f.write('Eccentricity,'+str(np.min(realdata[2,:,1]))+','+str(np.median(realdata[2,:,1]))+','+str(np.max(realdata[2,:,1]))+'\n')
f.write('Inclination,'+str(np.min(realdata[2,:,2])*180/np.pi)+','+str(np.median(realdata[2,:,2])*180/np.pi)+','+str(np.max(realdata[2,:,2])*180/np.pi)+'\n')
f.write('Right Ascension,'+str(np.min(realdata[2,:,3])*180/np.pi)+','+str(np.median(realdata[2,:,3])*180/np.pi)+','+str(np.max(realdata[2,:,3])*180/np.pi)+'\n')
f.write('Argument of Perigee,'+str(np.min(realdata[2,:,4])*180/np.pi)+','+str(np.median(realdata[2,:,4])*180/np.pi)+','+str(np.max(realdata[2,:,4])*180/np.pi)+'\n')
f.write('Mean Anomaly,'+str(np.min(realdata[2,:,5])*180/np.pi)+','+str(np.median(realdata[2,:,5])*180/np.pi)+','+str(np.max(realdata[2,:,5])*180/np.pi)+'\n')
# realaltitude = np.sqrt(realdata[0,:,0,:]**2 + realdata[0,:,1,:]**2 + realdata[0,:,2,:]**2).flatten()
# realspeed = np.sqrt(realdata[0,:,3,:]**2 + realdata[0,:,4,:]**2 + realdata[0,:,5,:]**2).flatten()
# f.write('Orbit altitude,'+str(np.min(realaltitude))+','+str(np.max(realaltitude))+'\n')
# f.write('Orbit speed,'+str(np.min(realspeed))+','+str(np.max(realspeed))+'\n')
f.close()

# f = open('plots/orbit_results.csv','w+')
# f.write('Method,MM-median,MM-5th,MM-95th,I-median,I-5th,I-95th,')


if dohist:
    # Error histograms
    for num in range(6):
        bin = np.linspace(-5,5,201)
        correction = 1
        if num == 1:
            bin = np.linspace(-1,2,51)
        else:
            bin = np.linspace(-np.pi,np.pi,91)/d2r
            correction = 180/np.pi
        plt.figure()
        # plt.hist([np.transpose(gausErr)[0],np.transpose(OEKFAO)[0],np.transpose(OENNKF)[0],np.transpose(OERFKF)[0]],bins=bin,alpha=0.9,histtype='barstacked')
        hG = plt.hist(np.mean(GaussData[3,:,num,:],0)*correction,bins=bin,alpha=0.5)#,color='r')
        hKFNN = plt.hist(np.mean(KFNNdata[3,:,num,:],0)*correction,bins=bin,alpha=0.5)#,color='k')
        hRFKF = plt.hist(np.mean(rangeforestData[3,:,num,:],0)*correction,bins=bin,alpha=0.5)#,color='b')
        hRFOE = plt.hist(np.mean(orbitforestData[3,:,num,:],0)*correction,bins=bin,alpha=0.5)#,color='r')
        hOENN = plt.hist(np.mean(NNOEdata[3,:,num,:],0)*correction,bins=bin,alpha=0.5)#,color='g')
        # hNN = plt.hist(np.transpose(OENNerr)[0],bins=bin,alpha=0.5)#,color='b')
        # hNN1 = plt.hist(np.transpose(modelOoe)[0],bins=bin,alpha=0.5)#,color='b')
        diffadd = np.mean(hG[1][1:] - hG[1][0:-1])/2
        plt.plot(hG[1][0:-1]+diffadd,hG[0],'sk')
        # plt.plot(hKF[1][0:-1]+diffadd,hKF[0],'oC0')
        plt.plot(hKFNN[1][0:-1]+diffadd,hKFNN[0],'vC1')
        plt.plot(hRFKF[1][0:-1]+diffadd,hRFKF[0],'dC2')
        plt.plot(hRFOE[1][0:-1]+diffadd,hRFOE[0],'*C3')
        plt.plot(hOENN[1][0:-1]+diffadd,hOENN[0],'.C4')
        # plt.plot(hNN1[1][0:-1]+diffadd,hNN1[0],'.C6')
        plt.legend(['Gauss','NN-KF','RF-KF','RF','NN'])
        if num == 0:
            plt.title('Mean Motion Error')
            plt.xlabel('1/days')
            plt.savefig('plots/orbitMeanMotionaccuracy'+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'.png')
        elif num == 1:
            plt.legend(['Gauss','NN-KF','RF-KF','RF','NN'])
            plt.title('Eccentricity Error')
            plt.savefig('plots/orbitEccentricityaccuracy'+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'.png')
        elif num == 2:
            plt.legend(['Gauss','NN-KF','RF-KF','RF','NN'])
            plt.title('Eccentricity Error')
            plt.savefig('plots/orbitEccentricityaccuracy'+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'.png')
        elif num == 3:
            plt.title('Right Ascension Error')
            plt.xlabel('degrees')
            plt.savefig('plots/orbitRAANaccuracy'+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'.png')
        elif num == 4:
            plt.title('Argument of Perigee Error')
            plt.xlabel('degrees')
            plt.savefig('plots/orbitArgPerigeeaccuracy'+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'.png')
        elif num == 5:
            plt.title('Mean Anomaly Error')
            plt.xlabel('degrees')
            plt.savefig('plots/orbitMeanAnomalyaccuracy'+str(dataseting)+'_'+str(dataseting1)+'_'+str(test_size)+'.png')
    plt.close()
