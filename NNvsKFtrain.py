import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
from sklearn import ensemble
# from sklearn import pipeline
# from sklearn import impute
# from sklearn import compose
from sklearn import neural_network
from sklearn import tree
import joblib

import torch.nn as nn
import torch.optim as optim

import neural_filter
import torchNN

import createDataLive
import OrbitDetermination as OD

import matplotlib as mpl
from matplotlib import pyplot as plt



dataseting = 6
datalegnths = {0:5,1:10,2:9,3:18,4:10,5:20,6:30}
datalength = datalegnths[dataseting]
print("Running with input size",datalength)
nn_append = '5second' #'-200'

maxIters = 1000

# training parameters
n_epochs = 10   # number of epochs to run
batch_size = 300  # max size of each batch
num_tles = 25000
obs_noise = 0.004 # .017 = 1 arcminute of noise, .004 = 14.4 arcseconds, .0003 = 1.1 arcseconds
batch_start = torch.arange(0, maxIters, batch_size)
test_size = 300
eval_size = 500
max_obs = 15
# min_obs = 5
tmin = 1
tmax = 10

dt = 10

rangenorm = 72000
d2r = np.pi/180

orbdet = OD.orbitDetermination()
uccs = neural_filter.locations.Location('UCCS',38.89588,-104.80232,1950)

# modelR = torchNN.Trnsfrmr1(datalength,1,(datalength,datalength))
# modelR = torchNN.RNN1(datalength,1)
modelR = torchNN.Dense91(datalength,1,(100,100))
modelO = torchNN.Dense91(datalength,6,(100,100)) # torchNN.Trnsfrmr1(datalength,6,(datalength,datalength))
modelSV = torchNN.Dense91(datalength,6,(100,100)) # torchNN.Trnsfrmr1(datalength,6,(datalength,datalength))

try:
    modelR.load_state_dict(torch.load('models/range_nn_'+str(dataseting)+'bestweights'+nn_append,weights_only=True))
    print("Loaded best weights")
except:
    print("Could not load range previous weights, training from scratch.")

# model2.load_state_dict(torch.load('models/NN_best_weights',weights_only=True))
# model1.load_state_dict(torch.load('models/RNN_best_weights',weights_only=True))
# model0.load_state_dict(torch.load('models/transformer_best_weights_all',weights_only=True))
# modelO.load_state_dict(torch.load('models/Orbit_Predict_best_weights',weights_only=True))

TLE = createDataLive.TLEdata()
Dmanip = torchNN.MLdataManipulation()

loss_fn = nn.MSELoss()  # mean square error
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
optimizerO = optim.Adam(modelO.parameters(), lr=0.001)
optimizerR = optim.Adam(modelR.parameters(), lr=0.001)
optimizerSV = optim.Adam(modelR.parameters(), lr=0.001)

modelO.train()
modelR.train()
modelSV.train()
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

estimators = 1000 # try 1000?
rft = ensemble.RandomForestRegressor(n_estimators = estimators, warm_start=True, max_depth=5) # can only train once
etr = ensemble.ExtraTreesRegressor(n_estimators = estimators,warm_start=True,max_depth=5) # can only train once
abr = ensemble.AdaBoostRegressor(n_estimators=1000,learning_rate=0.5) # joblib.load('adaboost_save_'+str(datalength)) # ensemble.AdaBoostRegressor(n_estimators=estimators,learning_rate=0.9) # can train repeatedly
bgr = ensemble.BaggingRegressor(n_estimators=1000,warm_start=True,estimator=tree.DecisionTreeRegressor(max_depth=5)) # joblib.load('bagging_save_'+str(datalength)) # ensemble.BaggingRegressor(n_estimators=estimators) # can train repeatedly
gbr = ensemble.GradientBoostingRegressor(n_estimators=1000,learning_rate=0.2,max_depth=4) # joblib.load('gradientboost_save_'+str(datalength)) # ensemble.GradientBoostingRegressor(n_estimators=estimators) # can train repeatedly
hbr = ensemble.HistGradientBoostingRegressor(learning_rate=0.1,max_iter=300,max_depth=5) # joblib.load('histgradboost_save_'+str(datalength)) # ensemble.HistGradientBoostingRegressor() # can train repeatedly
nn0 = neural_network.MLPRegressor(hidden_layer_sizes=(300),activation='relu',max_iter=int(n_epochs*batch_size))
# nn0 = joblib.load('sknn_save_'+str(datalength))
gbrOMM = ensemble.GradientBoostingRegressor(n_estimators=1000,learning_rate=0.3)
gbrOEc = ensemble.GradientBoostingRegressor(n_estimators=1000,learning_rate=0.3)
gbrOIn = ensemble.GradientBoostingRegressor(n_estimators=1000,learning_rate=0.3)
gbrORA = ensemble.GradientBoostingRegressor(n_estimators=1000,learning_rate=0.3)
gbrOAP = ensemble.GradientBoostingRegressor(n_estimators=1000,learning_rate=0.3)
gbrOMA = ensemble.GradientBoostingRegressor(n_estimators=1000,learning_rate=0.3)

# abr = joblib.load('adaboost_save_'+str(datalength))
# gbr = joblib.load('gradientboost_save_'+str(datalength))
# hbr = joblib.load('histgradboost_save_'+str(datalength))

# Hold the best model
best_mseR = np.inf   # init to infinity
best_weightsR = None
best_mseO = np.inf   # init to infinity
best_weightsO = None

historyO = []
historyR = []

dataSaveX = []
dataSaveY = []
orbitSave = []

tle = TLE.createRandomOrbit(num_tles)

for epoch in range(n_epochs):
    modelO.train()
    modelR.train()
    saveX = []
    saveY = []
    saveO = []
    for batch in range(batch_size):
        dt = np.random.randint(tmin,tmax+1)
        obsrvr, reality, orbit = TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs)
        observer = np.transpose(obsrvr)
        realtle = reality[1]
        reals = np.transpose(reality[0])
        siderealtime = TLE.OD.siderealTime(observer[0],TLE.location)
        # consideration - try setting data so input is cosines and sines of angles instead of direct angular measurements?
        Dtrain,Rtrain = Dmanip.organizeDataInput(observer,siderealtime,dataseting,rangenorm)
        orbitmod = [orbit[0],orbit[1],orbit[2]*d2r,orbit[3]*d2r,orbit[4]*d2r,orbit[5]*d2r]
        X_batch = []
        y_batch = []
        for ii in range(len(Dtrain)):
            X_batch.append(torch.tensor(Dtrain[ii].astype(float),dtype=torch.float32))
            y_batch.append(torch.tensor(Rtrain[ii].astype(float),dtype=torch.float32).reshape(1))
            saveX.append(Dtrain[ii])
            saveY.append(Rtrain[ii])
            saveO.append(orbitmod)
            if ii < 5:
                dataSaveX.append(Dtrain[ii])
                dataSaveY.append(Rtrain[ii])
            orbitSave.append(orbitmod)
        X_batch = np.array(X_batch)
        y_batch = torch.tensor(y_batch).reshape(len(y_batch),1)
        # Train range model
        y_pred = modelR(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # optimizer1.zero_grad() # should this be optimizer or model?
        # model.zero_grad()
        loss.backward()
        # update weights
        optimizerR.step()
        # Train orbit model
        y_predO = modelO(Dtrain[np.random.randint(0,len(Dtrain))])
        lossO = loss_fn(y_predO.flatten(),torch.tensor(orbitmod,dtype=torch.float32))
        lossO.backward()
        optimizerO.step()
        # Train state-vector model - needs to be updated to work
        # y_predSV = modelSV(X_batch)
        # loss = loss_fn(y_predSV, reality)
        # SKLearn version
        nn0.partial_fit(X_batch,y_batch.flatten())
    print("Fitting forests")
    abr.fit(saveX,saveY)
    # bgr.fit(saveX,saveY)
    gbr.fit(saveX,saveY)
    hbr.fit(saveX,saveY)
    saveO = np.transpose(saveO)
    gbrOMM.fit(saveX,saveO[0])
    gbrOEc.fit(saveX,saveO[1])
    gbrOIn.fit(saveX,saveO[2])
    gbrORA.fit(saveX,saveO[3])
    gbrOAP.fit(saveX,saveO[4])
    gbrOMA.fit(saveX,saveO[5])
    print('Testing models')
    optimizerR.zero_grad()
    optimizerO.zero_grad()
    modelR.zero_grad()
    modelO.zero_grad()
    X_batch = []
    y_batch = []
    orbitsave = []
    best_weights = copy.deepcopy(modelR.state_dict())
    for jj in range(test_size):
        dt = np.random.randint(5,61)
        obsrvr,reality,orbit = TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs)
        orbitmod = [orbit[0],orbit[1],orbit[2]*d2r,orbit[3]*d2r,orbit[4]*d2r,orbit[5]*d2r]
        observer = np.transpose(obsrvr)
        realtle = reality[1]
        reals = np.transpose(reality[0])
        siderealtime = TLE.OD.siderealTime(observer[0],TLE.location)
        Dtrain,Rtrain = Dmanip.organizeDataInput(observer,siderealtime,dataseting,rangenorm)
        for ii in range(len(Dtrain)):
            X_batch.append(torch.tensor(Dtrain[ii].astype(float),dtype=torch.float32))
            y_batch.append(torch.tensor(Rtrain[ii].astype(float),dtype=torch.float32).reshape(1))
            orbitsave.append(orbitmod)
    X_batch = np.array(X_batch)
    y_batch = torch.tensor(y_batch)
    modelR.eval()
    modelO.eval()
    y_pred = modelR(X_batch)
    y_predO = modelO(X_batch)
    print('Updating models')
    mse = loss_fn(y_pred.flatten(), y_batch)
    mse = float(mse)
    historyR.append(mse)
    mseO = loss_fn(y_predO, torch.tensor(orbitsave))
    mseO = float(mseO)
    historyO.append(mseO)
    if mse <= best_mseR:
        best_mseR = mse
        best_weightsR = copy.deepcopy(modelR.state_dict())
        print('OrbitPredict improved MSE',best_mseR)
    if mseO <= best_mseO:
        best_mseO = mseO
        best_weightsO = copy.deepcopy(modelO.state_dict())
        print('OrbitPredict improved MSE',best_mseO)
    resnn = nn0.predict(X_batch)
    scorsknn = nn0.score(X_batch,y_batch)
    rmsnn = np.sqrt(np.mean((resnn - np.array(y_batch).flatten())**2))
    print("Range NN MSE:",mse,'sklearn version:',rmsnn,scorsknn)
    # print("RFT MSE:",np.mean(RftErr))
    print("Finished epoch:",epoch+1,'of',n_epochs)

torch.save(best_weights,'models/range_nn_'+str(dataseting)+'bestweights'+nn_append)

dataSaveX = np.array(dataSaveX)
dataSaveY = np.array(dataSaveY)
print("Training ensemble models")
treeorbtrain = np.transpose(orbitSave)
# gbrOMM.fit(dataSaveX,treeorbtrain[0])
# gbrOEc.fit(dataSaveX,treeorbtrain[1])
# gbrOIn.fit(dataSaveX,treeorbtrain[2])
# gbrORA.fit(dataSaveX,treeorbtrain[3])
# gbrOAP.fit(dataSaveX,treeorbtrain[4])
# gbrOMA.fit(dataSaveX,treeorbtrain[5])
joblib.dump(gbrOMM,'models/gradboost_MM_predictor_'+str(dataseting)+nn_append)
joblib.dump(gbrOEc,'models/gradboost_Ecc_predictor_'+str(dataseting)+nn_append)
joblib.dump(gbrOIn,'models/gradboost_Inc_predictor_'+str(dataseting)+nn_append)
joblib.dump(gbrORA,'models/gradboost_RAAN_predictor_'+str(dataseting)+nn_append)
joblib.dump(gbrOAP,'models/gradboost_AP_predictor_'+str(dataseting)+nn_append)
joblib.dump(gbrOMA,'models/gradboost_MA_predictor_'+str(dataseting)+nn_append)

rft.fit(dataSaveX,dataSaveY)
etr.fit(dataSaveX,dataSaveY)
abr.fit(dataSaveX,dataSaveY) # only predicts one range?
bgr.fit(dataSaveX,dataSaveY)
gbr.fit(dataSaveX,dataSaveY)
hbr.fit(dataSaveX,dataSaveY)
# sgr.fit(dataSaveX,dataSaveY)
# vrr.fit(dataSaveX,dataSaveY)
nn0.fit(dataSaveX,dataSaveY) # train during or after?

torch.save(modelR,'models/range_nn_'+str(dataseting)+nn_append)
joblib.dump(rft,'models/randomforest_save_'+str(dataseting)+nn_append)
joblib.dump(etr,'models/extratrees_save_'+str(dataseting)+nn_append)
joblib.dump(abr,'models/adaboost_save_'+str(dataseting))
joblib.dump(bgr,'models/bagging_save_'+str(dataseting)+nn_append)
joblib.dump(gbr,'models/gradientboost_save_'+str(dataseting))
joblib.dump(hbr,'models/histgradboost_save_'+str(dataseting))
joblib.dump(nn0,'models/sknn_save_'+str(dataseting)+nn_append)

print("Evaluating")
# prrslts = []
modelR.eval()
modelO.eval()
errNN = []
errOrb = []
rsltsRFT = []
rsltsETR = []
rsltsABR = []
rsltsGBR = []
rsltsBGR = []
rsltsHBR = []
rsltsDNN = []
rsltsRNN = []
rsltsOrb = []
rsltsNN = []
dataSaveX = []
dataSaveY = []
OEforestErr = []
OEgaussErr = []
TLE = createDataLive.TLEdata()
TLE.createRandomOrbit(eval_size)
for ii in range(eval_size):
    dt = np.random.randint(5,61)
    tle = TLE.createRandomOrbit(1)
    obsrvr,reality,orbit = TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,minObs=max_obs,maxObs=max_obs,selecttle=ii)
    orbitmod = np.array([orbit[0],orbit[1],orbit[2]*d2r,orbit[3]*d2r,orbit[4]*d2r,orbit[5]*d2r])
    observer = np.transpose(obsrvr)
    realtle = reality[1]
    reals = np.transpose(reality[0])
    siderealtime = TLE.OD.siderealTime(observer[0],TLE.location)
    Dtrain,Rtrain = Dmanip.organizeDataInput(observer,siderealtime,dataseting,rangenorm)
    rsltsRFT.append(rft.predict(Dtrain))
    rsltsETR.append(etr.predict(Dtrain))
    rsltsABR.append(abr.predict(Dtrain))
    rsltsGBR.append(gbr.predict(Dtrain))
    rsltsHBR.append(hbr.predict(Dtrain))
    rsltsBGR.append(bgr.predict(Dtrain))
    gausoln = orbdet.GaussOrbitDetermination(observer[0],[observer[6].astype(float),observer[7].astype(float)],uccs,radecOrazel=0)
    gaussOE = orbdet.StateVector2OrbitalElements4(gausoln)
    OEgaussErr.append(gaussOE - orbit)
    OEforest = np.transpose([gbrOMM.predict(Dtrain),gbrOEc.predict(Dtrain),gbrOIn.predict(Dtrain),gbrORA.predict(Dtrain),gbrOAP.predict(Dtrain),gbrOMA.predict(Dtrain)])
    for line in OEforest:
        OEforestErr.append(line - orbitmod)
    X_batch = []
    y_batch = []
    for ii in range(len(Dtrain)):
        X_batch.append(torch.tensor(Dtrain[ii].astype(float),dtype=torch.float32))
        y_batch.append(torch.tensor(Rtrain[ii].astype(float),dtype=torch.float32).reshape(1))
        dataSaveX.append(Dtrain[ii])
        dataSaveY.append(Rtrain[ii])
    X_batch = np.array(X_batch)
    y_batch = torch.tensor(y_batch)
    rsltO = modelO(Dtrain[np.random.randint(0,len(Dtrain))])
    rslt = modelR(X_batch)
    rsltsNN.append(rslt.flatten().detach())
    rsltsOrb.append(rsltO.flatten().detach())
    errNN.append(rslt.flatten().detach()-y_batch)
    errOrb.append(rsltO.flatten().detach()-np.array(orbitmod))
# rsltsRFT = rft.predict(dataSaveX)
# rsltsETR = etr.predict(dataSaveX)
dataSaveY = np.array(dataSaveY)
effectRFT = np.sqrt(np.mean((rft.predict(dataSaveX)*rangenorm - dataSaveY*rangenorm)**2)) #rft.score(dataSaveX,np.array(dataSaveY))
effectETR = np.sqrt(np.mean((etr.predict(dataSaveX)*rangenorm - dataSaveY*rangenorm)**2)) #etr.score(dataSaveX,np.array(dataSaveY))
effectABR = np.sqrt(np.mean((abr.predict(dataSaveX)*rangenorm - dataSaveY*rangenorm)**2)) #abr.score(dataSaveX,np.array(dataSaveY))
effectBGR = np.sqrt(np.mean((bgr.predict(dataSaveX)*rangenorm - dataSaveY*rangenorm)**2)) #bgr.score(dataSaveX,np.array(dataSaveY))
effectGBR = np.sqrt(np.mean((gbr.predict(dataSaveX)*rangenorm - dataSaveY*rangenorm)**2)) #gbr.score(dataSaveX,np.array(dataSaveY))
effectHBR = np.sqrt(np.mean((hbr.predict(dataSaveX)*rangenorm - dataSaveY*rangenorm)**2)) #hbr.score(dataSaveX,np.array(dataSaveY))
effectTNN = np.sqrt(np.mean((np.array(errNN)*rangenorm)**2))
effectOrb = np.sqrt(np.mean((np.array(errOrb))**2))
effectOEforest = np.sqrt(np.mean(np.array(OEforestErr)**2))
effectOEgauss = np.sqrt(np.mean(np.array(OEgaussErr)**2)) # np.sum(np.isnan(OEgaussErr),0)
# effectVRR = vrr.score(dataSaveX,np.transpose(dataSaveY)[1])
# effectSGR = sgr.score(dataSaveX,np.transpose(dataSaveY)[1])
resnn = nn0.predict(dataSaveX)
scorsknn = nn0.score(dataSaveX,dataSaveY)
effectSKN = np.sqrt(np.mean((resnn*rangenorm - np.array(dataSaveY).flatten()*rangenorm)**2))

print("random-forest RMS:",effectRFT)
print("Extra-trees RMS:",effectETR)
print("Ada-boost RMS:",effectABR)
print("Bagging RMS:",effectBGR)
print("Gradient-boost RMS:",effectGBR)
print("Hist-Grad-Boost RMS:",effectHBR)
# print("Voting score:",effectVRR)
# print("Stacking score:",effectSGR)
print('NN RMS:',effectTNN)
print('SK NN RMS:',effectSKN)
print('Orbit-predict NN RMS:',effectOrb)
print('Orbit-predict forest RMS:',effectOEforest)
print('Gauss RMS:',effectOEgauss)
# f = open('scorestore.csv','a+')
# f.write(str(datalength)+','+str(effectTNN)+','+str(effectABR)+','+str(effectBGR)+','+str(effectGBR)+','+str(effectHBR)+'\n')
# f.close()


# plt.figure()
# plt.plot(rsltsDNN,'*')
# plt.plot(rsltsRNN,'*')
# plt.plot(rsltsNN,'*')
# plt.plot(rsltsRFT,'*')
# plt.plot(rsltsETR,'.')
# plt.plot(rsltsABR,'.')
# plt.plot(rsltsBGR,'.')
# plt.plot(rsltsGBR,'.')
# plt.plot(rsltsHBR,'.')
# plt.savefig('plots/NN_test1'+str(dataseting)+'.png')

# plt.figure()
# plt.plot(historyR,alpha=0.5)
# plt.plot(historyO,alpha=0.5)
# # plt.plot(historyRFT)
# plt.savefig('plots/NN_training_history'+str(dataseting)+'.png')

plt.figure()
plt.plot(Rtrain[0:len(rsltsNN[-1])] * rangenorm,':.')
plt.plot(rsltsNN[-1] * rangenorm)
plt.plot(rsltsGBR[-1] * rangenorm)
plt.plot(rsltsRFT[-1] * rangenorm)
plt.plot(rsltsBGR[-1] * rangenorm)
plt.plot(rsltsABR[-1] * rangenorm)
plt.legend(['truth','trnsfrmr','DNN','RNN','gradboost','rand frst','bagging','adaboost'])
plt.savefig('plots/MLrangeTrackingComparison'+str(dataseting)+'.png')

bins = np.linspace(-5000,5000,51)
plt.figure()
plt.hist(np.transpose(errNN)[0:3].flatten()*rangenorm,bins=bins,alpha=0.5)
plt.hist(abr.predict(dataSaveX)*rangenorm-dataSaveY*rangenorm,bins=bins,alpha=0.5)
plt.hist(bgr.predict(dataSaveX)*rangenorm-dataSaveY*rangenorm,bins=bins,alpha=0.5)
plt.hist(gbr.predict(dataSaveX)*rangenorm-dataSaveY*rangenorm,bins=bins,alpha=0.5)
plt.legend(['Transformer','Adaboost','Bagging','Grad-boost'])
plt.savefig('plots/MLerrHist'+str(dataseting)+'.png')

plt.show()
