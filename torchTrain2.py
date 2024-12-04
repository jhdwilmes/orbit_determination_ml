
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

import torch.nn as nn
import torch.optim as optim

import neural_filter
import torchNN

import createDataLive

import matplotlib as mpl
from matplotlib import pyplot as plt



#
dataseting = 6
datalegnths = {0:5,1:10,2:9,3:18,4:10,5:20,6:30}
datalength = datalegnths[dataseting]
print("Running with input size",datalength)

maxIters = 1000

# training parameters
n_epochs = 10   # number of epochs to run
batch_size = 100  # max size of each batch
num_tles = 10000
obs_noise = 0.02 # .017 = 1 arcminute of noise
batch_start = torch.arange(0, maxIters, batch_size)
test_size = 100
eval_size = 100
max_obs = 15
# min_obs = 5

dt = 10

rangenorm = 72000
d2r = np.pi/180

model0 = torchNN.Trnsfrmr1(datalength,1,(datalength,datalength))
model1 = torchNN.RNN1(datalength,1)
model2 = torchNN.Dense91(datalength,1,(100,100))
modelO = torchNN.Dense91(datalength,6,(100,100)) # torchNN.Trnsfrmr1(datalength,6,(datalength,datalength))

# model2.load_state_dict(torch.load('models/NN_best_weights',weights_only=True))
# model1.load_state_dict(torch.load('models/RNN_best_weights',weights_only=True))
# model0.load_state_dict(torch.load('models/transformer_best_weights_all',weights_only=True))
# modelO.load_state_dict(torch.load('models/Orbit_Predict_best_weights',weights_only=True))


TLE = createDataLive.TLEdata()
Dmanip = torchNN.MLdataManipulation()

loss_fn = nn.MSELoss()  # mean square error
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
optimizerO = optim.Adam(modelO.parameters(), lr=0.01)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer0 = optim.Adam(model0.parameters(), lr=0.001)

modelO.train()
model2.train()
model1.train()
model0.train()
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

estimators = 2000 # try 1000?
rft = ensemble.RandomForestRegressor(n_estimators = estimators, warm_start=True, max_depth=10) # can only train once
etr = ensemble.ExtraTreesRegressor(n_estimators = estimators,warm_start=True,max_depth=10) # can only train once
abr = ensemble.AdaBoostRegressor(n_estimators=1000,learning_rate=0.5) # joblib.load('adaboost_save_'+str(datalength)) # ensemble.AdaBoostRegressor(n_estimators=estimators,learning_rate=0.9) # can train repeatedly
bgr = ensemble.BaggingRegressor(n_estimators=2000,warm_start=True) # joblib.load('bagging_save_'+str(datalength)) # ensemble.BaggingRegressor(n_estimators=estimators) # can train repeatedly
gbr = ensemble.GradientBoostingRegressor(n_estimators=2000,learning_rate=0.1) # joblib.load('gradientboost_save_'+str(datalength)) # ensemble.GradientBoostingRegressor(n_estimators=estimators) # can train repeatedly
hbr = ensemble.HistGradientBoostingRegressor(learning_rate=0.1,max_iter=300,max_depth=20) # joblib.load('histgradboost_save_'+str(datalength)) # ensemble.HistGradientBoostingRegressor() # can train repeatedly
nn0 = neural_network.MLPRegressor(hidden_layer_sizes=(300),activation='relu',max_iter=int(n_epochs*batch_size))
# nn0 = joblib.load('sknn_save_'+str(datalength))
gbrO = ensemble.GradientBoostingRegressor(n_estimators=2000,learning_rate=0.1)

# abr = joblib.load('adaboost_save_'+str(datalength))
# gbr = joblib.load('gradientboost_save_'+str(datalength))
# hbr = joblib.load('histgradboost_save_'+str(datalength))

# Hold the best model
best_mse0 = np.inf   # init to infinity
best_weights0 = None
best_mse1 = np.inf   # init to infinity
best_weights1 = None
best_mse2 = np.inf   # init to infinity
best_weights2 = None
best_mseO = np.inf   # init to infinity
best_weightsO = None

history2 = []
history1 = []
history0 = []
historyO = []

dataSaveX = []
dataSaveY = []
orbitSave = []
# training loop
tle = TLE.createRandomOrbit(num_tles)
for epoch in range(n_epochs):
    model0.train()
    model1.train()
    model2.train()
    saveX = []
    saveY = []
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            dt = np.random.randint(5,61)
            obsrvr, reality, orbit = TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs)
            observer = np.transpose(obsrvr)
            realtle = reality[1]
            reals = np.transpose(reality[0])
            siderealtime = TLE.OD.siderealTime(observer[0],TLE.location)
            # consideration - try setting data so input is cosines and sines of angles instead of direct angular measurements?
            Dtrain,Rtrain = Dmanip.organizeDataInput(observer,siderealtime,dataseting,rangenorm)
            dataSaveX.append(Dtrain[0])
            dataSaveX.append(Dtrain[1])
            dataSaveX.append(Dtrain[2])
            dataSaveY.append(Rtrain[0])
            dataSaveY.append(Rtrain[1])
            dataSaveY.append(Rtrain[2])
            # saveX.append(Dtrain[0])
            # saveX.append(Dtrain[1])
            # saveX.append(Dtrain[2])
            # saveY.append(Rtrain[0])
            # saveY.append(Rtrain[1])
            # saveY.append(Rtrain[2])
            # dataSaveX.append(Dtrain[0:3])
            # dataSaveY.append(Rtrain[0:3])
            X_batch = []
            y_batch = []
            for ii in range(len(Dtrain)):
                X_batch.append(torch.tensor(Dtrain[ii].astype(float),dtype=torch.float32))
                y_batch.append(torch.tensor(Rtrain[ii].astype(float),dtype=torch.float32).reshape(1))
                saveX.append(Dtrain[ii])
                saveY.append(Rtrain[ii])
            X_batch = np.array(X_batch)
            y_batch = torch.tensor(y_batch).reshape(len(y_batch),1)
            orbitmod = [orbit[0],orbit[1],orbit[2]*d2r,orbit[3]*d2r,orbit[4]*d2r,orbit[5]*d2r]
            # forward pass
            y_pred1 = model1(X_batch)
            loss = loss_fn(y_pred1, y_batch)
            # backward pass
            # optimizer1.zero_grad() # should this be optimizer or model?
            # model.zero_grad()
            loss.backward()
            # update weights
            optimizer1.step()
            #
            y_pred0 = model0(X_batch)
            loss0 = loss_fn(y_pred0, y_batch)
            # optimizer0.zero_grad() # should this be optimizer or model?
            loss0.backward()
            optimizer0.step()
            # model 2 stuff
            y_pred2 = model2(X_batch)
            loss2 = loss_fn(y_pred2, y_batch)
            loss2.backward()
            optimizer2.step()
            # Orbit predicting model stuff:
            y_predO = modelO(Dtrain[np.random.randint(0,len(Dtrain))])
            lossO = loss_fn(y_predO.flatten(),torch.tensor(orbitmod,dtype=torch.float32))
            lossO.backward()
            optimizerO.step()
            # sklearn version
            nn0.partial_fit(X_batch,y_batch.flatten())
            # do states need reset?
            # print progress
            bar.set_postfix(mse=float(loss))
        orbitSave.append(orbitmod)
    # evaluate accuracy at end of each epoch
    abr.fit(saveX,saveY)
    # bgr.fit(saveX,saveY)
    gbr.fit(saveX,saveY)
    hbr.fit(saveX,saveY)
    print('Testing models')
    optimizer0.zero_grad()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizerO.zero_grad()
    model0.zero_grad()
    model1.zero_grad()
    model2.zero_grad()
    modelO.zero_grad()
    X_batch = []
    y_batch = []
    orbitsave = []
    for jj in range(test_size):
        dt = np.random.randint(5,61)
        obsrvr,reality,orbit = TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs)
        orbitmod = [orbit[0],orbit[1],orbit[2]*d2r,orbit[3]*d2r,orbit[4]*d2r,orbit[5]*d2r]
        observer = np.transpose(obsrvr)
        realtle = reality[1]
        reals = np.transpose(reality[0])
        siderealtime = TLE.OD.siderealTime(observer[0],TLE.location)
        Dtrain,Rtrain = Dmanip.organizeDataInput(observer,siderealtime,dataseting,rangenorm)
        # Xrft = rft.predict(Dtrain)
        # RftErr = np.sqrt(np.mean(np.square(Rtrain - Xrft)))
        # historyRFT.append(RftErr)
        # Errrft = abs(Xrft - Rtrain)
        for ii in range(len(Dtrain)):
            X_batch.append(torch.tensor(Dtrain[ii].astype(float),dtype=torch.float32))
            y_batch.append(torch.tensor(Rtrain[ii].astype(float),dtype=torch.float32).reshape(1))
            orbitsave.append(orbitmod)
    X_batch = np.array(X_batch)
    y_batch = torch.tensor(y_batch)
    model0.eval()
    model1.eval()
    model2.eval()
    modelO.eval()
    # t = np.random.randint(0,len(Dtest))
    y_pred2 = model2(X_batch)
    y_pred1 = model1(X_batch)
    y_pred0 = model0(X_batch)
    y_predO = modelO(X_batch)
    print('Updating models')
    mse2 = loss_fn(y_pred2.flatten(), y_batch)
    mse2 = float(mse2)
    history2.append(mse2)
    mse1 = loss_fn(y_pred1.flatten(), y_batch)
    mse1 = float(mse1)
    history1.append(mse1)
    mse0 = loss_fn(y_pred0.flatten(), y_batch)
    mse0 = float(mse0)
    history0.append(mse0)
    mseO = loss_fn(y_predO, torch.tensor(orbitsave))
    mseO = float(mseO)
    historyO.append(mseO)
    if mse1 <= best_mse1:
        best_mse1 = mse1
        best_weights1 = copy.deepcopy(model1.state_dict())
        print('RNN Improved MSE',best_mse1)
    if mse2 <= best_mse2:
        best_mse2 = mse2
        best_weights2 = copy.deepcopy(model2.state_dict())
        print('NN Improved MSE',best_mse2)
    if mse0 <= best_mse0:
        best_mse0 = mse0
        best_weights0 = copy.deepcopy(model0.state_dict())
        print('Tfrmr Improved MSE',best_mse0)
    if mseO <= best_mseO:
        best_mseO = mseO
        best_weightsO = copy.deepcopy(modelO.state_dict())
        print('OrbitPredict improved MSE',best_mseO)
    # best_weights0 = copy.deepcopy(model0.state_dict())
    resnn = nn0.predict(X_batch)
    scorsknn = nn0.score(X_batch,y_batch)
    rmsnn = np.sqrt(np.mean((resnn - np.array(y_batch).flatten())**2))
    print("RNN MSE:",mse1,"Transformer MSE:",mse0,'Dense MSE:',mse2,'sklearn version:',rmsnn,scorsknn)
    # print("RFT MSE:",np.mean(RftErr))
    print("Finished epoch:",epoch+1,'of',n_epochs)

torch.save(best_weights0,'models/transformer_bestweights_inputsize_'+str(dataseting))
torch.save(best_weights1,'models/RNN_bestweights_inputsize_'+str(dataseting))
torch.save(best_weights2,'models/NN_bestweights_inputsize_'+str(dataseting))
torch.save(best_weightsO,'models/Orbit_Predict_bestweights_inputsize_'+str(dataseting))

# gbdt_pipeline = pipeline.make_pipeline(
#     tree_preprocessor, HistGradientBoostingRegressor(random_state=0)
# )
# estimators = [
#     ("Random Forest", rf_pipeline),
#     ("Lasso", lasso_pipeline),
#     ("Gradient Boosting", gbdt_pipeline),
# ]
# vrr = ensemble.VotingRegressor((rft,etr,abr,bgr,gbr,hbr))
# sgr = ensemble.StackingRegressor((rft,etr,abr,bgr,gbr,hbr))

print("Training ensemble models")
rft.fit(dataSaveX,np.transpose(dataSaveY))
etr.fit(dataSaveX,np.transpose(dataSaveY))
abr.fit(dataSaveX,np.transpose(dataSaveY)) # only predicts one range?
bgr.fit(dataSaveX,np.transpose(dataSaveY))
gbr.fit(dataSaveX,np.transpose(dataSaveY))
hbr.fit(dataSaveX,np.transpose(dataSaveY))
# sgr.fit(dataSaveX,np.transpose(dataSaveY))
# vrr.fit(dataSaveX,np.transpose(dataSaveY))
nn0.fit(dataSaveX,np.transpose(dataSaveY))

# abr.predict(Dtrain[0].reshape(1,-1))

# restore model and return best accuracy
# model.load_state_dict(best_weights)
torch.save(model1,'models/range_rnn_'+str(dataseting))
torch.save(model0,'models/range_trnsfm_'+str(dataseting))
torch.save(model2,'models/range_nn_'+str(dataseting))
joblib.dump(rft,'models/randomforest_save_'+str(dataseting))
joblib.dump(etr,'models/extratrees_save_'+str(dataseting))
joblib.dump(abr,'models/adaboost_save_'+str(dataseting))
joblib.dump(bgr,'models/bagging_save_'+str(dataseting))
joblib.dump(gbr,'models/gradientboost_save_'+str(dataseting))
joblib.dump(hbr,'models/histgradboost_save_'+str(dataseting))
joblib.dump(nn0,'models/sknn_save_'+str(dataseting))

print("Evaluating")
# prrslts = []
model1.eval()
model0.eval()
errRNN = []
errTFRMR = []
errDNN = []
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
rsltsTFRMR = []
dataSaveX = []
dataSaveY = []
TLE = createDataLive.TLEdata()
TLE.createRandomOrbit(eval_size)
for ii in range(eval_size):
    dt = np.random.randint(5,61)
    tle = TLE.createRandomOrbit(1)
    obsrvr,reality,orbit = TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs,selecttle=ii)
    orbitmod = [orbit[0],orbit[1],orbit[2]*d2r,orbit[3]*d2r,orbit[4]*d2r,orbit[5]*d2r]
    observer = np.transpose(obsrvr)
    realtle = reality[1]
    reals = np.transpose(reality[0])
    siderealtime = TLE.OD.siderealTime(observer[0],TLE.location)
    Dtrain,Rtrain = Dmanip.organizeDataInput(observer,siderealtime,dataseting,rangenorm)
    dataSaveX.append(Dtrain[0])
    dataSaveX.append(Dtrain[1])
    dataSaveX.append(Dtrain[2])
    dataSaveY.append(Rtrain[0])
    dataSaveY.append(Rtrain[1])
    dataSaveY.append(Rtrain[2])
    rsltsRFT.append(rft.predict(Dtrain))
    rsltsETR.append(etr.predict(Dtrain))
    rsltsABR.append(abr.predict(Dtrain))
    rsltsGBR.append(gbr.predict(Dtrain))
    rsltsHBR.append(hbr.predict(Dtrain))
    rsltsBGR.append(bgr.predict(Dtrain))
    X_batch = []
    y_batch = []
    for ii in range(len(Dtrain)):
        X_batch.append(torch.tensor(Dtrain[ii].astype(float),dtype=torch.float32))
        y_batch.append(torch.tensor(Rtrain[ii].astype(float),dtype=torch.float32).reshape(1))
    X_batch = np.array(X_batch)
    y_batch = torch.tensor(y_batch)
    rsltO = modelO(Dtrain[np.random.randint(0,len(Dtrain))])
    rslt2 = model2(X_batch)
    rslt1 = model1(X_batch)
    rslt0 = model0(X_batch)
    rsltsDNN.append(rslt2.flatten().detach())
    rsltsRNN.append(rslt1.flatten().detach())
    rsltsTFRMR.append(rslt0.flatten().detach())
    rsltsOrb.append(rsltO.flatten().detach())
    errRNN.append(rslt1.flatten().detach()-y_batch)
    errTFRMR.append(rslt0.flatten().detach()-y_batch)
    errDNN.append(rslt2.flatten().detach()-y_batch)
    errOrb.append(rsltO.flatten().detach()-np.array(orbitmod))
# rsltsRFT = rft.predict(dataSaveX)
# rsltsETR = etr.predict(dataSaveX)
dataSaveY = np.array(dataSaveY)
effectRFT = np.sqrt(np.mean((rft.predict(dataSaveX)*rangenorm-dataSaveY*rangenorm)**2)) #rft.score(dataSaveX,np.array(dataSaveY))
effectETR = np.sqrt(np.mean((etr.predict(dataSaveX)*rangenorm-dataSaveY*rangenorm)**2)) #etr.score(dataSaveX,np.array(dataSaveY))
effectABR = np.sqrt(np.mean((abr.predict(dataSaveX)*rangenorm-dataSaveY*rangenorm)**2)) #abr.score(dataSaveX,np.array(dataSaveY))
effectBGR = np.sqrt(np.mean((bgr.predict(dataSaveX)*rangenorm-dataSaveY*rangenorm)**2)) #bgr.score(dataSaveX,np.array(dataSaveY))
effectGBR = np.sqrt(np.mean((gbr.predict(dataSaveX)*rangenorm-dataSaveY*rangenorm)**2)) #gbr.score(dataSaveX,np.array(dataSaveY))
effectHBR = np.sqrt(np.mean((hbr.predict(dataSaveX)*rangenorm-dataSaveY*rangenorm)**2)) #hbr.score(dataSaveX,np.array(dataSaveY))
effectRNN = np.sqrt(np.mean((np.array(errRNN)*rangenorm)**2))
effectTFM = np.sqrt(np.mean((np.array(errTFRMR)*rangenorm)**2))
effectDNN = np.sqrt(np.mean((np.array(errDNN)*rangenorm)**2))
effectOrb = np.sqrt(np.mean((np.array(errOrb))**2))
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
print('DNN RMS:',effectDNN)
print('RNN RMS:',effectRNN)
print('Transformer RMS:',effectTFM)
print('SK NN RMS:',effectSKN)
print('Orbit-predict RMS:',effectOrb)

f = open('scorestore.csv','a+')
f.write(str(datalength)+','+str(effectDNN)+','+str(effectRNN)+','+str(effectTFM)+','+str(effectABR)+','+str(effectBGR)+','+str(effectGBR)+','+str(effectHBR)+'\n')
f.close()


plt.figure()
plt.plot(rsltsDNN,'*')
plt.plot(rsltsRNN,'*')
plt.plot(rsltsTFRMR,'*')
plt.plot(rsltsRFT,'*')
plt.plot(rsltsETR,'.')
plt.plot(rsltsABR,'.')
plt.plot(rsltsBGR,'.')
plt.plot(rsltsGBR,'.')
plt.plot(rsltsHBR,'.')
plt.savefig('plots/NN_test1'+str(dataseting)+'.png')

plt.figure()
plt.plot(history2,alpha=0.5)
plt.plot(history1,alpha=0.5)
plt.plot(history0,alpha=0.5)
# plt.plot(historyRFT)
plt.savefig('plots/NN_training_history'+str(dataseting)+'.png')

plt.figure()
plt.plot(Rtrain[0:len(rsltsTFRMR[-1])] * rangenorm,':.')
plt.plot(rsltsTFRMR[-1] * rangenorm)
plt.plot(rsltsDNN[-1] * rangenorm)
plt.plot(rsltsRNN[-1] * rangenorm)
plt.plot(rsltsGBR[-1] * rangenorm)
plt.plot(rsltsRFT[-1] * rangenorm)
plt.plot(rsltsBGR[-1] * rangenorm)
plt.plot(rsltsABR[-1] * rangenorm)
plt.legend(['truth','trnsfrmr','DNN','RNN','gradboost','rand frst','bagging','adaboost'])
plt.savefig('plots/MLrangeTrackingComparison'+str(dataseting)+'.png')

bins = np.linspace(-5000,5000,51)
plt.figure()
plt.hist(np.transpose(errTFRMR)[0:3].flatten()*rangenorm,bins=bins,alpha=0.5)
plt.hist(abr.predict(dataSaveX)*rangenorm-dataSaveY*rangenorm,bins=bins,alpha=0.5)
plt.hist(bgr.predict(dataSaveX)*rangenorm-dataSaveY*rangenorm,bins=bins,alpha=0.5)
plt.hist(gbr.predict(dataSaveX)*rangenorm-dataSaveY*rangenorm,bins=bins,alpha=0.5)
plt.legend(['Transformer','Adaboost','Bagging','Grad-boost'])
plt.savefig('plots/MLerrHist'+str(dataseting)+'.png')

# plt.figure()
# plt.plot(prrslts,'r.')
# plt.plot(rslts,'b.')
# #plt.text(60000,60000,'y=x')
# plt.xlabel('Occurance')
# plt.ylabel('Error')
# # ax = plt.gca()
# # ax.set_ylim([-1e6, 1e8])
# #plt.title('Dataset:'+dataset+', Range Correlation:'+"{:.2f}".format(ccf[0][1])+', Track Correlation:'+"{:.2f}".format(neural_filter.np.mean(ccfavg[1])))
# plt.savefig('plots/NN_test'+'.png')
# #plt.savefig('RNN_LSTM_v0.png')

# plt.figure()
# plt.plot(drlsts,'k.')
# plt.ylabel('change since training')
# plt.savefig('plots/NN_change_since_training'+'.png')

plt.show()
