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

import torch.nn as nn
import torch.optim as optim

# import neural_filter
import torchNN
import OrbitDetermination as OD
import orbitFunctions
import kfnn

import createDataLive

import matplotlib as mpl
from matplotlib import pyplot as plt



class testML():
    TLE = createDataLive.TLEdata()
    orbdet = OD.orbitDetermination()
    OF = orbitFunctions.orbitFunctions()
    location = createDataLive.locations.Location('UCCS',38.89588,-104.80232,1950)
    Dmanip = torchNN.MLdataManipulation()
    
    
    def deg2rad(self,angle):
        return angle*np.pi/180


    def rad2deg(self,angle):
        return angle*180/np.pi


    def rangetranslation(self,input,direction=1,mag=72000):
        output = input
        for ii in range(len(input)):
            if direction == 0: # forward
                output[ii] = input[ii] / mag
            elif direction == 1: # backward
                output[ii] = input[ii] * mag
        return np.array(output)


    def SV2Orbit(self,SVs):
        SVs = np.array(SVs)
        orbits = []
        if len(SVs.shape) > 1:
            for ii in range(SVs.shape[0]):
                orbits.append(self.orbdet.StateVector2OrbitalElements4([self.OF.time,SVs[ii][0:3],SVs[ii][3:6]]))
        else:
            orbits = [self.orbdet.StateVector2OrbitalElements4([self.OF.time,SVs[ii][0:3],SVs[ii][3:6]])]
        return np.array(orbits)


    def compareOrbits(self,orbit1,orbit2):
        if len(orbit1) != len(orbit2):
            print("Orbit length error")
        orbitDiff = []
        orbitDot = []
        orbit1 = np.array(orbit1)
        orbit2 = np.array(orbit2)
        for ii in range(len(orbit1)):
            if ii > len(orbit2):
                break
            orbitDiff.append(orbit2[ii]-orbit1[ii])
            orbitDot.append(np.dot(orbit1[ii],orbit2[ii])/np.linalg.norm(orbit2[ii])**2)
        return orbitDiff,orbitDot


    def compareSVs(self,SV1,SV2):
        if len(SV1) != len(SV2):
            print("SV length error")
        SVDiff = []
        SVdot = []
        SV1 = np.array(SV1)
        SV2 = np.array(SV2)
        for ii in range(len(SV1)):
            if ii > len(SV2):
                break
            SVDiff.append(SV2[ii]-SV1[ii])
            SVdot.append(np.dot(SV1[ii],SV2[ii])/np.linalg.norm(SV2[ii])**2)
        return SVDiff,SVdot


    def writeData(self,data,filename='dataoutput.csv'):
        f = open(filename,'a+')
        lenshape = len(data.shape)
        for ii in range(len(data)):
            f.write(str(data.shape))
            for jj in range(len(data[ii])):
                if lenshape > 2:
                    for kk in range(len(data[ii,jj])):
                        if lenshape > 3:
                            for ll in range(len(data[ii,jj,kk])):
                                if ll < len(data[ii,jj,kk])-1:
                                    f.write(',')
                                else:
                                    f.write('\n')
                        else:
                            f.write(str(data[ii,jj,kk]))
                            if kk < len(data[ii,jj])-1:
                                f.write(',')
                            else:
                                f.write('\n')
                else:
                    f.write(str(data[ii,jj]))
                    if ll < len(data[ii])-1:
                        f.write(',')
                    else:
                        f.write('\n')
        f.close()


    def readData(self,filename='dataoutput.csv'):
        f = open(filename,'w')
        data = f.readlines()
        f.close()
        format = data[0].split(']')
        formatsplit = format[0].translate((None,'[]')).split(',')
        outputdata = []
        dataformat = []
        for ii in range(len(formatsplit)):
            dataformat.append(int(formatsplit[ii]))
        if len(dataformat > 0):
            for ii in range(dataformat[0]):
                dataline = data[ii].split(']').split(',')
                if len(dataformat > 1):
                    line1 = []
                    for jj in range(dataformat[1]):
                        if len(dataformat > 2):
                            line2 = []
                            for kk in range(dataformat[2]):
                                if len(dataformat > 3):
                                    line3 = []
                                    for ll in range(dataformat[3]):
                                        line3.append(float(dataline[ii*dataformat[0] + jj*dataformat[1] + kk*dataformat[2] + ll]))
                                    line2.append(line3)
                                else:
                                    line2.append(float(dataline[ii*dataformat[0] + jj*dataformat[1] + kk]))
                            line1.append(line2)
                        else:
                            line1.append(float(dataline[ii*dataformat[0] + jj]))
                else:
                    line1.append(float(dataline[ii]))
                outputdata.append(line1)
        return np.array(outputdata)


    def angleDiff(self,angle1,angle2):
        anglediff = angle1 - angle2
        anglediff = np.min([abs(anglediff),abs(anglediff - np.pi),abs(anglediff + np.pi)],0) * np.sign(anglediff) * (((anglediff < -np.pi) + (anglediff > np.pi))-.5)*2
        return anglediff
    
    
    def testKFML(self,KFmodel,dt=10,xstart=[],observer=[],dataseting=4,eval_size = 100,num_tles = 1000,obs_noise=.004,max_obs=10,rangenorm=72000):
        if len(observer)==0:
            observer,reality,orbit = self.TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs,selectrandtle=False)
        if len(xstart)==0:
            xstart = np.ones(KFmodel.x.shape)
        KFmodel.x = xstart
        # observer = np.transpose(observer)
        siderealtime = self.orbdet.siderealTime(observer[0],self.TLE.location)
        Dtest,Rtest = self.Dmanip.grabInputNoNorm(observer,siderealtime,dataseting)
        Dtrain,Rtrain = self.Dmanip.organizeDataInput(observer,siderealtime,dataseting,rangenorm)
        statePredictions = []
        KFmodel.setStartTime(observer[0][0])
        for ii in range(len(Dtrain)):
            x = KFmodel.forward(Dtest[ii],dt=dt)
            statePredictions.append(x)
        statePredictions = np.array(statePredictions)
        return statePredictions


    def testKF_NNonly(self,model,dt=10,observer=[],dataseting=4,eval_size = 100,num_tles = 1000,obs_noise=.004,max_obs=10,rangenorm=72000):
        if len(observer)==0:
            observer,reality,orbit = self.TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs,selectrandtle=False)
        # orbitmod = [orbit[0],orbit[1],self.deg2rad(orbit[2]),self.deg2rad(orbit[3]),self.deg2rad(orbit[4]),self.deg2rad(orbit[5])]
        # observer = np.transpose(observer)
        # reality = np.transpose(reality)
        siderealtime = self.orbdet.siderealTime(observer[0],self.TLE.location)
        Dtrain,Rtrain = self.Dmanip.grabInputNoNorm(observer,siderealtime,dataseting)
        X_batch = []
        # y_batch = []
        prediction = []
        for ii in range(len(Dtrain)):
            # X_batch.append(torch.tensor(Dtrain[ii].astype(float),dtype=torch.float32))
            # y_batch.append(torch.tensor(Rtrain[ii].astype(float),dtype=torch.float32).reshape(1))
            prediction.append(model.nn_only(Dtrain[ii].astype(float)))
        prediction = np.array(prediction)
        if len(prediction.shape) > 1:
            if prediction.shape[0] == 1 or prediction.shape[1] == 1:
                prediction = prediction.flatten()
        # X_batch = np.array(X_batch)
        # y_batch = torch.tensor(y_batch)
        # if prediction.shape[1] == 1: # range data
        #     prediction = prediction.flatten()
        # elif prediction.shape[1] == 6: # orbit prediction
        return prediction


    def testNN(self,model,dt=10,observer=[],dataseting=4,eval_size = 100,num_tles = 1000,obs_noise=.004,max_obs=10,rangenorm=72000):
        if len(observer)==0:
            observer,reality,orbit = self.TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs,selectrandtle=False)
        # orbitmod = [orbit[0],orbit[1],self.deg2rad(orbit[2]),self.deg2rad(orbit[3]),self.deg2rad(orbit[4]),self.deg2rad(orbit[5])]
        # observer = np.transpose(observer)
        # reality = np.transpose(reality)
        siderealtime = self.orbdet.siderealTime(observer[0],self.TLE.location)
        Dtrain,Rtrain = self.Dmanip.organizeDataInput(observer,siderealtime,dataseting,rangenorm)
        X_batch = []
        # y_batch = []
        for ii in range(len(Dtrain)):
            X_batch.append(torch.tensor(Dtrain[ii].astype(float),dtype=torch.float32))
            # y_batch.append(torch.tensor(Rtrain[ii].astype(float),dtype=torch.float32).reshape(1))
        X_batch = np.array(X_batch)
        # y_batch = torch.tensor(y_batch)
        prediction = model(X_batch).detach()
        # if prediction.shape[1] == 1: # range data
        #     prediction = prediction.flatten()
        # elif prediction.shape[1] == 6: # orbit prediction
        return np.array(prediction)


    def testForest(self,model,dt=10,observer=[],dataseting=4,eval_size = 100,num_tles = 1000,obs_noise=.004,max_obs=10,rangenorm=1):
        if len(observer)==0:
            observer,reality,orbit = self.TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs,selectrandtle=False)
        # orbitmod = [orbit[0],orbit[1],self.deg2rad(orbit[2]),self.deg2rad(orbit[3]),self.deg2rad(orbit[4]),self.deg2rad(orbit[5])]
        # observer = np.transpose(observer)
        # reality = np.transpose(reality)
        siderealtime = self.orbdet.siderealTime(observer[0],self.TLE.location)
        Dtrain,Rtrain = self.Dmanip.organizeDataInput(observer,siderealtime,dataseting,rangenorm)
        X_batch = []
        y_batch = []
        for ii in range(len(Dtrain)):
            X_batch.append(torch.tensor(Dtrain[ii].astype(float),dtype=torch.float32))
        X_batch = np.array(X_batch)
        prediction = model.predict(X_batch)
        # if prediction.shape[1] == 1: # range data
        #     prediction = prediction.flatten()
        # elif prediction.shape[1] == 6: # orbit prediction
        return np.array(prediction)


    def testForests(self,models,dt=10,observer=[],dataseting=4,eval_size = 100,num_tles = 1000,obs_noise=.004,max_obs=10,rangenorm=1):
        if len(observer)==0:
            observer,reality,orbit = self.TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs,selectrandtle=False)
        # orbitmod = [orbit[0],orbit[1],self.deg2rad(orbit[2]),self.deg2rad(orbit[3]),self.deg2rad(orbit[4]),self.deg2rad(orbit[5])]
        # observer = np.transpose(observer)
        # reality = np.transpose(reality)
        siderealtime = self.orbdet.siderealTime(observer[0],self.TLE.location)
        Dtrain,Rtrain = self.Dmanip.organizeDataInput(observer,siderealtime,dataseting,rangenorm)
        X_batch = []
        y_batch = []
        for ii in range(len(Dtrain)):
            X_batch.append(torch.tensor(Dtrain[ii].astype(float),dtype=torch.float32))
        X_batch = np.array(X_batch)
        predictions = []
        for ii in range(len(models)):
            predictions.append(models[ii].predict(X_batch))
        # if prediction.shape[1] == 1: # range data
        #     prediction = prediction.flatten()
        # elif prediction.shape[1] == 6: # orbit prediction
        return np.array(predictions).transpose()


    def organizeTestData(self,MLdata,observer,reality,dataType=0):
        # if len(orbit) == 0:
        #     orbit = self.orbdet.StateVector2OrbitalElements4(reality[0])
        orbitpredict = []
        orbitdiff = []
        SVdot = []
        diffset = []
        orbitDot = []
        orbitSV = []
        for ii in range(len(MLdata)):
            orbit = self.orbdet.StateVector2OrbitalElements4([0,reality[ii][0:3],reality[ii][3:6]])
            if dataType == 0:
                orbitKFML = self.orbdet.StateVector2OrbitalElements4([observer[0][ii],MLdata[ii][0:3],MLdata[ii][3:6]])
                diffset.append(MLdata[ii] - reality[ii])
                dorbit = self.angleDiff(orbitKFML,orbit)
                dorbit[0:2] = (orbitKFML - orbit)[0:2]
                orbitdiff.append(dorbit)
                orbitpredict.append(orbitKFML)
                orbitDot.append(np.dot(orbitKFML,orbit)/np.linalg.norm(orbit)**2)
                SVdot.append(np.dot(MLdata[ii],reality[ii])/np.linalg.norm(reality[ii])**2)
                orbitSV.append(MLdata[ii])
            elif dataType == 1:
                svKFML = self.orbdet.OrbitElement2StateVector4(MLdata[ii][0:6]).flatten()
                diffset.append(svKFML - reality[ii])
                dorbit = self.angleDiff(MLdata[ii],orbit)
                dorbit[0:2] = (MLdata[ii] - orbit)[0:2]
                orbitdiff.append(dorbit)
                orbitpredict.append(MLdata[ii])
                orbitDot.append(np.dot(MLdata[ii],orbit)/np.linalg.norm(orbit)**2)
                SVdot.append(np.dot(svKFML,reality[ii])/np.linalg.norm(reality[ii])**2)
                orbitSV.append(svKFML)
        return orbitSV, diffset, orbitpredict, orbitdiff, SVdot, orbitDot


    def testKFModels(self,kfmodels,train_size=100,test_size=100,obs_noise=.02,max_obs=10,num_tles=1000):
        if len(self.TLE.tles) < num_tles:
            self.TLE.createRandomOrbit(num_tles - len(self.TLE.tles) + 1)
        data = []
        orbitdiffs = []
        diffs = []
        orbits = []
        SVdots = []
        orbitDots = []
        realitySave = []
        realOrbits = []
        # for ii in range(len(kfmodels)):
        #     data.append([])
        for ii in range(train_size):
            dt = np.random.randint(5,61)
            observer,reality,orbit = self.TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs,selectrandtle=False)
            dataset = []
            diffset = []
            orbitdiff = []
            orbitpredict = []
            orbitDot = []
            SVdot = []
            gausoln = self.orbdet.GaussOrbitDetermination(observer[0],[observer[6].astype(float),observer[7].astype(float)],self.location,radecOrazel=0)
            for jj in range(len(kfmodels)):
                output = self.testKFML(kfmodels[jj],dt=dt,xstart=np.append(gausoln[1],gausoln[2]),observer=observer)
                dataset.append(output)
                diffset.append(output - reality)
                orbitd = []
                for kk in range(len(output)):
                    orbitKFML = self.orbdet.StateVector2OrbitalElements4([observer[0][0],output[0:3],output[3:6]])
                    orbitd.append(orbitKFML)
                orbitpredict.append(orbitd)
                orbitdiff.append(orbitKFML - orbit)
                SVdot.append(np.dot(output,reality,0))
            orbitdiffs.append(orbitdiff)
            diffs.append(diffset)
            orbits.append(orbitpredict)
            data.append(dataset)
            realitySave.append(reality)
            realOrbits.append(orbit)
        data = np.array(data)
        orbits = np.array(orbits)
        orbitdiffs = np.array(orbitdiffs)
        diffs = np.array(diffs)
        return data,diffs,orbits,orbitdiffs


    def testNNModels(self,nnmodels,train_size=100,test_size=100,obs_noise=.004,max_obs=10,num_tles=1000): # only to be used with direct orbit prediction methods
        if len(self.TLE.tles) < num_tles:
            self.TLE.createRandomOrbit(num_tles - len(self.TLE.tles) + 1)
        data = []
        # for ii in range(len(nnmodels)):
        #     data.append([])
        orbitdiffs = []
        diffs = []
        orbits = []
        for ii in range(train_size):
            dt = np.random.randint(5,61)
            observer,reality,orbit = self.TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs,selectrandtle=False)
            dataset = []
            diffset = []
            orbitdiff = []
            orbitpredict = []
            for jj in range(len(nnmodels)):
                output = self.testNN(nnmodels[jj],dt=dt,observer=observer)
                dataset.append(output)
                diffset.append(output - reality)
                orbitKFML = self.orbdet.StateVector2OrbitalElements4([observer[0][0],output[0:3],output[3:6]])
                orbitpredict.append(orbitKFML)
                orbitdiff.append(orbitKFML - orbit)
            orbitdiffs.append(orbitdiff)
            diffs.append(diffset)
            orbits.append(orbitpredict)
            data.append(dataset)
        data = np.array(data)
        orbits = np.array(orbits)
        orbitdiffs = np.array(orbitdiffs)
        diffs = np.array(diffs)
        return data,diffs,orbits,orbitdiffs


    def testAllTogether(self,kfmodels=[],nnmodels=[],forestOrbitModels = [],forestRangeModels=[],train_size=100,test_size=100,obs_noise=.02,max_obs=10,num_tles=1000):
        if len(self.TLE.tles) < num_tles:
            self.TLE.createRandomOrbit(num_tles - len(self.TLE.tles) + 1)
        points = sigma_points.MerweScaledSigmaPoints(6, alpha=.01, beta=2., kappa=-3)
        UKF  = UnscentedKalmanFilter(dim_x=6, dim_z=5, dt=dt, fx=self.OF.orbitFunction6, hx=self.OF.orbitSensorModelRt, points=points)
        datakf = []
        datann = []
        # for ii in range(len(kfmodels)):
        #     datakf.append([])
        # for ii in range(len(nnmodels)):
        #     datann.append([])
        for ii in range(train_size):
            dt = np.random.randint(5,61)
            observer,reality,orbit = self.TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs,selectrandtle=False)
            gausoln = self.orbdet.GaussOrbitDetermination(observer[0],[observer[6].astype(float),observer[7].astype(float)],self.location,radecOrazel=0)
            datasetkf = []
            for jj in range(len(kfmodels)):
                output = self.testKFML(kfmodels[jj],dt=dt,xstart=np.append(gausoln[1],gausoln[2]),observer=observer)
                datasetkf.append(output)
            datasetnn = []
            for jj in range(len(nnmodels)):
                output = self.testNN(nnmodels[jj],dt=dt,observer=observer)
                datasetnn.append(output)
            datasetforestrange = []
            forestranges = []
            UKF.x = np.append(gausoln[1],gausoln[2])
            for jj in range(len(forestRangeModels)):
                output = self.testForest(forestRangeModels[jj],dt=dt,observer=observer) # need to set dataseting?
                z = observer[0:5]
                z = np.insert(z[1:5],2,self.rangetranslation(output,direction=1))
                forestranges.append(output)
                UKF.predict()
                UKF.update(z)
                datasetforestrange.append(UKF.x)
            for jj in range(len(forestOrbitModels)):
                output = self.testForests(forestOrbitModels[jj],observer=observer)


    def logData(self,data,methodname='',filename='plots/orbit_results.csv',firstline=0):
        f = open('plots/orbit_results.csv','a+')
        if firstline: # unsure if TA or MA
            f.write('Method,MM-median,MM-5th,MM-95th,I-median,I-5th,I-95th,E-median,E-5th,E-95th,RA-median,RA-5th,RA-95th,AP-median,AP-5th,AP-95th,TA-median,TA-5th,TA-95th\n')
        f.write(methodname)
        for num in range(6):
            f.write(','+str(np.median(data[3,:,num]))+','+str(np.percentile(data[3,:,num],5))+','+str(np.percentile(data[3,:,num],95)))
        f.write('\n')
        f.close()


    # def testNN(self,model,dataseting=4,eval_size = 100,num_tles = 1000,obs_noise=.02,max_obs=10,rangenorm=72000):
    #     d2r = np.pi/180
    #     datasave = []
    #     if len(self.TLE.tles) < num_tles:
    #         self.TLE.createRandomOrbit(num_tles - len(self.TLE.tles) + 1)
    #     for ii in range(eval_size):
    #         dt = np.random.randint(5,61)
    #         obsrvr,reality,orbit = self.TLE.oneTLEdata(tstep=dt,tryagain=num_tles-1,noise=obs_noise,maxObs=max_obs,selectrandtle=False)
    #         orbitmod = [orbit[0],orbit[1],orbit[2]*d2r,orbit[3]*d2r,orbit[4]*d2r,orbit[5]*d2r]
    #         observer = np.transpose(obsrvr)
    #         self.OF.time = observer[0][0]
    #         realtle = reality[1]
    #         reals = np.transpose(reality[0])
    #         gausoln = self.orbdet.GaussOrbitDetermination(observer[0],[observer[6].astype(float),observer[7].astype(float)],self.location,radecOrazel=0)
    #         UKF_nn.x = np.append(gausoln[1],gausoln[2])
    #         UKF_rf.x = np.append(gausoln[1],gausoln[2])
    #         UKFAO.x = np.append(gausoln[1],gausoln[2])
    #         OEguass = self.orbdet.StateVector2OrbitalElements4(gausoln)
    #         siderealtime = self.orbdet.siderealTime(observer[0],self.TLE.location)
    #         zAzElRngAERt = np.transpose([observer[1],observer[2],observer[3],observer[8],observer[9]])
    #         Dtrain,Rtrain = Dmanip.organizeDataInput(observer,siderealtime,dataseting,rangenorm)
    #         rsltsRFT.append(rft.predict(Dtrain))
    #         rsltsETR.append(etr.predict(Dtrain))
    #         rsltsABR.append(abr.predict(Dtrain))
    #         rsltsGBR.append(gbr.predict(Dtrain))
    #         rsltsHBR.append(hbr.predict(Dtrain))
    #         zbgr = bgr.predict(Dtrain)
    #         rsltsBGR.append(zbgr)
    #         OEforest = np.transpose([gbrOMM.predict(Dtrain),gbrOEc.predict(Dtrain),gbrOIn.predict(Dtrain),gbrORA.predict(Dtrain),gbrOAP.predict(Dtrain),gbrOMA.predict(Dtrain)])
    #         # for line in OEforest:
    #         #     OEforestErr.append(line - orbitmod)
    #         #     OEforsetRslts.append(line)
    #         X_batch = []
    #         y_batch = []
    #         dataSaveX.append(Dtrain)
    #         dataSaveY.append(Rtrain[0:len(Dtrain)])
    #         for ii in range(len(Dtrain)):
    #             X_batch.append(torch.tensor(Dtrain[ii].astype(float),dtype=torch.float32))
    #             y_batch.append(torch.tensor(Rtrain[ii].astype(float),dtype=torch.float32).reshape(1))
    #         X_batch = np.array(X_batch)
    #         y_batch = torch.tensor(y_batch)
    #         rsltO = modelO(Dtrain[np.random.randint(0,len(Dtrain))])
    #         rsltR = modelR(X_batch)
    #         ukfnnbatch = []
    #         ukfrfbatch = []
    #         ukfaobatch = []
    #         ukfnnOE = []
    #         ukfrfOE = []
    #         ukfaoOE = []
    #         nnOE = []
    #         nnOEbatch = []
    #         Dtest,Rtest = Dmanip.grabInputNoNorm(observer,siderealtime,dataseting)
    #         nnkf.setStartTime(observer[0][0])
    #         effect = []
    #         effectp = []
    #         effectv = []
    #         for ii in range(len(rsltR)):
    #             zR = zAzElRngAERt[ii]
    #             zR[2] = Dmanip.rangetranslation(np.array(rsltR[ii].detach())[0],1)
    #             UKF_nn.predict(dt=dt)
    #             UKF_nn.update(zR)
    #             zF = zAzElRngAERt[ii]
    #             zF[2] = Dmanip.rangetranslation(zbgr[ii],1)
    #             UKF_rf.predict(dt=dt)
    #             UKF_rf.update(zF)
    #             z = np.append(zAzElRngAERt[ii][0:2],zAzElRngAERt[ii][3:5])
    #             UKFAO.predict(dt=dt)
    #             UKFAO.update(z)
    #             nnkfx = nnkf.forward(Dtest[ii],dt=dt)
    #             nnoex = oenn.nn_only(Dtest[ii],dt=dt)
    #             if ii > 2: # first couple values are usually wrong and should not be included in the results
    #                 ukfnnbatch.append(nnkfx)
    #                 ukfrfbatch.append(UKF_rf.x)
    #                 ukfaobatch.append(UKFAO.x)
    #                 nnOEbatch.append(nnoex)
    #                 truvector = orbdet.StateVector2OrbitalElements4([OF.time,reality[ii][0:3],reality[ii][3:6]])
    #                 ukfnnOE.append(orbdet.StateVector2OrbitalElements4([OF.time,nnkfx[0:3],nnkfx[3:6]])-truvector)
    #                 ukfrfOE.append(orbdet.StateVector2OrbitalElements4([OF.time,UKF_rf.x[0:3],UKF_rf.x[3:6]])-truvector)
    #                 ukfaoOE.append(orbdet.StateVector2OrbitalElements4([OF.time,UKFAO.x[0:3],UKFAO.x[3:6]])-truvector)
    #                 nnOE.append(orbdet.StateVector2OrbitalElements4([OF.time,nnoex[0:3],nnoex[3:6]])-truvector)
    #                 OEforestErr.append(OEforest[ii] - truvector)
    #                 OEforestRslts.append(OEforest[ii])
    #                 gausErr.append(OEguass - truvector)
    #                 forestSV = orbdet.OrbitElement2StateVector4(OEforest[ii])
    #                 nnop = np.array(rsltO.detach().flatten())
    #                 nnop[2:] = nnop[2:] #/d2r
    #                 nnSV = orbdet.OrbitElement2StateVector4(nnop)
    #                 eff = [np.array(gausoln[1:]).flatten() - reality[ii],UKFAO.x - reality[ii],UKF_rf.x - reality[ii],np.array(forestSV).flatten()-reality[ii],nnkfx - reality[ii],nnoex-reality[ii],np.array(nnSV).flatten()-reality[ii]]
    #                 p = reality[ii][0:3]
    #                 v = reality[ii][3:]
    #                 effp = [np.dot(np.array(gausoln[1]).flatten(),p),np.dot(UKFAO.x[0:3],p),np.dot(UKF_rf.x[0:3],p),np.dot(np.array(forestSV[0]).flatten(),p),np.dot(nnkfx[0:3],p),np.dot(nnoex[0:3],p),np.dot(np.array(nnSV[0]).flatten(),p)]
    #                 effv = [np.dot(np.array(gausoln[2]).flatten(),v),np.dot(UKFAO.x[3:],v),np.dot(UKF_rf.x[3:],v),np.dot(np.array(forestSV[1]).flatten(),v),np.dot(nnkfx[3:],v),np.dot(nnoex[3:],v),np.dot(np.array(nnSV[1]).flatten(),v)]
    #                 effect.append([np.linalg.norm(eff[0]),np.linalg.norm(eff[1]),np.linalg.norm(eff[2]),np.linalg.norm(eff[3]),np.linalg.norm(eff[4]),np.linalg.norm(eff[5]),np.linalg.norm(eff[6])])
    #                 effectp.append(np.array(effp)/np.linalg.norm(p)**2)
    #                 effectv.append(np.array(effv)/np.linalg.norm(v)**2)
    #         OENNKF.append(ukfnnOE)
    #         OEKFAO.append(ukfaoOE)
    #         OERFKF.append(ukfrfOE)
    #         OENNerr.append(nnOE)
    #         OENNrslts.append(nnOEbatch)
    #         rsltsNNKF.append(ukfnnbatch)
    #         rsltsRFKF.append(ukfrfbatch)
    #         rsltsKFAO.append(ukfaobatch)
    #         rsltsDNN.append(rsltR.flatten().detach())
    #         rsltsOrb.append(rsltO.flatten().detach())
    #         errDNN.append(rsltR.flatten().detach()-y_batch)
    #         errOrb.append(rsltO.flatten().detach()-np.array(orbitmod))
    #         effectsave.append(effect)
    #         effectsavep.append(effectp)
    #         effectsavev.append(effectv)
    #         if ii % 100 == 0:
    #             print('Finished iteration',ii-1,'of',eval_size)

