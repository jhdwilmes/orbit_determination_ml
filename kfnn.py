# from filterpy.kalman import kalman_filter
import copy
from filterpy.kalman import sigma_points
from filterpy.kalman import unscented_transform
from filterpy.kalman import UnscentedKalmanFilter

import numpy as np
from numpy import linalg
import kalman_filter

import OrbitDetermination
import orbitFunctions
import torchNN
import neural_filter


class orbitkfnn(kalman_filter.OrbitKalmanFilter):

    def SFNNAO(self,ztrue,xh,u = 0,alpha=None,dt = 60): # angles-only update version
        # Kalman Filter implementation specifically for orbits, based on https://www.mdpi.com/2411-9660/5/3/54
        # utilizes a Sigma-Point Kalman Filter
        # 
        # Step 1a
        x = np.concatenate((xh,np.zeros([len(self.SigmaW),1]))).astype('d')
        x = np.concatenate((x,np.transpose([ztrue[1:]]))).astype('d') # what am I doing here? ztrue*0
        if(type(alpha) == 'NoneType'):
            alpha = self.createAlpha(x)
        alpham = alpha
        alphac = alpha
        Sigma = linalg.block_diag(self.SigmaX,self.SigmaW,self.SigmaV)
        Chi = np.transpose(self.createChi(x,Sigma)) # sigma points - not sure why Chi is the normal notation for this
        ChiX = np.transpose(Chi[0:len(xh)])
        ChiW = Chi[len(xh):len(xh)+len(self.SigmaW)]
        ChiV = Chi[len(xh)+len(self.SigmaW):]
        #Chi1 = np.transpose(Chi)
        ii = 0
        Xxkm = []
        while(ii<len(Chi[0])):
            Ah = self.createOrbitAhat6(ChiX[ii],dt)
            Xxkm.append(np.matmul(Ah,xh) + np.transpose([ChiX[ii]]))
            #self.A*ChiX[ii] + self.B*ChiW[ii]
            ii+=1
        #ChiX = np.transpose(ChiX)
        xkm = np.matmul(np.transpose(ChiX),alpham).astype('d')# + xh # need to verify equations here are right
        # step 1b
        ii = 0
        SigmaX = np.zeros([len(self.SigmaX),len(self.SigmaX)])
        while(ii<len(alphac)):
            ChiXxh = np.transpose([ChiX[ii]]) - xh
            SigmaX += alphac[ii] * np.matmul(ChiXxh,np.transpose(ChiXxh)).astype('d')
            ii+=1
        d = np.diag(SigmaX)
        ii=0
        while(ii<len(d)):
            if(d[ii] == 0):
                #print('Correcting Sigma 0')
                SigmaX[ii][ii] = 1 # some reasonable error to prevent zeroing of error and subsequent crash
            ii+=1
        #self.SigmaX = SigmaX
        # step 1c
        ii = 0
        Z = []
        while(ii<len(Chi[0])):
            Z.append(self.orbitSensorModelAO(Xxkm[ii],ztrue))
            ii+=1
        z = np.matmul(np.transpose(Z),alpham)
        # step 2a
        Zzk = np.zeros([len(z),len(z)])
        Zxzk = np.zeros([len(xh),len(z)])
        ii=0
        while(ii<len(alphac)):
            ChiXxh = np.transpose([ChiX[ii]]) - xh
            ChiZz = np.transpose([Z[ii]]) - z
            Zzk += alphac[ii]*np.matmul(ChiZz,np.transpose(ChiZz))
            Zxzk += alphac[ii]*np.matmul(ChiXxh,np.transpose(ChiZz))
            ii+=1
        Sigmazk = Zzk #(Z-z)*np.transpose(Z-z)*alphac
        Sigmaxzk = Zxzk #(Xxkm-xkm)*np.transpose(Z-z)*alphac
        d = np.linalg.det(Sigmazk)
        if(d!=0): # Sigmazk is non-determinable, do not update SigmaX
            Lk = np.matmul(Sigmaxzk,np.linalg.inv(Sigmazk))
            # step 2b
            dz = np.transpose([ztrue[1:]]) - z
            xh = xkm + np.matmul(Lk,dz).astype('d')
            # step 2c
            self.SigmaX = SigmaX - np.matmul(np.matmul(Lk,Sigmazk),np.transpose(Lk))
            d = np.diag(SigmaX)
            ii = 0
            while(ii<len(d)):
                if(d[ii] == 0):
                    print('Problematix AO SigmaX')
                    self.SigmaX[ii][ii] = 1e-6
                ii+=1
        else:
            print('Singular Sigma Encountered, predicted AO state:',np.concatenate(xh),' sensor input:',ztrue)
        return [xh,self.SigmaX]



class KFNN_v0():
    # What data does the neural network need to work with?  What does it modify in the kalman filter?
    # Idea 1 - neural network has access to KF inputs and KF covariance (diagonal?), improves result to reduce covariance
    def __init__(self,dim = 6,dimz = 5,dimin = 5,dt=10,dnntype=0,covarianceStrength=1e-6):
        self.dt = 10
        self.OF = orbitFunctions.orbitFunctions()
        self.points = sigma_points.MerweScaledSigmaPoints(dim, alpha=.1, beta=2., kappa=-3)
        self.filter = UnscentedKalmanFilter(dim_x=6, dim_z=dimz, dt=dt, fx=self.OF.orbitFunction6, hx=self.OF.orbitSensorModelRt, points=self.points)
        self.dnn = torchNN.Dense91(dimin,1,(300,300),activation_fc=torchNN.torch.nn.functional.relu)
        if dnntype == 1:
            self.dnn = torchNN.RNN1(dimin,1,(300,300),activation_fc=torchNN.torch.nn.functional.relu,rnnsets=1,layersperset=3)
        elif dnntype == 2:
            self.dnn = torchNN.Trnsfrmr1(dimin,1,(300,300),activation_fc=torchNN.torch.nn.functional.relu,hiddenlayers=0,encoderlayers=1)
        elif dnntype == 3:
            self.dnn = torchNN.Trnsfrmr2(dimin,1,(300,300),activation_fc=torchNN.torch.nn.functional.relu,hiddenlayers=1,encoderlayers=1)
        self.loss_fn = torchNN.torch.nn.MSELoss()
        self.optimizer = torchNN.torch.optim.Adam(self.dnn.parameters(), lr=0.001)
        self.location = neural_filter.locations.Location('UCCS',38.89588,-104.80232,1950)
        self.filter.R = np.diag([0.02,0.02,1e2,.02,.02]) # measurement error? 1e2 vs 1e3
        self.filter.P = np.diag([1e6,1e6,1e6,1e2,1e2,1e2])
        self.filter.x = np.array([1e6,-1e6,0,-7,7,0])
        self.covtuning = covarianceStrength


    def saveNN(self,filename='models/KFNN_v0_dnn'):
        best_weights = copy.deepcopy(self.dnn.state_dict())
        torchNN.torch.save(best_weights,filename)


    def loadNN(self,filename='models/KFNN_v0_dnn'):
        self.dnn.load_state_dict(torchNN.torch.load(filename,weights_only=True))


    def setStartTime(self,time):
        self.OF.time = time


    def kalmantranslation(self,input,range=0):
        output = input
        # d2r = np.pi/180
        if self.dnn.input_layer.in_features == 10 and len(input) == 10:
            output = np.array([np.arctan2(input[2],input[3]),np.arctan2(input[4],input[5]),range,np.arctan2(input[6],input[7]),np.arctan2(input[8],input[9])])
        elif self.dnn.input_layer.in_features == 20 and len(input) == 20:
            output = np.array([np.arctan2(input[2],input[3]),np.arctan2(input[4],input[5]),range,np.arctan2(input[6],input[7]),np.arctan2(input[8],input[9])])
        elif self.dnn.input_layer.in_features == 30 and len(input) == 30:
            output = np.array([np.arctan2(input[12],input[13]),np.arctan2(input[14],input[15]),range,np.arctan2(input[16],input[17]),np.arctan2(input[18],input[19])])
        return output


    def inputranslation(self,input):
        output = input
        d2r = np.pi/180
        if self.dnn.input_layer.in_features == 4:
            output = np.array(input * d2r).astype(float)
        elif self.dnn.input_layer.in_features == 5 and len(input) == 5:
            output = np.array(input * d2r).astype(float)
        elif self.dnn.input_layer.in_features == 8 and len(input) == 4:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3])]).astype(float)
        elif self.dnn.input_layer.in_features == 9 and len(input) == 5:
            input = np.array(input) * d2r
            output = np.array([input[0],np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4])]).astype(float)
        elif self.dnn.input_layer.in_features == 10 and len(input) == 5:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4])]).astype(float)
        elif self.dnn.input_layer.in_features == 20 and len(input) == 10:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4]),
                               np.sin(input[5]),np.cos(input[5]),np.sin(input[6]),np.cos(input[6]),np.sin(input[7]),np.cos(input[7]),np.sin(input[8]),np.cos(input[8]),np.sin(input[9]),np.cos(input[9])]).astype(float)
        elif self.dnn.input_layer.in_features == 30 and len(input) == 15:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4]),
                               np.sin(input[5]),np.cos(input[5]),np.sin(input[6]),np.cos(input[6]),np.sin(input[7]),np.cos(input[7]),np.sin(input[8]),np.cos(input[8]),np.sin(input[9]),np.cos(input[9]),
                               np.sin(input[10]),np.cos(input[10]),np.sin(input[11]),np.cos(input[11]),np.sin(input[12]),np.cos(input[12]),np.sin(input[13]),np.cos(input[13]),np.sin(input[14]),np.cos(input[14])]).astype(float)
        return torchNN.torch.tensor(output.astype(np.float32),dtype=torchNN.torch.float32)


    def rangetranslation(self,range,direction=1,mag=72000):
        output = range
        if direction == 0: # forward
            output = range / mag
        elif direction == 1: # backward
            output = range * mag
        return torchNN.torch.tensor([output],dtype=torchNN.torch.float32)


    def forward(self,inputdata,dt=None,dnndata=None):
        if dt == None:
            dt = self.dt
        dnnin = self.inputranslation(inputdata)
        r = dnndata
        if dnndata == None:
            r = self.dnn(dnnin)[0]
        self.filter.predict(dt)
        # z = self.kalmantranslation(inputdata)
        # if len(origdata) > 0:
        #     z = origdata
        z = inputdata[0:5]
        z = np.insert(z[1:5],2,self.rangetranslation(r.detach(),direction=1))
        # z = inputdata
        # z[2] = self.rangetranslation(r,direction=1)
        self.filter.update(z)
        return self.filter.x


    def nn_only(self,inputdata,dt=None):
        if dt == None:
            dt = self.dt
        dnnin = self.inputranslation(inputdata)
        r = self.dnn(dnnin)[0].detach()
        return self.rangetranslation(r,direction=1)


    def trainstep(self,inputdata,inputrange,dt=None):
        if dt == None:
            dt = self.dt
        # P0 = np.copy(self.filter.P)
        dnnin = self.inputranslation(inputdata)
        rpred = self.dnn(dnnin)[0]
        if inputrange > 100:
            inputrange = self.rangetranslation(inputrange,direction=0)
        rerr = self.loss_fn(rpred, inputrange)
        self.forward(inputdata,dt=dt,dnndata=rpred)
        P1 = self.filter.P #self.filter.z - inputdata #self.filter.P
        Perr = self.loss_fn(torchNN.torch.tensor([np.linalg.norm(P1)*self.covtuning]),torchNN.torch.tensor([0.])) # want to drive the coveriance of the system to zero - but might need an offset to avoid model not fitting
        # loss = rerr + Perr
        (rerr + Perr).backward()
        self.optimizer.step()


    def trainBatch(self,inputdata,inputrange,dt=None):
        if dt == None:
            dt = self.dt
        # P0 = np.copy(self.filter.P)
        Zsaves = []
        inputs = []
        for ii in range(len(inputdata)):
            dnnin = self.inputranslation(inputdata[ii])
            if np.sum(self.filter.z == None) > 0:
                self.filter.z = np.zeros(self.filter.z.shape)
            self.filter.predict(dt) # can have prediction occur here...
            inputs.append(dnnin)
            rpred = self.dnn(inputs[-1])[0]
            if inputrange[ii] > 100:
                inputrange[ii] = self.rangetranslation(inputrange[ii],direction=0)
            # self.forward(inputdata,dt=dt)
            # self.filter.predict(dt) # ...or here.
            # rerr = self.loss_fn(rpred, inputrange[ii])
            z = inputdata[ii][0:5]
            z = np.insert(z[1:5],2,self.rangetranslation(rpred.detach(),direction=1))
            self.filter.update(z)
            Zinput = np.copy(self.filter.y)
            Zinput[2] = self.rangetranslation(Zinput[2],0) # to prevent range error exploding z error
            Z1 = np.array([np.tan(Zinput[0] * .5*np.pi/180),np.tan(Zinput[1] * .5*np.pi/180),Zinput[2],np.tan(Zinput[3] * .5*np.pi/180),np.tan(Zinput[4] * .5*np.pi/180)])
            # Zs = torchNN.torch.tensor(Z1)#np.linalg.norm(Z1))
            Zsaves.append(Z1)
        Zsaves = torchNN.torch.tensor(np.array(Zsaves),dtype=torchNN.torch.float32)
        Zerr = self.loss_fn(Zsaves,torchNN.torch.zeros_like(Zsaves,dtype=torchNN.torch.float32))
        rpreds = self.dnn(np.array(inputs))[:,0]
        rerr = self.loss_fn(rpreds, torchNN.torch.tensor(inputrange,dtype=torchNN.torch.float32))
        # loss = rerr + Perr
        (rerr + Zerr).backward()
        self.optimizer.step()


class KFNN_v1():
    # What data does the neural network need to work with?  What does it modify in the kalman filter?
    # Idea 1 - neural network has access to KF inputs, maybe KF residuals?, improves result to reduce residuals
    def __init__(self,dim = 6,dimz = 5,dimin = 5,dt=10,dnntype=0,residualStrength=1e-1):
        self.dt = 10
        self.OF = orbitFunctions.orbitFunctions()
        self.points = sigma_points.MerweScaledSigmaPoints(dim, alpha=.1, beta=2., kappa=-3)
        self.filter = UnscentedKalmanFilter(dim_x=6, dim_z=dimz, dt=dt, fx=self.OF.orbitFunction6, hx=self.OF.orbitSensorModelRt, points=self.points)
        self.dnn = torchNN.Dense91(dimin,1,(100,100),activation_fc=torchNN.torch.nn.functional.relu)
        if dnntype == 1:
            self.dnn = torchNN.RNN1(dimin,1,(100,100),activation_fc=torchNN.torch.nn.functional.relu,rnnsets=1,layersperset=3)
        elif dnntype == 2:
            self.dnn = torchNN.Trnsfrmr1(dimin,1,(100,100),activation_fc=torchNN.torch.nn.functional.relu,hiddenlayers=0,encoderlayers=1)
        elif dnntype == 3:
            self.dnn = torchNN.Trnsfrmr2(dimin,1,(100,100),activation_fc=torchNN.torch.nn.functional.relu,hiddenlayers=1,encoderlayers=1)
        self.loss_fn = torchNN.torch.nn.MSELoss()
        self.optimizer = torchNN.torch.optim.Adam(self.dnn.parameters(), lr=0.001)
        self.location = neural_filter.locations.Location('UCCS',38.89588,-104.80232,1950)
        self.filter.R = np.diag([0.02,0.02,1e2,.02,.02]) # measurement error? 1e2 vs 1e3
        self.filter.P = np.diag([1e6,1e6,1e6,1e2,1e2,1e2])
        self.filter.x = np.array([1e6,-1e6,0,-7,7,0])
        self.residualtuning = residualStrength


    def saveNN(self,filename='models/KFNN_v1_dnn'):
        best_weights = copy.deepcopy(self.dnn.state_dict())
        torchNN.torch.save(best_weights,filename)


    def loadNN(self,filename='models/KFNN_v1_dnn'):
        self.dnn.load_state_dict(torchNN.torch.load(filename,weights_only=True))


    def setStartTime(self,time):
        self.OF.time = time


    def horizon_to_az_elev(self,top_s, top_e, top_z):
        range_sat = torchNN.torch.sqrt((top_s * top_s) + (top_e * top_e) + (top_z * top_z))
        elevation = torchNN.torch.asin(top_z / range_sat)
        azimuth = torchNN.torch.atan2(-top_e, top_s) + torchNN.torch.pi
        return azimuth, elevation


    def orbitFunction6(self,xh,dt):#,u,xw,dt): # 6-variable version of the orbital dynamics model
        # dt = self.dt
        r = torchNN.torch.sqrt(xh[0]**2 + xh[1]**2 + xh[2]**2)
        cr = -orbitFunctions.constants.MU_E/r**3
        A1 = [0,0,0,1,0,0]
        A2 = [0,0,0,0,1,0]
        A3 = [0,0,0,0,0,1]
        A4 = [cr,0,0,0,0,0]
        A5 = [0,cr,0,0,0,0]
        A6 = [0,0,cr,0,0,0]
        A = torchNN.torch.tensor([A1,A2,A3,A4,A5,A6])
        xh1 = torchNN.torch.matmul(A,xh)*dt + xh #+ xw
        return xh1


    def orbitSensorModelRt(self,xh):#,zh):
        if(xh.ndim>1):
            xh = xh.flatten()
        time = self.OF.time # zh[0] # need to ensure this is correct
        t = orbitFunctions.utils.gstime_from_datetime(self.OF.Jtime2Datetime(time))
        ecef = orbitFunctions.coordinate_systems.eci_to_ecef(xh[0:3],t)
        # horizon = orbitFunctions.coordinate_systems.to_horizon(self.OF.Sensor.latitude_rad,self.OF.Sensor.longitude_rad,self.OF.Sensor.position_ecef,ecef)
        # azel = self.horizon_to_az_elev(horizon[0],horizon[1],horizon[2])
        # azel = [azel[0]*180/np.pi, azel[1]*180/np.pi]
        rng = self.OF.Sensor.slant_range_km(ecef)
        # observation = [azel[0],azel[1],rng] # torchNN.torch.hstack([azel[0],azel[1],azelrate[0],azelrate[1]])
        return rng #observation


    def sensorOrbitModelRt(self,rng,ob):
        if(ob.ndim>1):
            ob = ob.flatten()
        time = self.OF.time # zh[0] # need to ensure this is correct
        t = orbitFunctions.utils.gstime_from_datetime(self.OF.Jtime2Datetime(time))
        # need coordinate rotation to go from observation to orbit


    def inputranslation(self,input):
        output = input
        d2r = np.pi/180
        if self.dnn.input_layer.in_features == 4:
            output = np.array(input * d2r).astype(float)
        elif self.dnn.input_layer.in_features == 5 and len(input) == 5:
            output = np.array(input * d2r).astype(float)
        elif self.dnn.input_layer.in_features == 8 and len(input) == 4:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3])]).astype(float)
        elif self.dnn.input_layer.in_features == 9 and len(input) == 5:
            input = np.array(input) * d2r
            output = np.array([input[0],np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4])]).astype(float)
        elif self.dnn.input_layer.in_features == 10 and len(input) == 5:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4])]).astype(float)
        elif self.dnn.input_layer.in_features == 20 and len(input) == 10:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4]),
                               np.sin(input[5]),np.cos(input[5]),np.sin(input[6]),np.cos(input[6]),np.sin(input[7]),np.cos(input[7]),np.sin(input[8]),np.cos(input[8]),np.sin(input[9]),np.cos(input[9])]).astype(float)
        elif self.dnn.input_layer.in_features == 30 and len(input) == 15:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4]),
                               np.sin(input[5]),np.cos(input[5]),np.sin(input[6]),np.cos(input[6]),np.sin(input[7]),np.cos(input[7]),np.sin(input[8]),np.cos(input[8]),np.sin(input[9]),np.cos(input[9]),
                               np.sin(input[10]),np.cos(input[10]),np.sin(input[11]),np.cos(input[11]),np.sin(input[12]),np.cos(input[12]),np.sin(input[13]),np.cos(input[13]),np.sin(input[14]),np.cos(input[14])]).astype(float)
        return torchNN.torch.tensor(output.astype(np.float32),dtype=torchNN.torch.float32)


    def rangetranslation(self,range,direction=1,mag=72000):
        output = range
        if direction == 0: # forward
            output = range / mag
        elif direction == 1: # backward
            output = range * mag
        return torchNN.torch.tensor([output],dtype=torchNN.torch.float32)


    def forward(self,inputdata,dt=None):
        if dt == None:
            dt = self.dt
        dnnin = self.inputranslation(inputdata)
        r = self.dnn(dnnin)[0].detach()
        self.filter.predict(dt)
        z = inputdata[0:5]
        z = np.insert(z[1:5],2,self.rangetranslation(r.detach(),direction=1))
        # z = inputdata
        # z[2] = self.rangetranslation(r,direction=1)
        self.filter.update(z)
        return self.filter.x


    def nn_only(self,inputdata,dt=None):
        if dt == None:
            dt = self.dt
        dnnin = self.inputranslation(inputdata)
        r = self.dnn(dnnin)[0].detach()
        return self.rangetranslation(r,direction=1)


    def trainstep(self,inputdata,inputrange,dt=None):
        if dt == None:
            dt = self.dt
        # P0 = np.copy(self.filter.P)
        dnnin = self.inputranslation(inputdata)
        rpred = self.dnn(dnnin)[0]
        if inputrange > 100:
            inputrange = self.rangetranslation(inputrange,direction=0)
        rerr = self.loss_fn(rpred, inputrange)
        self.forward(inputdata,dt=dt)
        Z1 = self.filter.y # - inputdata[-5:] # is this better or 0:5?
        Z1[2] = self.rangetranslation(Z1[2],0) # to prevent range error exploding z error
        Z1[0] = np.tan(Z1[0]*.5*np.pi/180);Z1[1] = np.tan(Z1[1]*.5*np.pi/180);Z1[3] = np.tan(Z1[3]*.5*np.pi/180);Z1[4] = np.tan(Z1[4]*.5*np.pi/180)
        Perr = self.loss_fn(torchNN.torch.tensor([np.linalg.norm(Z1)*self.residualtuning]),torchNN.torch.tensor([0.])) # want to drive the coveriance of the system to zero - but might need an offset to avoid model not fitting
        # loss = rerr + Perr
        (rerr + Perr).backward()
        self.optimizer.step()


    def trainBatch(self,inputdata,inputrange,dt=None):
        if dt == None:
            dt = self.dt
        # P0 = np.copy(self.filter.P)
        Zsaves = []
        inputs = []
        for ii in range(len(inputdata)):
            dnnin = self.inputranslation(inputdata[ii])
            if np.sum(self.filter.z == None) > 0:
                self.filter.z = np.zeros(self.filter.z.shape)
            self.filter.predict(dt) # can have prediction occur here...
            inputs.append(dnnin)
            rpred = self.dnn(inputs[-1])[0]
            if inputrange[ii] > 100:
                inputrange[ii] = self.rangetranslation(inputrange[ii],direction=0)
            # self.forward(inputdata,dt=dt)
            # self.filter.predict(dt) # ...or here.
            # rerr = self.loss_fn(rpred, inputrange[ii])
            z = inputdata[ii][0:5]
            z = np.insert(z[1:5],2,self.rangetranslation(rpred.detach(),direction=1))
            self.filter.update(z)
            Zinput = np.copy(self.filter.y)
            Zinput[2] = self.rangetranslation(Zinput[2],0) # to prevent range error exploding z error
            Z1 = np.array([np.tan(Zinput[0] * .5*np.pi/180),np.tan(Zinput[1] * .5*np.pi/180),Zinput[2],np.tan(Zinput[3] * .5*np.pi/180),np.tan(Zinput[4] * .5*np.pi/180)])
            # Zs = torchNN.torch.tensor(Z1)#np.linalg.norm(Z1))
            Zsaves.append(Z1)
        Zsaves = torchNN.torch.tensor(Zsaves,dtype=torchNN.torch.float32)
        Zerr = self.loss_fn(Zsaves,torchNN.torch.zeros_like(Zsaves,dtype=torchNN.torch.float32))
        rpreds = self.dnn(np.array(inputs))[:,0]
        rerr = self.loss_fn(rpreds, torchNN.torch.tensor(inputrange,dtype=torchNN.torch.float32))
        # loss = rerr + Perr
        (rerr + Zerr).backward()
        self.optimizer.step()


class KFNN_v2():
    # Idea 2 - neural network has access to KF inputs and (normalized) state prediction, improves result to improve matching
    # Maybe try to minimize residuals in measurements (residuals as error)?
    def __init__(self,dim = 6,dimz = 5,dimin = 5,dt=10,dnntype=0,covarianceStrength=1e-6,zerrstrength=1.,usezdiff = False):
        self.dt = 10
        self.OF = orbitFunctions.orbitFunctions()
        self.points = sigma_points.MerweScaledSigmaPoints(dim, alpha=.1, beta=2., kappa=-3)
        self.filter = UnscentedKalmanFilter(dim_x=6, dim_z=dimz, dt=dt, fx=self.OF.orbitFunction6, hx=self.OF.orbitSensorModelRt, points=self.points)
        self.dnn = torchNN.Dense91(dimin+dimz+dimz-2,1,(300,300),activation_fc=torchNN.torch.nn.functional.relu)
        if dnntype == 1:
            self.dnn = torchNN.RNN1(dimin+dimz+dimz-2,1,(300,300),activation_fc=torchNN.torch.nn.functional.relu,rnnsets=1,layersperset=3)
        elif dnntype == 2:
            self.dnn = torchNN.Trnsfrmr1(dimin+dimz+dimz-2,1,(300,300),activation_fc=torchNN.torch.nn.functional.relu,hiddenlayers=0,encoderlayers=1)
        elif dnntype == 3:
            self.dnn = torchNN.Trnsfrmr2(dimin+dimz+dimz-2,1,(300,300),activation_fc=torchNN.torch.nn.functional.relu,hiddenlayers=1,encoderlayers=1)
        self.loss_fn = torchNN.torch.nn.MSELoss()
        self.optimizer = torchNN.torch.optim.Adam(self.dnn.parameters(), lr=0.001)
        self.location = neural_filter.locations.Location('UCCS',38.89588,-104.80232,1950)
        self.filter.R = np.diag([0.02,0.02,1e2,.02,.02]) # measurement error? 1e2 vs 1e3
        self.filter.P = np.diag([1e6,1e6,1e6,1e2,1e2,1e2])
        self.filter.x = np.array([1e6,-1e6,0,-7,7,0])
        self.covtuning = covarianceStrength
        self.ztuning = zerrstrength
        self.usey = usezdiff


    def saveNN(self,filename='models/KFNN_v2_dnn'):
        best_weights = copy.deepcopy(self.dnn.state_dict())
        torchNN.torch.save(best_weights,filename)


    def loadNN(self,filename='models/KFNN_v2_dnn'):
        self.dnn.load_state_dict(torchNN.torch.load(filename,weights_only=True))


    def setStartTime(self,time):
        self.OF.time = time


    def inputranslation(self,input):
        output = input
        d2r = np.pi/180
        if self.dnn.input_layer.in_features == 4:
            output = np.array(input * d2r).astype(float)
        elif self.dnn.input_layer.in_features == 5 and len(input) == 5:
            output = np.array(input * d2r).astype(float)
        elif self.dnn.input_layer.in_features == 8 and len(input) == 4:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3])]).astype(float)
        elif self.dnn.input_layer.in_features == 9 and len(input) == 5:
            input = np.array(input) * d2r
            output = np.array([input[0],np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4])]).astype(float)
        elif len(input) == 4:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3])]).astype(float)
        elif len(input) == 5:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4])]).astype(float)
        elif len(input) == 10:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4]),
                               np.sin(input[5]),np.cos(input[5]),np.sin(input[6]),np.cos(input[6]),np.sin(input[7]),np.cos(input[7]),np.sin(input[8]),np.cos(input[8]),np.sin(input[9]),np.cos(input[9])]).astype(float)
        elif len(input) == 15:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4]),
                               np.sin(input[5]),np.cos(input[5]),np.sin(input[6]),np.cos(input[6]),np.sin(input[7]),np.cos(input[7]),np.sin(input[8]),np.cos(input[8]),np.sin(input[9]),np.cos(input[9]),
                               np.sin(input[10]),np.cos(input[10]),np.sin(input[11]),np.cos(input[11]),np.sin(input[12]),np.cos(input[12]),np.sin(input[13]),np.cos(input[13]),np.sin(input[14]),np.cos(input[14])]).astype(float)
        return torchNN.torch.tensor(output.astype(np.float32),dtype=torchNN.torch.float32)


    def rangetranslation(self,range,direction=1,mag=72000):
        output = range
        if direction == 0: # forward
            output = range / mag
        elif direction == 1: # backward
            output = range * mag
        return torchNN.torch.tensor([output],dtype=torchNN.torch.float32)


    def horizon_to_az_elev(self,top_s, top_e, top_z):
        range_sat = torchNN.torch.sqrt((top_s * top_s) + (top_e * top_e) + (top_z * top_z))
        elevation = torchNN.torch.asin(top_z / range_sat)
        azimuth = torchNN.torch.atan2(-top_e, top_s) + torchNN.torch.pi
        return azimuth, elevation


    def orbitFunction6(self,xh,dt):#,u,xw,dt): # 6-variable version of the orbital dynamics model
        # dt = self.dt
        r = torchNN.torch.sqrt(xh[0]**2 + xh[1]**2 + xh[2]**2)
        cr = -orbitFunctions.constants.MU_E/r**3
        A1 = [0,0,0,1,0,0]
        A2 = [0,0,0,0,1,0]
        A3 = [0,0,0,0,0,1]
        A4 = [cr,0,0,0,0,0]
        A5 = [0,cr,0,0,0,0]
        A6 = [0,0,cr,0,0,0]
        A = torchNN.torch.tensor([A1,A2,A3,A4,A5,A6])
        xh1 = torchNN.torch.matmul(A,xh)*dt + xh #+ xw
        return xh1


    def orbitSensorModelAORts(self,xh):#,zh):
        if(xh.ndim>1):
            xh = xh.flatten()
        time = self.OF.time # zh[0] # need to ensure this is correct
        t = orbitFunctions.utils.gstime_from_datetime(self.OF.Jtime2Datetime(time))
        ecef = orbitFunctions.coordinate_systems.eci_to_ecef(xh[0:3],t)
        horizon = orbitFunctions.coordinate_systems.to_horizon(self.OF.Sensor.latitude_rad,self.OF.Sensor.longitude_rad,self.OF.Sensor.position_ecef,ecef)
        azel = self.horizon_to_az_elev(horizon[0],horizon[1],horizon[2])
        azel = [azel[0]*180/np.pi, azel[1]*180/np.pi]
        dt = 1 # timedelta in seconds for rate calculations
        xh1 = self.orbitFunction6(xh,dt)
        t1 = (t + dt/86400) % (np.pi * 2)
        ecef1 = orbitFunctions.coordinate_systems.eci_to_ecef(xh1[0:3],t1)
        horizon1 = orbitFunctions.coordinate_systems.to_horizon(self.OF.Sensor.latitude_rad,self.OF.Sensor.longitude_rad,self.OF.Sensor.position_ecef,ecef1)
        azel1 = self.horizon_to_az_elev(horizon1[0],horizon1[1],horizon1[2])
        azelrate = [azel1[0]*180/np.pi - azel[0], azel1[1]*180/np.pi - azel[1]]
        observation = [azel[0],azel[1],azelrate[0],azelrate[1]] # torchNN.torch.hstack([azel[0],azel[1],azelrate[0],azelrate[1]])
        return observation


    def forward(self,inputdata,dt=None):
        if dt == None:
            dt = self.dt
        if np.sum(self.filter.z == None) > 0:
            self.filter.z = np.zeros(self.filter.z.shape)
        dnnin = self.inputranslation(inputdata)
        if np.sum(self.filter.z == None) > 0:
            self.filter.z = np.zeros(self.filter.z.shape)
        self.filter.predict(dt) # can have prediction occur here...
        znnin = self.inputranslation(np.append(self.filter.z.flatten()[0:2],self.filter.z.flatten()[3:])*np.pi/180)
        if self.usey:
            znnin = self.inputranslation(np.append(self.filter.y.flatten()[0:2],self.filter.y.flatten()[3:])*np.pi/180)
        r = self.dnn(torchNN.torch.concat([dnnin,znnin.flatten()]))[0].detach()
        # self.filter.predict(dt) # ...or here.
        z = inputdata[0:5]
        z = np.insert(z[1:5],2,self.rangetranslation(r.detach(),direction=1))
        # z = inputdata
        # z[2] = self.rangetranslation(r,direction=1)
        self.filter.update(z)
        return self.filter.x


    def nn_only(self,inputdata,dt=None,Z=None):
        if dt == None:
            dt = self.dt
        dnnin = self.inputranslation(inputdata)
        if np.sum(self.filter.z == None) > 0:
            self.filter.z = np.zeros(self.filter.z.shape)
        self.filter.predict(dt)
        znnin = Z
        if Z == None:
            znnin = self.inputranslation(np.append(self.filter.z.flatten()[0:2],self.filter.z.flatten()[3:])*np.pi/180)
            if self.usey:
                znnin = self.inputranslation(np.append(self.filter.y.flatten()[0:2],self.filter.y.flatten()[3:])*np.pi/180)
        r = self.dnn(torchNN.torch.concat([dnnin,znnin.flatten()]))[0].detach()
        z = inputdata[0:5]
        z = np.insert(z[1:5],2,self.rangetranslation(r,direction=1))
        self.filter.update(z)
        return self.rangetranslation(r,direction=1)


    def trainstep(self,inputdata,inputrange,dt=None):
        if dt == None:
            dt = self.dt
        # P0 = np.copy(self.filter.P)
        dnnin = self.inputranslation(inputdata)
        if np.sum(self.filter.z == None) > 0:
            self.filter.z = np.zeros(self.filter.z.shape)
        self.filter.predict(dt) # can have prediction occur here...
        znnin = self.inputranslation(np.append(self.filter.z.flatten()[0:2],self.filter.z.flatten()[3:])*np.pi/180) # what is filter.y and filter.z?
        if self.usey:
            znnin = self.inputranslation(np.append(self.filter.y.flatten()[0:2],self.filter.y.flatten()[3:])*np.pi/180)
        rpred = self.dnn(torchNN.torch.concat([dnnin,znnin.flatten()]))[0]
        if inputrange > 100:
            inputrange = self.rangetranslation(inputrange,direction=0)
        rerr = self.loss_fn(rpred, inputrange)
        # self.forward(inputdata,dt=dt)
        # self.filter.predict(dt) # ...or here.
        z = inputdata[0:5]
        z = np.insert(z[1:5],2,self.rangetranslation(rpred.detach(),direction=1))
        self.filter.update(z)
        Zinput = np.copy(self.filter.y)
        Zinput[2] = self.rangetranslation(Zinput[2],0) # to prevent range error exploding z error
        Z1 = np.array([np.tan(Zinput[0] * .5*np.pi/180),np.tan(Zinput[1] * .5*np.pi/180),Zinput[2],np.tan(Zinput[3] * .5*np.pi/180),np.tan(Zinput[4] * .5*np.pi/180)])
        Zs = torchNN.torch.tensor(Z1)#np.linalg.norm(Z1))
        Zerr = self.loss_fn(Zs,torchNN.torch.zeros_like(Zs)) * self.ztuning
        P1 = self.filter.P
        Perr = self.loss_fn(torchNN.torch.tensor([np.linalg.norm(P1)*self.covtuning]),torchNN.torch.tensor([0.])) # want to drive the coveriance of the system to zero - but might need an offset to avoid model not fitting
        # loss = rerr + Perr
        (rerr + Perr + Zerr).backward()
        self.optimizer.step()


    def trainBatch(self,inputdata,inputrange,dt=None):
        if dt == None:
            dt = self.dt
        # P0 = np.copy(self.filter.P)
        Zsaves = []
        inputs = []
        for ii in range(len(inputdata)):
            dnnin = self.inputranslation(inputdata[ii])
            if np.sum(self.filter.z == None) > 0:
                self.filter.z = np.zeros(self.filter.z.shape)
            self.filter.predict(dt) # can have prediction occur here...
            znnin = self.inputranslation(np.append(self.filter.z.flatten()[0:2],self.filter.z.flatten()[3:])*np.pi/180) # what is filter.y and filter.z?
            if self.usey:
                znnin = self.inputranslation(np.append(self.filter.y.flatten()[0:2],self.filter.y.flatten()[3:])*np.pi/180)
            inputs.append(np.concatenate([dnnin,znnin]))
            rpred = self.dnn(inputs[-1])[0]
            if inputrange[ii] > 100:
                inputrange[ii] = self.rangetranslation(inputrange[ii],direction=0)
            # self.forward(inputdata,dt=dt)
            # self.filter.predict(dt) # ...or here.
            # rerr = self.loss_fn(rpred, inputrange[ii])
            z = inputdata[ii][0:5]
            z = np.insert(z[1:5],2,self.rangetranslation(rpred.detach(),direction=1))
            self.filter.update(z)
            Zinput = np.copy(self.filter.y)
            Zinput[2] = self.rangetranslation(Zinput[2],0) # to prevent range error exploding z error
            Z1 = np.array([np.tan(Zinput[0] * .5*np.pi/180),np.tan(Zinput[1] * .5*np.pi/180),Zinput[2],np.tan(Zinput[3] * .5*np.pi/180),np.tan(Zinput[4] * .5*np.pi/180)])
            # Zs = torchNN.torch.tensor(Z1)#np.linalg.norm(Z1))
            Zsaves.append(Z1)
        Zsaves = torchNN.torch.tensor(np.array(Zsaves),dtype=torchNN.torch.float32)
        Zerr = self.loss_fn(Zsaves,torchNN.torch.zeros_like(Zsaves,dtype=torchNN.torch.float32)) * self.ztuning
        rpreds = self.dnn(np.array(inputs))[:,0]
        rerr = self.loss_fn(rpreds, torchNN.torch.tensor(inputrange,dtype=torchNN.torch.float32))
        # loss = rerr + Perr
        (rerr + Zerr).backward()
        self.optimizer.step()


class KFNN_v3():
    # Idea 3 - neural network has access to KF inputs and KF expected observations, improves result to match (replace update step?)
    # What data does the neural network need to work with?  What does it modify in the kalman filter?
    # Idea 1 - neural network has access to KF inputs and KF covariance (diagonal?), improves result to reduce covariance
    def __init__(self,dim = 6,dimz = 4,dimin = 10,dt=10,dnntype=0,covarianceStrength=1e-6,zerrstrength=1.,usezerr=False):
        self.dt = 10
        self.OF = orbitFunctions.orbitFunctions()
        self.points = sigma_points.MerweScaledSigmaPoints(dim, alpha=.1, beta=2., kappa=-3)
        self.filter = UnscentedKalmanFilter(dim_x=6, dim_z=dimz, dt=dt, fx=self.OF.orbitFunction6, hx=self.OF.orbitSensorModelAORts, points=self.points)
        self.dnn = torchNN.Dense91(dimin+dimz+dimz,dim,(300,300),activation_fc=torchNN.torch.nn.functional.relu)
        if dnntype == 1:
            self.dnn = torchNN.RNN1(dimin+dimz+dimz,dim,(300,300),activation_fc=torchNN.torch.nn.functional.relu,rnnsets=1,layersperset=3)
        elif dnntype == 2:
            self.dnn = torchNN.Trnsfrmr1(dimin+dimz+dimz,dim,(300,300),activation_fc=torchNN.torch.nn.functional.relu,hiddenlayers=0,encoderlayers=1)
        elif dnntype == 3:
            self.dnn = torchNN.Trnsfrmr2(dimin+dimz+dimz,dim,(300,300),activation_fc=torchNN.torch.nn.functional.relu,hiddenlayers=1,encoderlayers=1)
        self.loss_fn = torchNN.torch.nn.MSELoss()
        self.optimizer = torchNN.torch.optim.Adam(self.dnn.parameters(), lr=0.001)
        self.location = neural_filter.locations.Location('UCCS',38.89588,-104.80232,1950)
        self.filter.R = np.diag([0.02,0.02,.02,.02]) # measurement error? 1e2 vs 1e3
        self.filter.P = np.diag([1e3,1e3,1e3,1e1,1e1,1e1])
        self.filter.x = np.array([1e6,-1e6,0,-7,7,0])
        self.forwardConversionMatrix = np.diag([1e6,1e6,1e6,1e1,1e1,1e1])
        self.backwardConversionMatrix = torchNN.torch.tensor(np.diag([1e-6,1e-6,1e-6,1e-1,1e-1,1e-1]),dtype=torchNN.torch.float32)
        self.covtuning = covarianceStrength
        self.ztuning = zerrstrength
        self.usey = usezerr


    def saveNN(self,filename='models/KFNN_v3_dnn'):
        best_weights = copy.deepcopy(self.dnn.state_dict())
        torchNN.torch.save(best_weights,filename)


    def loadNN(self,filename='models/KFNN_v3_dnn'):
        self.dnn.load_state_dict(torchNN.torch.load(filename,weights_only=True))


    def setStartTime(self,time):
        self.OF.time = time


    def horizon_to_az_elev(self,top_s, top_e, top_z):
        range_sat = torchNN.torch.sqrt((top_s * top_s) + (top_e * top_e) + (top_z * top_z))
        elevation = torchNN.torch.asin(top_z / range_sat)
        azimuth = torchNN.torch.atan2(-top_e, top_s) + torchNN.torch.pi
        return azimuth, elevation


    def orbitFunction6(self,xh,dt):#,u,xw,dt): # 6-variable version of the orbital dynamics model
        # dt = self.dt
        r = torchNN.torch.sqrt(xh[0]**2 + xh[1]**2 + xh[2]**2)
        cr = -orbitFunctions.constants.MU_E/r**3
        A1 = [0,0,0,1,0,0]
        A2 = [0,0,0,0,1,0]
        A3 = [0,0,0,0,0,1]
        A4 = [cr,0,0,0,0,0]
        A5 = [0,cr,0,0,0,0]
        A6 = [0,0,cr,0,0,0]
        A = torchNN.torch.tensor([A1,A2,A3,A4,A5,A6])
        xh1 = torchNN.torch.matmul(A,xh)*dt + xh #+ xw
        return xh1


    def orbitSensorModelAORts(self,xh):#,zh):
        if(xh.ndim>1):
            xh = xh.flatten()
        time = self.OF.time # zh[0] # need to ensure this is correct
        t = orbitFunctions.utils.gstime_from_datetime(self.OF.Jtime2Datetime(time))
        ecef = orbitFunctions.coordinate_systems.eci_to_ecef(xh[0:3],t)
        horizon = orbitFunctions.coordinate_systems.to_horizon(self.OF.Sensor.latitude_rad,self.OF.Sensor.longitude_rad,self.OF.Sensor.position_ecef,ecef)
        azel = self.horizon_to_az_elev(horizon[0],horizon[1],horizon[2])
        azel = [azel[0]*180/np.pi, azel[1]*180/np.pi]
        dt = 1 # timedelta in seconds for rate calculations
        xh1 = self.orbitFunction6(xh,dt)
        t1 = (t + dt/86400) % (np.pi * 2)
        ecef1 = orbitFunctions.coordinate_systems.eci_to_ecef(xh1[0:3],t1)
        horizon1 = orbitFunctions.coordinate_systems.to_horizon(self.OF.Sensor.latitude_rad,self.OF.Sensor.longitude_rad,self.OF.Sensor.position_ecef,ecef1)
        azel1 = self.horizon_to_az_elev(horizon1[0],horizon1[1],horizon1[2])
        azelrate = [azel1[0]*180/np.pi - azel[0], azel1[1]*180/np.pi - azel[1]]
        observation = [azel[0],azel[1],azelrate[0],azelrate[1]] # torchNN.torch.hstack([azel[0],azel[1],azelrate[0],azelrate[1]])
        return observation


    def inputranslation(self,input):
        output = input
        d2r = np.pi/180
        if self.dnn.input_layer.in_features == 4:
            output = np.array(input * d2r).astype(float)
        elif self.dnn.input_layer.in_features == 5 and len(input) == 5:
            output = np.array(input * d2r).astype(float)
        elif self.dnn.input_layer.in_features == 8 and len(input) == 4:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3])]).astype(float)
        elif self.dnn.input_layer.in_features == 9 and len(input) == 5:
            input = np.array(input) * d2r
            output = np.array([input[0],np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4])]).astype(float)
        elif len(input) == 4:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3])]).astype(float)
        elif len(input) == 5:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4])]).astype(float)
        elif len(input) == 10:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4]),
                               np.sin(input[5]),np.cos(input[5]),np.sin(input[6]),np.cos(input[6]),np.sin(input[7]),np.cos(input[7]),np.sin(input[8]),np.cos(input[8]),np.sin(input[9]),np.cos(input[9])]).astype(float)
        elif len(input) == 15:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4]),
                               np.sin(input[5]),np.cos(input[5]),np.sin(input[6]),np.cos(input[6]),np.sin(input[7]),np.cos(input[7]),np.sin(input[8]),np.cos(input[8]),np.sin(input[9]),np.cos(input[9]),
                               np.sin(input[10]),np.cos(input[10]),np.sin(input[11]),np.cos(input[11]),np.sin(input[12]),np.cos(input[12]),np.sin(input[13]),np.cos(input[13]),np.sin(input[14]),np.cos(input[14])]).astype(float)
        return torchNN.torch.tensor(output.astype(np.float32),dtype=torchNN.torch.float32)


    def rangetranslation(self,range,direction=1,mag=72000):
        output = range
        if direction == 0: # forward
            output = range / mag
        elif direction == 1: # backward
            output = range * mag
        return torchNN.torch.tensor([output],dtype=torchNN.torch.float32)


    def forward(self,inputdata,dt=None):
        if dt == None:
            dt = self.dt
        if np.sum(self.filter.z == None) > 0:
            self.filter.z = np.zeros(self.filter.z.shape)
        dnnin = self.inputranslation(inputdata)
        znnin = self.inputranslation(self.filter.z.flatten()*np.pi/180) # what is filter.y and filter.z?
        if self.usey:
            znnin = self.inputranslation(self.filter.y.flatten()*np.pi/180) # what is filter.y and filter.z?
        x = self.dnn(torchNN.torch.concat([dnnin,znnin]))[0].detach()
        try:
            self.filter.predict(dt) # is this step needed?
        except:
            # print("Failed prediction - covariance possibly inaccurate")
            self.filter.P = np.diag([1e3,1e3,1e3,1e1,1e1,1e1])
        self.filter.x = np.array(np.matmul(x, self.forwardConversionMatrix))
        # z = np.append(np.append(inputdata[1:3],self.rangetranslation(r,direction=1)),inputdata[3:5])
        z = np.array(inputdata[1:5])
        # z = inputdata
        # z[2] = self.rangetranslation(r,direction=1)
        self.filter.update(z)
        self.filter.x = np.array(np.matmul(x, self.forwardConversionMatrix)) # Does this need to be done here also?
        return self.filter.x


    def nn_only(self,inputdata,dt=None,Z=None):
        if dt == None:
            dt = self.dt
        dnnin = self.inputranslation(inputdata)
        znnin = Z
        if Z == None:
            znnin = self.inputranslation(self.filter.z.flatten()*np.pi/180) # what is filter.y and filter.z?
            if self.usey:
                znnin = self.inputranslation(self.filter.y.flatten()*np.pi/180) # what is filter.y and filter.z?
        x = self.dnn(torchNN.torch.concat([dnnin,znnin]))[0].detach()
        try:
            self.filter.predict(dt) # is this step needed?
        except:
            # print("Failed prediction - covariance possibly inaccurate")
            self.filter.P = np.diag([1e3,1e3,1e3,1e1,1e1,1e1])
        self.filter.x = np.array(np.matmul(x, self.forwardConversionMatrix))
        z = np.array(inputdata[1:5])
        self.filter.update(z)
        return np.matmul(x, self.forwardConversionMatrix)


    def trainstep(self,inputdata,inputSV,dt=None):
        if dt == None:
            dt = self.dt
        # P0 = np.copy(self.filter.P)
        dnnin = self.inputranslation(inputdata)
        # self.forward(inputdata, dt=dt)
        if np.sum(self.filter.z == None) > 0:
            self.filter.z = np.zeros(self.filter.z.shape)
        znnin = self.inputranslation(self.filter.z.flatten()*np.pi/180) # what is filter.y and filter.z?
        if self.usey:
            znnin = self.inputranslation(self.filter.y.flatten()*np.pi/180) # what is filter.y and filter.z?
        xpred = self.dnn(torchNN.torch.concat([dnnin,znnin]))[0]
        z = np.array(inputdata[0:5])
        xerr = self.loss_fn(xpred, torchNN.torch.matmul(torchNN.torch.tensor(inputSV,dtype=torchNN.torch.float32), self.backwardConversionMatrix))
        self.filter.x = np.array(np.matmul(xpred.detach().numpy(), self.forwardConversionMatrix))
        try:
            self.filter.predict(dt) # is this step needed?
        except: # x gets reset every time, this simply ensures it isn't nan
            self.filter.x = np.array(np.matmul(xpred.detach().numpy(), self.forwardConversionMatrix))
        self.filter.update(z[1:5])
        P1 = self.filter.P
        Zinput = self.filter.y
        # Zinput[2] = self.rangetranslation(Zinput[2],0) # to prevent range error exploding z error
        Z1 = np.tan(Zinput * .5*np.pi/180)
        # Z1[0] = np.tan(Z1[0]);Z1[1] = np.tan(Z1[1]);Z1[3] = np.tan(Z1[3]);Z1[4] = np.tan(Z1[4])
        Zs = torchNN.torch.tensor(Z1)#np.linalg.norm(Z1))
        Zerr = self.loss_fn(np.tan(.5*Zs),torchNN.torch.zeros_like(Zs)) * self.ztuning # tangent or sine?
        # Perr = self.loss_fn(torchNN.torch.tensor([np.linalg.norm(P1)*self.covtuning]),torchNN.torch.tensor([0.])) # want to drive the coveriance of the system to zero - but might need an offset to avoid model not fitting
        Pinput = torchNN.torch.matmul(torchNN.torch.tensor(np.diag(self.filter.P),dtype=torchNN.torch.float32),self.backwardConversionMatrix) * self.covtuning
        Perr = self.loss_fn(Pinput,torchNN.torch.zeros_like(Pinput))
        # loss = rerr + Perr
        (xerr.type(torchNN.torch.float32) + Perr.type(torchNN.torch.float32) + Zerr.type(torchNN.torch.float32)).backward()
        self.optimizer.step()


    def trainBatch(self,inputdata,inputSV,dt=None):
        if dt == None:
            dt = self.dt
        # P0 = np.copy(self.filter.P)
        xpreds = []
        physloss = []
        SVs = []
        Zsave = []
        inputs = []
        for ii in range(len(inputdata)):
            dnnin = self.inputranslation(inputdata[ii])
            # self.forward(inputdata, dt=dt)
            if np.sum(self.filter.z == None) > 0:
                self.filter.z = np.zeros(self.filter.z.shape)
            znnin = self.inputranslation(self.filter.z.flatten()*np.pi/180) # what is filter.y and filter.z?
            if self.usey:
                znnin = self.inputranslation(self.filter.y.flatten()*np.pi/180) # what is filter.y and filter.z?
            inputs.append(np.concatenate([dnnin,znnin]))
            xpred = self.dnn(inputs[-1])[0]
            xpreds.append(xpred)
            SVs.append(torchNN.torch.matmul(torchNN.torch.tensor(inputSV[ii],dtype=torchNN.torch.float32), self.backwardConversionMatrix))
            z = np.array(inputdata[ii][0:5])
            self.filter.x = np.array(np.matmul(xpred.detach().numpy(), self.forwardConversionMatrix))
            try:
                self.filter.predict(dt) # is this step needed?
            except: # x gets reset every time, this simply ensures it isn't nan
                self.filter.x = np.array(np.matmul(xpred.detach().numpy(), self.forwardConversionMatrix))
            self.filter.update(z[1:5])
            P1 = self.filter.P
            Zinput = self.filter.y
            # a = torchNN.torch.randn(size=([6]), requires_grad=True)
            d = torchNN.torch.hstack(self.orbitSensorModelAORts(xpred))
            physloss.append(d)
            # d.backward()
            # a.grad
            # Zinput[2] = self.rangetranslation(Zinput[2],0) # to prevent range error exploding z error
            Z1 = np.tan(Zinput * .5*np.pi/180)
            # Z1[0] = np.tan(Z1[0]);Z1[1] = np.tan(Z1[1]);Z1[3] = np.tan(Z1[3]);Z1[4] = np.tan(Z1[4])
            # Zs = torchNN.torch.tensor(Z1)#np.linalg.norm(Z1))
            Zsave.append(Z1)
        # xpreds = self.dnn(np.array(inputs))
        # self.loss_fn(torchNN.torch.vstack(physloss),torchNN.torch.tensor(np.vstack(inputdata)[:,1:5]))
        # Zsave = torchNN.torch.tensor(Zsave,dtype=torchNN.torch.float32)
        # Zerr = self.loss_fn(np.tan(.5*Zsave),torchNN.torch.zeros_like(Zsave)) * self.ztuning # tangent or sine?
        # xerr = self.loss_fn(xpreds, torchNN.torch.tensor(np.array(SVs),dtype=torchNN.torch.float32))
        Zerr = self.loss_fn(torchNN.torch.vstack(physloss),torchNN.torch.tensor(np.vstack(inputdata)[:,1:5],dtype=torchNN.torch.float32))
        xerr = self.loss_fn(torchNN.torch.vstack(xpreds),torchNN.torch.vstack(SVs))
        # loss = rerr + Perr
        (xerr.type(torchNN.torch.float32) + Zerr.type(torchNN.torch.float32)).backward()
        self.optimizer.step()


class KFNN_v4():
    # What data does the neural network need to work with?  What does it modify in the kalman filter?
    # Idea 1 - neural network has access to KF inputs and KF covariance (diagonal?), improves result to reduce covariance
    def __init__(self,dim = 6,dimz = 4,dimin = 10,dt=10,dnntype=0,covarianceStrength=1e-3,zerrstrength=1.):
        self.dt = 10
        self.OF = orbitFunctions.orbitFunctions()
        self.points = sigma_points.MerweScaledSigmaPoints(dim, alpha=.1, beta=2., kappa=-3)
        self.filter = UnscentedKalmanFilter(dim_x=6, dim_z=dimz, dt=dt, fx=self.OF.orbitFunction6, hx=self.OF.orbitSensorModelAORts, points=self.points)
        self.dnn = torchNN.Dense91(dimin,dim,(100,100),activation_fc=torchNN.torch.nn.functional.relu)
        if dnntype == 1:
            self.dnn = torchNN.RNN1(dimin,dim,(100,100),activation_fc=torchNN.torch.nn.functional.relu,rnnsets=1,layersperset=3)
        elif dnntype == 2:
            self.dnn = torchNN.Trnsfrmr1(dimin,dim,(100,100),activation_fc=torchNN.torch.nn.functional.relu,hiddenlayers=0,encoderlayers=1)
        elif dnntype == 3:
            self.dnn = torchNN.Trnsfrmr2(dimin,dim,(100,100),activation_fc=torchNN.torch.nn.functional.relu,hiddenlayers=1,encoderlayers=1)
        self.loss_fn = torchNN.torch.nn.MSELoss()
        self.optimizer = torchNN.torch.optim.Adam(self.dnn.parameters(), lr=0.001)
        self.location = neural_filter.locations.Location('UCCS',38.89588,-104.80232,1950)
        self.filter.R = np.diag([0.02,0.02,.02,.02]) # measurement error? 1e2 vs 1e3
        self.filter.P = np.diag([1e3,1e3,1e3,1e1,1e1,1e1])
        self.filter.x = np.array([1e6,-1e6,0,-7,7,0])
        self.forwardConversionMatrix = np.diag([1e6,1e6,1e6,1e1,1e1,1e1])
        self.backwardConversionMatrix = torchNN.torch.tensor(np.diag([1e-6,1e-6,1e-6,1e-1,1e-1,1e-1]),dtype=torchNN.torch.float32)
        self.covtuning = covarianceStrength
        self.ztuning = zerrstrength


    def saveNN(self,filename='models/KFNN_v4_dnn'):
        best_weights = copy.deepcopy(self.dnn.state_dict())
        torchNN.torch.save(best_weights,filename)


    def loadNN(self,filename='models/KFNN_v4_dnn'):
        self.dnn.load_state_dict(torchNN.torch.load(filename,weights_only=True))


    def setStartTime(self,time):
        self.OF.time = time


    def horizon_to_az_elev(self,top_s, top_e, top_z):
        range_sat = torchNN.torch.sqrt((top_s * top_s) + (top_e * top_e) + (top_z * top_z))
        elevation = torchNN.torch.asin(top_z / range_sat)
        azimuth = torchNN.torch.atan2(-top_e, top_s) + torchNN.torch.pi
        return azimuth, elevation


    def orbitFunction6(self,xh,dt):#,u,xw,dt): # 6-variable version of the orbital dynamics model
        # dt = self.dt
        r = torchNN.torch.sqrt(xh[0]**2 + xh[1]**2 + xh[2]**2)
        cr = -orbitFunctions.constants.MU_E/r**3
        A1 = [0,0,0,1,0,0]
        A2 = [0,0,0,0,1,0]
        A3 = [0,0,0,0,0,1]
        A4 = [cr,0,0,0,0,0]
        A5 = [0,cr,0,0,0,0]
        A6 = [0,0,cr,0,0,0]
        A = torchNN.torch.tensor([A1,A2,A3,A4,A5,A6])
        xh1 = torchNN.torch.matmul(A,xh)*dt + xh #+ xw
        return xh1


    def orbitSensorModelAORts(self,xh):#,zh):
        if(xh.ndim>1):
            xh = xh.flatten()
        time = self.OF.time # zh[0] # need to ensure this is correct
        t = orbitFunctions.utils.gstime_from_datetime(self.OF.Jtime2Datetime(time))
        ecef = orbitFunctions.coordinate_systems.eci_to_ecef(xh[0:3],t)
        horizon = orbitFunctions.coordinate_systems.to_horizon(self.OF.Sensor.latitude_rad,self.OF.Sensor.longitude_rad,self.OF.Sensor.position_ecef,ecef)
        azel = self.horizon_to_az_elev(horizon[0],horizon[1],horizon[2])
        azel = [azel[0]*180/np.pi, azel[1]*180/np.pi]
        dt = 1 # timedelta in seconds for rate calculations
        xh1 = self.orbitFunction6(xh,dt)
        t1 = (t + dt/86400) % (np.pi * 2)
        ecef1 = orbitFunctions.coordinate_systems.eci_to_ecef(xh1[0:3],t1)
        horizon1 = orbitFunctions.coordinate_systems.to_horizon(self.OF.Sensor.latitude_rad,self.OF.Sensor.longitude_rad,self.OF.Sensor.position_ecef,ecef1)
        azel1 = self.horizon_to_az_elev(horizon1[0],horizon1[1],horizon1[2])
        azelrate = [azel1[0]*180/np.pi - azel[0], azel1[1]*180/np.pi - azel[1]]
        observation = [azel[0],azel[1],azelrate[0],azelrate[1]] # torchNN.torch.hstack([azel[0],azel[1],azelrate[0],azelrate[1]])
        return observation


    def inputranslation(self,input):
        output = input
        d2r = np.pi/180
        if self.dnn.input_layer.in_features == 4:
            output = np.array(input * d2r).astype(float)
        elif self.dnn.input_layer.in_features == 5 and len(input) == 5:
            output = np.array(input * d2r).astype(float)
        elif self.dnn.input_layer.in_features == 8 and len(input) == 4:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3])]).astype(float)
        elif self.dnn.input_layer.in_features == 9 and len(input) == 5:
            input = np.array(input) * d2r
            output = np.array([input[0],np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4])]).astype(float)
        elif self.dnn.input_layer.in_features == 10 and len(input) == 5:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4])]).astype(float)
        elif self.dnn.input_layer.in_features == 20 and len(input) == 10:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4]),
                               np.sin(input[5]),np.cos(input[5]),np.sin(input[6]),np.cos(input[6]),np.sin(input[7]),np.cos(input[7]),np.sin(input[8]),np.cos(input[8]),np.sin(input[9]),np.cos(input[9])]).astype(float)
        elif self.dnn.input_layer.in_features == 30 and len(input) == 15:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4]),
                               np.sin(input[5]),np.cos(input[5]),np.sin(input[6]),np.cos(input[6]),np.sin(input[7]),np.cos(input[7]),np.sin(input[8]),np.cos(input[8]),np.sin(input[9]),np.cos(input[9]),
                               np.sin(input[10]),np.cos(input[10]),np.sin(input[11]),np.cos(input[11]),np.sin(input[12]),np.cos(input[12]),np.sin(input[13]),np.cos(input[13]),np.sin(input[14]),np.cos(input[14])]).astype(float)
        return torchNN.torch.tensor(output,dtype=torchNN.torch.float32)


    def rangetranslation(self,range,direction=1,mag=72000):
        output = range
        if direction == 0: # forward
            output = range / mag
        elif direction == 1: # backward
            output = range * mag
        return torchNN.torch.tensor([output],dtype=torchNN.torch.float32)


    def forward(self,inputdata,dt=None):
        if dt == None:
            dt = self.dt
        dnnin = self.inputranslation(inputdata)
        x = self.dnn(dnnin)[0].detach()
        try:
            self.filter.predict(dt) # is this step needed?
        except:
            # print("Failed prediction - covariance possibly inaccurate")
            self.filter.P = np.diag([1e3,1e3,1e3,1e1,1e1,1e1])
        self.filter.x = np.array(np.matmul(x, self.forwardConversionMatrix))
        # z = np.append(np.append(inputdata[1:3],self.rangetranslation(r,direction=1)),inputdata[3:5])
        z = np.array(inputdata[1:5])
        # z = inputdata
        # z[2] = self.rangetranslation(r,direction=1)
        self.filter.update(z)
        self.filter.x = np.array(np.matmul(x, self.forwardConversionMatrix)) # Does this need to be done here also?
        return self.filter.x


    def outputTranslation(self,outputdata):
        output = []
        for entry in outputdata:
            output.append(np.matmul(entry, self.forwardConversionMatrix))
        return np.array(output)


    def nn_only(self,inputdata,dt=None):
        if dt == None:
            dt = self.dt
        dnnin = self.inputranslation(inputdata)
        r = self.dnn(dnnin)[0].detach().numpy()
        return np.matmul(r, self.forwardConversionMatrix)


    def trainstep(self,inputdata,inputSV,dt=None,usePredict=1):
        if dt == None:
            dt = self.dt
        # P0 = np.copy(self.filter.P)
        dnnin = self.inputranslation(inputdata)
        # self.forward(inputdata, dt=dt)
        xpred = self.dnn(dnnin)[0]
        z = np.copy(inputdata[0:5])
        xerr = self.loss_fn(xpred, torchNN.torch.matmul(torchNN.torch.tensor(inputSV,dtype=torchNN.torch.float32), self.backwardConversionMatrix))
        self.filter.x = np.array(np.matmul(xpred.detach().numpy(), self.forwardConversionMatrix))
        if usePredict:
            try:
                self.filter.predict(dt) # is this step needed?
            except: # x gets reset every time, this simply ensures it isn't nan
                self.filter.x = np.array(np.matmul(xpred.detach().numpy(), self.forwardConversionMatrix))
        self.filter.update(z[1:])
        P1 = self.filter.P
        # Perr = self.loss_fn(torchNN.torch.tensor([np.linalg.norm(P1)*self.covtuning]),torchNN.torch.tensor([0.])) # want to drive the coveriance of the system to zero - but might need an offset to avoid model not fitting
        Zinput = self.filter.y
        # Zinput[2] = self.rangetranslation(Zinput[2],0) # to prevent range error exploding z error
        Pinput = torchNN.torch.matmul(torchNN.torch.tensor(np.diag(self.filter.P),dtype=torchNN.torch.float32),self.backwardConversionMatrix) * self.covtuning
        Perr = self.loss_fn(Pinput,torchNN.torch.zeros_like(Pinput))
        Z1 = np.tan(Zinput * .5*np.pi/180)
        # Z1[0] = np.tan(Z1[0]);Z1[1] = np.tan(Z1[1]);Z1[3] = np.tan(Z1[3]);Z1[4] = np.tan(Z1[4])
        Zs = torchNN.torch.tensor(Z1) #np.linalg.norm(Z1))
        Zerr = self.loss_fn(Zs,torchNN.torch.zeros_like(Zs)) * self.ztuning
        # loss = rerr + Perr
        (xerr.type(torchNN.torch.float32) + Perr.type(torchNN.torch.float32) + Zerr.type(torchNN.torch.float32)).backward()
        self.optimizer.step()


    def trainBatch(self,inputdata,inputSV,dt=None):
        if dt == None:
            dt = self.dt
        # P0 = np.copy(self.filter.P)
        Zsaves = []
        inputs = []
        for ii in range(len(inputdata)):
            dnnin = self.inputranslation(inputdata[ii])
            if np.sum(self.filter.z == None) > 0:
                self.filter.z = np.zeros(self.filter.z.shape)
            inputs.append(dnnin)
            xpred = self.dnn(inputs[-1])[0]
            self.filter.x = np.matmul(xpred.detach().numpy(),self.forwardConversionMatrix)
            self.filter.predict(dt) # can have prediction occur here...
            # self.forward(inputdata,dt=dt)
            # self.filter.predict(dt) # ...or here.
            # rerr = self.loss_fn(rpred, inputrange[ii])
            z = inputdata[ii][0:5]
            self.filter.update(z[1:5])
            Zinput = np.copy(self.filter.y)
            # Zs = torchNN.torch.tensor(Z1)#np.linalg.norm(Z1))
            Zsaves.append(Zinput)
        Zsaves = torchNN.torch.tensor(Zsaves,dtype=torchNN.torch.float32)
        Zerr = self.loss_fn(Zsaves,torchNN.torch.zeros_like(Zsaves,dtype=torchNN.torch.float32)) * self.ztuning
        svpreds = self.dnn(np.array(inputs))
        rerr = self.loss_fn(svpreds, torchNN.torch.tensor(inputSV,dtype=torchNN.torch.float32))
        # loss = rerr + Perr
        (rerr + Zerr).backward()
        self.optimizer.step()


class KFNN_v5():
    # What data does the neural network need to work with?  What does it modify in the kalman filter?
    # Idea 1 - neural network has access to KF inputs and KF covariance (diagonal?), improves result to reduce covariance
    def __init__(self,dim = 6,dimz = 4,dimin = 10,dt=10,dnntype=0,covarianceStrength=1e-3,zerrstrength=1.,usez=True,usey=False):
        self.dt = 10
        self.OD = OrbitDetermination.orbitDetermination()
        self.OF = orbitFunctions.orbitFunctions()
        self.points = sigma_points.MerweScaledSigmaPoints(dim, alpha=.1, beta=2., kappa=-3)
        self.filter = UnscentedKalmanFilter(dim_x=6, dim_z=dimz, dt=dt, fx=self.OF.orbitFunction6, hx=self.OF.orbitSensorModelAORts, points=self.points)
        inputsize = dimin
        if usez:
            inputsize = dimin + dimz + dimz
        self.dnn = torchNN.Dense91(inputsize,dim,(300,300),activation_fc=torchNN.torch.nn.functional.relu)
        if dnntype == 1:
            self.dnn = torchNN.RNN1(inputsize,dim,(300,300),activation_fc=torchNN.torch.nn.functional.relu,rnnsets=1,layersperset=3)
        elif dnntype == 2:
            self.dnn = torchNN.Trnsfrmr1(inputsize,dim,(300,300),activation_fc=torchNN.torch.nn.functional.relu,hiddenlayers=0,encoderlayers=1)
        elif dnntype == 3:
            self.dnn = torchNN.Trnsfrmr2(inputsize,dim,(300,300),activation_fc=torchNN.torch.nn.functional.relu,hiddenlayers=1,encoderlayers=1)
        self.loss_fn = torchNN.torch.nn.MSELoss()
        self.optimizer = torchNN.torch.optim.Adam(self.dnn.parameters(), lr=0.001)
        self.location = neural_filter.locations.Location('UCCS',38.89588,-104.80232,1950)
        self.filter.R = np.diag([0.02,0.02,.02,.02]) # measurement error? 1e2 vs 1e3
        self.filter.P = np.diag([1e3,1e3,1e3,1e1,1e1,1e1])
        self.filter.x = np.array([1e6,-1e6,0,-7,7,0])
        self.forwardConversionMatrix = np.diag([1e6,1e6,1e6,1e1,1e1,1e1])
        self.backwardConversionMatrix = torchNN.torch.tensor(np.diag([1e-6,1e-6,1e-6,1e-1,1e-1,1e-1]),dtype=torchNN.torch.float32)
        self.covtuning = covarianceStrength
        self.ztuning = zerrstrength
        self.usez = usez
        self.usey = usey


    def saveNN(self,filename='models/KFNN_v5_dnn'):
        best_weights = copy.deepcopy(self.dnn.state_dict())
        torchNN.torch.save(best_weights,filename)


    def loadNN(self,filename='models/KFNN_v5_dnn'):
        self.dnn.load_state_dict(torchNN.torch.load(filename,weights_only=True))


    def setStartTime(self,time):
        self.OF.time = time


    def inputranslation(self,input):
        output = input
        d2r = np.pi/180
        if self.dnn.input_layer.in_features == 4:
            output = np.array(input * d2r).astype(float)
        elif self.dnn.input_layer.in_features == 5 and len(input) == 5:
            output = np.array(input * d2r).astype(float)
        elif self.dnn.input_layer.in_features == 8 and len(input) == 4:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3])]).astype(float)
        elif self.dnn.input_layer.in_features == 9 and len(input) == 5:
            input = np.array(input) * d2r
            output = np.array([input[0],np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4])]).astype(float)
        elif len(input) == 4:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3])]).astype(float)
        elif len(input) == 5:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4])]).astype(float)
        elif len(input) == 10:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4]),
                               np.sin(input[5]),np.cos(input[5]),np.sin(input[6]),np.cos(input[6]),np.sin(input[7]),np.cos(input[7]),np.sin(input[8]),np.cos(input[8]),np.sin(input[9]),np.cos(input[9])]).astype(float)
        elif len(input) == 15:
            input = np.array(input) * d2r
            output = np.array([np.sin(input[0]),np.cos(input[0]),np.sin(input[1]),np.cos(input[1]),np.sin(input[2]),np.cos(input[2]),np.sin(input[3]),np.cos(input[3]),np.sin(input[4]),np.cos(input[4]),
                               np.sin(input[5]),np.cos(input[5]),np.sin(input[6]),np.cos(input[6]),np.sin(input[7]),np.cos(input[7]),np.sin(input[8]),np.cos(input[8]),np.sin(input[9]),np.cos(input[9]),
                               np.sin(input[10]),np.cos(input[10]),np.sin(input[11]),np.cos(input[11]),np.sin(input[12]),np.cos(input[12]),np.sin(input[13]),np.cos(input[13]),np.sin(input[14]),np.cos(input[14])]).astype(float)
        return torchNN.torch.tensor(output.astype(np.float32),dtype=torchNN.torch.float32)


    def horizon_to_az_elev(self,top_s, top_e, top_z):
        range_sat = torchNN.torch.sqrt((top_s * top_s) + (top_e * top_e) + (top_z * top_z))
        elevation = torchNN.torch.asin(top_z / range_sat)
        azimuth = torchNN.torch.atan2(-top_e, top_s) + torchNN.torch.pi
        return azimuth, elevation


    def orbitFunction6(self,xh,dt):#,u,xw,dt): # 6-variable version of the orbital dynamics model
        # dt = self.dt
        r = torchNN.torch.sqrt(xh[0]**2 + xh[1]**2 + xh[2]**2)
        cr = -orbitFunctions.constants.MU_E/r**3
        A1 = [0,0,0,1,0,0]
        A2 = [0,0,0,0,1,0]
        A3 = [0,0,0,0,0,1]
        A4 = [cr,0,0,0,0,0]
        A5 = [0,cr,0,0,0,0]
        A6 = [0,0,cr,0,0,0]
        A = torchNN.torch.tensor([A1,A2,A3,A4,A5,A6])
        xh1 = torchNN.torch.matmul(A,xh)*dt + xh #+ xw
        return xh1


    def orbitSensorModelAORts(self,xh):#,zh):
        if(xh.ndim>1):
            xh = xh.flatten()
        time = self.OF.time # zh[0] # need to ensure this is correct
        t = orbitFunctions.utils.gstime_from_datetime(self.OF.Jtime2Datetime(time))
        ecef = orbitFunctions.coordinate_systems.eci_to_ecef(xh[0:3],t)
        horizon = orbitFunctions.coordinate_systems.to_horizon(self.OF.Sensor.latitude_rad,self.OF.Sensor.longitude_rad,self.OF.Sensor.position_ecef,ecef)
        azel = self.horizon_to_az_elev(horizon[0],horizon[1],horizon[2])
        azel = [azel[0]*180/np.pi, azel[1]*180/np.pi]
        dt = 1 # timedelta in seconds for rate calculations
        xh1 = self.orbitFunction6(xh,dt)
        t1 = (t + dt/86400) % (np.pi * 2)
        ecef1 = orbitFunctions.coordinate_systems.eci_to_ecef(xh1[0:3],t1)
        horizon1 = orbitFunctions.coordinate_systems.to_horizon(self.OF.Sensor.latitude_rad,self.OF.Sensor.longitude_rad,self.OF.Sensor.position_ecef,ecef1)
        azel1 = self.horizon_to_az_elev(horizon1[0],horizon1[1],horizon1[2])
        azelrate = [azel1[0]*180/np.pi - azel[0], azel1[1]*180/np.pi - azel[1]]
        observation = [azel[0],azel[1],azelrate[0],azelrate[1]] # torchNN.torch.hstack([azel[0],azel[1],azelrate[0],azelrate[1]])
        return observation


    def rangetranslation(self,range,direction=1,mag=72000):
        output = range
        if direction == 0: # forward
            output = range / mag
        elif direction == 1: # backward
            output = range * mag
        return torchNN.torch.tensor([output],dtype=torchNN.torch.float32)


    def nn_only(self,inputdata,dt=None,Z=None):
        if dt == None:
            dt = self.dt
        dnnin = self.inputranslation(inputdata)
        inputd = dnnin
        if self.usez:
            znnin = Z
            if np.sum(self.filter.z == None) > 0:
                self.filter.z = np.zeros(self.filter.z.shape)
            if Z == None:
                znnin = self.inputranslation(self.filter.z.flatten()*np.pi/180) # what is filter.y and filter.z?
                if self.usey:
                    znnin = self.inputranslation(self.filter.y.flatten()*np.pi/180) # what is filter.y and filter.z?
            inputd = np.concatenate([dnnin,znnin])
        oe = self.dnn(inputd)[0].detach().numpy()
        return self.OEtranslation(oe,direction=0)


    def OEtranslation(self,OE,direction=1):
        OEnew = []
        if direction == 1:
            d2r = np.pi/180
            OEnew = np.copy([OE[0]/10,OE[1],OE[2]*d2r,OE[3]*d2r,OE[4]*d2r,OE[5]*d2r])
        else:
            r2d = 180/np.pi
            OEnew = np.copy([OE[0]*10,OE[1],OE[2]*r2d,OE[3]*r2d,OE[4]*r2d,OE[5]*r2d])
        return OEnew


    def forward(self,inputdata,dt=None):
        if dt == None:
            dt = self.dt
        dnnin = self.inputranslation(inputdata)
        inputd = dnnin
        # self.forward(inputdata, dt=dt)
        if self.usez:
            if np.sum(self.filter.z == None) > 0:
                self.filter.z = np.zeros(self.filter.z.shape)
            znnin = self.inputranslation(self.filter.z.flatten()*np.pi/180) # what is filter.y and filter.z?
            if self.usey:
                znnin = self.inputranslation(self.filter.y.flatten()*np.pi/180) # what is filter.y and filter.z?
            inputd = np.concatenate([dnnin,znnin])
        oe = self.dnn(inputd)[0].detach()
        x = self.OD.OrbitElement2StateVector4(self.OEtranslation(oe,direction=0)).flatten()
        try:
            self.filter.predict(dt) # is this step needed?
        except:
            # print("Failed prediction - covariance possibly inaccurate")
            self.filter.P = np.diag([1e3,1e3,1e3,1e1,1e1,1e1])
        self.filter.x = np.copy(x)
        # z = np.append(np.append(inputdata[1:3],self.rangetranslation(r,direction=1)),inputdata[3:5])
        z = np.array(inputdata[1:5])
        # z = inputdata
        # z[2] = self.rangetranslation(r,direction=1)
        self.filter.update(z)
        self.filter.x = x # Does this need to be done here also?
        return self.filter.x


    def trainstep(self,inputdata,inputOE,dt=None):
        if dt == None:
            dt = self.dt
        # P0 = np.copy(self.filter.P)
        dnnin = self.inputranslation(inputdata)
        # self.forward(inputdata, dt=dt)
        inputd = dnnin
        # self.forward(inputdata, dt=dt)
        if self.usez:
            if np.sum(self.filter.z == None) > 0:
                self.filter.z = np.zeros(self.filter.z.shape)
            znnin = self.inputranslation(self.filter.z.flatten()*np.pi/180) # what is filter.y and filter.z?
            if self.usey:
                znnin = self.inputranslation(self.filter.y.flatten()*np.pi/180) # what is filter.y and filter.z?
            inputd = np.concatenate([dnnin,znnin])
        oepred = self.dnn(inputd)[0]
        z = np.copy(inputdata[0:5])
        oeerr = self.loss_fn(oepred, torchNN.torch.tensor(self.OEtranslation(inputOE.flatten()),dtype=torchNN.torch.float32))
        x = self.OD.OrbitElement2StateVector4(self.OEtranslation(oepred.detach().numpy(),direction=0)).flatten()
        self.filter.x = x
        try:
            self.filter.predict(dt) # is this step needed?
        except: # x gets reset every time, this simply ensures it isn't nan
            self.filter.x = x
        self.filter.update(z[1:])
        P1 = self.filter.P
        # Perr = self.loss_fn(torchNN.torch.tensor([np.linalg.norm(P1)*self.covtuning]),torchNN.torch.tensor([0.])) # want to drive the coveriance of the system to zero - but might need an offset to avoid model not fitting
        Zinput = self.filter.y
        # Zinput[2] = self.rangetranslation(Zinput[2],0) # to prevent range error exploding z error
        Pinput = torchNN.torch.matmul(torchNN.torch.tensor(np.diag(self.filter.P),dtype=torchNN.torch.float32),self.backwardConversionMatrix) * self.covtuning
        Perr = self.loss_fn(Pinput,torchNN.torch.zeros_like(Pinput))
        Z1 = np.tan(Zinput * .5*np.pi/180)
        # Z1[0] = np.tan(Z1[0]);Z1[1] = np.tan(Z1[1]);Z1[3] = np.tan(Z1[3]);Z1[4] = np.tan(Z1[4])
        Zs = torchNN.torch.tensor(Z1) #np.linalg.norm(Z1))
        Zerr = self.loss_fn(Zs,torchNN.torch.zeros_like(Zs)) * self.ztuning
        # loss = rerr + Perr
        (oeerr.type(torchNN.torch.float32) + Perr.type(torchNN.torch.float32) + Zerr.type(torchNN.torch.float32)).backward()
        self.optimizer.step()


    def trainBatch(self,inputdata,inputOE,dt=None):
        if dt == None:
            dt = self.dt
        # P0 = np.copy(self.filter.P)
        oepreds = []
        OEs = []
        Zsave = []
        inputs = []
        for ii in range(len(inputdata)):
            dnnin = self.inputranslation(inputdata[ii])
            inputd = dnnin
            # self.forward(inputdata, dt=dt)
            if self.usez:
                if np.sum(self.filter.z == None) > 0:
                    self.filter.z = np.zeros(self.filter.z.shape)
                znnin = self.inputranslation(self.filter.z.flatten()*np.pi/180) # what is filter.y and filter.z?
                if self.usey:
                    znnin = self.inputranslation(self.filter.y.flatten()*np.pi/180) # what is filter.y and filter.z?
                inputd = np.concatenate([dnnin,znnin])
            inputs.append(inputd)
            oepred = self.dnn(inputs[-1])[0]
            oepreds.append(oepred)
            x = self.OD.OrbitElement2StateVector4(self.OEtranslation(oepred.detach().numpy(),direction=0)).flatten()
            OEs.append(self.OEtranslation(inputOE[ii].flatten()))
            z = np.array(inputdata[ii][0:5])
            self.filter.x = x
            try:
                self.filter.predict(dt) # is this step needed?
            except: # x gets reset every time, this simply ensures it isn't nan
                self.filter.x = x
            self.filter.update(z[1:5])
            P1 = self.filter.P
            Zinput = self.filter.y
            # a = torchNN.torch.randn(size=([6]), requires_grad=True)
            # d = self.orbitSensorModelAORts(a)
            # d.backward()
            # a.grad
            # Zinput[2] = self.rangetranslation(Zinput[2],0) # to prevent range error exploding z error
            Z1 = np.tan(Zinput * .5*np.pi/180)
            # Z1[0] = np.tan(Z1[0]);Z1[1] = np.tan(Z1[1]);Z1[3] = np.tan(Z1[3]);Z1[4] = np.tan(Z1[4])
            # Zs = torchNN.torch.tensor(Z1)#np.linalg.norm(Z1))
            Zsave.append(Z1)
        oepreds = self.dnn(np.array(inputs))
        Zsave = torchNN.torch.tensor(np.array(Zsave),dtype=torchNN.torch.float32)
        Zerr = self.loss_fn(np.tan(.5*Zsave),torchNN.torch.zeros_like(Zsave)) * self.ztuning # tangent or sine?
        xerr = self.loss_fn(oepreds, torchNN.torch.tensor(np.array(OEs),dtype=torchNN.torch.float32))
        # loss = rerr + Perr
        (xerr.type(torchNN.torch.float32) + Zerr.type(torchNN.torch.float32)).backward()
        self.optimizer.step()

