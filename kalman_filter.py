
import numpy as np
import pandas as pd
from orbit_predictor import constants
from orbit_predictor import locations
from orbit_predictor import utils
from orbit_predictor import coordinate_systems
from scipy import linalg
import datetime
import math

import orbitFunctions


class OrbitKalmanFilter(orbitFunctions.orbitFunctions):
    xh = np.array([]) # predicted X vector
    SigmaX = np.array([]) # Model error prediction
    SigmaW = np.array([]) # Forcing error
    SigmaV = np.array([]) # Sensor error
    A = np.array([]) # Model A matrix (physics)
    B = np.array([]) # Model B matrix (forcing)
    C = np.array([]) # Model C matrix (measurement)
    D = np.array([]) # Model D matrix (hysteresis)
    S = np.array([])
    Sensor = locations.Location('UCCS',38.89588,-104.80232,1950)


    # def orbitModel(self,xh): # 
    #     r = np.sqrt(xh[3]**2 + xh[4]**2 + xh[5]**2)
    #     cr = -constants.MU_E/r**3
    #     A7 = [cr,0,0,0,0,0,0,0,0]
    #     A8 = [0,cr,0,0,0,0,0,0,0]
    #     A9 = [0,0,cr,0,0,0,0,0,0]
    #     A1 = [0,0,0,1,0,0,0,0,0]
    #     A2 = [0,0,0,0,1,0,0,0,0]
    #     A3 = [0,0,0,0,0,1,0,0,0]
    #     A4 = [0,0,0,0,0,0,1,0,0]
    #     A5 = [0,0,0,0,0,0,0,1,0]
    #     A6 = [0,0,0,0,0,0,0,0,1]
    #     self.A = np.array([A1,A2,A3,A4,A5,A6,A7,A8,A9])
    #     self.B = np.array([0,0,0,0,0,0,0,0,0])
    #     self.C = np.array([])
    #     #self.D = 


    def createChi(self,x,Sigma,gamma=np.sqrt(3)):
        try:
            Csigma = np.linalg.cholesky(Sigma)
        except:
            print('Treating positive-definite Sigma')
            Sigma = np.diag(np.diag(Sigma)) + np.diag(np.ones(len(Sigma)))
            try:
                Csigma = np.linalg.cholesky(Sigma)
            except:
                Sigma = np.diag(abs(np.diag(Sigma))) + np.diag(np.ones(len(Sigma))*np.pi)
                Csigma = np.linalg.cholesky(Sigma)
        E = np.eye(len(x))
        X = np.column_stack([x, x + gamma*E*Csigma, x - gamma*E*Csigma])
        return np.transpose(X)


    def createAlpha(self,x,h = np.sqrt(6)):
        alpha = np.zeros([len(x)*2 + 1, 1]) + 1/(2*h**2)
        alpha[0] = (h**2-2*len(x))/(h**2)
        #alpha[0] = (h**2-len(x))/(h**2) # not quite correct, but let's try it...
        return alpha


    def simpleFilter(self,ztrue,xh,u=0,alpha=None,dt=60):
        # Step 1a
        if len(xh.shape) == 1:
            xh = np.transpose([xh])
        x = np.concatenate((xh,np.zeros([len(self.SigmaW),1]))).astype('d')
        x = np.concatenate((x,np.transpose([ztrue]))).astype('d') # what am I doing here? ztrue*0 [1:]
        # if(type(alpha) == type(None)):
        #     alpha = self.createAlpha(x)
        # alpham = alpha
        # alphac = alpha
        # Sigma = linalg.block_diag(self.SigmaX,self.SigmaW,self.SigmaV)
        # Chi = np.transpose(self.createChi(x,Sigma)) # sigma points - not sure why Chi is the normal notation for this
        # ChiX = np.transpose(Chi[0:len(xh)])
        # ChiW = Chi[len(xh):len(xh)+len(self.SigmaW)]
        # ChiV = Chi[len(xh)+len(self.SigmaW):]
        # #Chi1 = np.transpose(Chi)
        # ii = 0
        # Xxkm = []
        # while(ii<len(Chi[0])):
        #     Ah = self.createOrbitAhat(ChiX[ii],dt)
        #     Xxkm.append(np.matmul(Ah,xh) + np.transpose([ChiX[ii]]))
        #     #self.A*ChiX[ii] + self.B*ChiW[ii]
        #     ii+=1
        # #ChiX = np.transpose(ChiX)
        # xkm = np.matmul(np.transpose(ChiX),alpham).astype('d')# + xh # need to verify equations here are right
        if len(xh) == 9:
            xkm = self.orbitFunction(xh,u,0,dt)
        else:
            xkm = self.orbitFunction6(xh,u,0,dt)
        # step 1b
        # ii = 0
        # SigmaX = np.zeros([len(self.SigmaX),len(self.SigmaX)])
        # while(ii<len(alphac)):
        #     ChiXxh = np.transpose([ChiX[ii]]) - xh
        #     SigmaX += alphac[ii] * np.matmul(ChiXxh,np.transpose(ChiXxh)).astype('d')
        #     ii+=1
        # d = np.diag(SigmaX)
        # ii=0
        # while(ii<len(d)):
        #     if(d[ii] == 0):
        #         #print('Correcting Sigma 0')
        #         SigmaX[ii][ii] = 1 # some reasonable error to prevent zeroing of error and subsequent crash
        #     ii+=1
        SigmaX = np.matmul(self.A,self.SigmaX)
        SigmaX = np.matmul(SigmaX,np.transpose(self.A))
        self.SigmaX = SigmaX
        # step 1c
        # ii = 0
        # Z = []
        # while(ii<len(Chi[0])):
        #     Z.append(self.orbitSensorModel(Xxkm[ii],ztrue))
        #     ii+=1
        # z = np.matmul(np.transpose(Z),alpham)
        z = self.orbitSensorModelAO(x,ztrue)
        # step 2a
        # Zzk = np.zeros([len(z),len(z)])
        # Zxzk = np.zeros([len(xh),len(z)])
        # ii=0
        # while(ii<len(alphac)):
        #     ChiXxh = np.transpose([ChiX[ii]]) - xh
        #     ChiZz = np.transpose([Z[ii]]) - z
        #     Zzk += alphac[ii]*np.matmul(ChiZz,np.transpose(ChiZz))
        #     Zxzk += alphac[ii]*np.matmul(ChiXxh,np.transpose(ChiZz))
        #     ii+=1
        # Sigmazk = Zzk #(Z-z)*np.transpose(Z-z)*alphac
        # Sigmaxzk = Zxzk #(Xxkm-xkm)*np.transpose(Z-z)*alphac
        L1 = np.matmul(self.C,SigmaX)
        L1 = np.matmul(L1,np.transpose(self.C))
        d = np.linalg.det(L1)
        if(d!=0): # Sigmazk is non-determinable, do not update SigmaX
            # Lk = np.matmul(Sigmaxzk,np.linalg.inv(Sigmazk))
            Lk = np.matmul(SigmaX,np.transpose(self.C))
            Lk = np.matmul(Lk,np.invert(L1))
            # step 2b
            # dz = np.transpose([ztrue[1:]]) - z
            xh = xkm + np.matmul(Lk,ztrue - z).astype('d')
            # step 2c
            self.SigmaX = SigmaX - np.matmul(np.matmul(Lk,self.C),SigmaX)
            # d = np.diag(SigmaX)
            # ii = 0
            # while(ii<len(d)):
            #     if(d[ii] == 0):
            #         print('Problematix SigmaX')
            #         self.SigmaX[ii][ii] = 1e-6
            #     ii+=1
        else:
            print('Singular Sigma Encountered, predicted state:',np.concatenate(xh),' sensor input:',ztrue)
        return [xh,self.SigmaX]


    def sigmaFilter(self,ztrue,xh,u = 0,alpha=None,dt = 60): # update step for orbital kalman filter
        # Kalman Filter implementation specifically for orbits, based on https://www.mdpi.com/2411-9660/5/3/54
        # utilizes a Sigma-Point Kalman Filter
        # 
        # Step 1a
        if len(xh.shape) == 1:
            xh = np.transpose([xh])
        x = np.concatenate((xh,np.zeros([len(self.SigmaW),1]))).astype('d')
        x = np.concatenate((x,np.transpose([ztrue[1:]]))).astype('d') # what am I doing here? ztrue*0 [1:] - removing datetime element?
        if(type(alpha) == type(None)):
            alpha = self.createAlpha(x)
        alpham = alpha
        alphac = np.copy(alpha)
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
                    print('Problematix SigmaX')
                    self.SigmaX[ii][ii] = 1e-6
                ii+=1
        else:
            print('Singular Sigma Encountered, predicted state:',np.concatenate(xh),' sensor input:',ztrue)
        return [xh,self.SigmaX]


    def sigmaFilter2(self,ztrue,xh,u = 0,alpha=None,dt = 60): # second variation of orbital filter update step
        # Kalman Filter implementation specifically for orbits, based on https://www.mdpi.com/2411-9660/5/3/54
        # utilizes a Sigma-Point Kalman Filter
        # 
        # Step 1a
        x = np.concatenate((xh,np.zeros([len(self.SigmaW),1]))).astype('d')
        x = np.concatenate((x,np.transpose([ztrue]))).astype('d') # what am I doing here? ztrue*0
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
            #Ah = self.createOrbitAhat(ChiX[ii],dt)
            #Xxkm.append(np.matmul(Ah,xh) + np.transpose([ChiX[ii]]))
            Xxkm.append(self.orbitFunction(ChiX[ii],u,ChiW[ii],dt))
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
                SigmaX[ii][ii] = 1 # some value to prevent zeroing of error and subsequent crash
            ii+=1
        self.SigmaX = SigmaX
        # step 1c
        ii = 0
        Z = []
        while(ii<len(Chi[0])):
            Z.append(self.orbitSensorModel(Xxkm[ii],ztrue))
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
            dz = np.transpose([ztrue]) - z
            xh = xkm + np.matmul(Lk,dz).astype('d')
            # step 2c
            self.SigmaX = SigmaX - np.matmul(np.matmul(Lk,Sigmazk),np.transpose(Lk))
            d = np.diag(SigmaX)
            ii = 0
            while(ii<len(d)):
                if(d[ii] == 0):
                    SigmaX[ii][ii] = 1e-6
                ii+=1
        else:
            print('Singular Sigma Encountered, predicted state:',np.concatenate(xh),' sensor input:',ztrue)
        return [xh,self.SigmaX]


    def sigmaFilterAO(self,ztrue,xh,u = 0,alpha=None,dt = 60): # angles-only update version
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


    def sigmaFilterSR(self,ztrue,xh,w = [1],v = [1],u = 0,alpha=None):
        if(alpha == None):
            alpha = self.createAlpha(xh)
        xhapk_1 = np.concatenate(np.transpose(xh),np.transpose(w),np.transpose(v))
        Sigma = linalg.block_diag(self.SigmaX,self.SigmaW,self.SigmaV)
        Chi = self.createChi(self.xh,Sigma) # self.createChi(x,Sigma) # which is correct?
        return [xh,self.SigmaX]


    def KalmanFilter(self,ztrue,u,xh,SigmaX,SigmaW,SigmaV): # Kalman filter 
        # Kalman filter implementation
        # Takes as input the kalman filter inputs and returns an estimated state
        xh = self.A*xh + self.B*u; #(1)
        SigmaX = self.A*SigmaX*np.transpose(self.A) + SigmaW; #(2)
        zh = self.C*xh + self.D*u; #(3)
        L = SigmaX*np.transpose(self.C)/(self.C*SigmaX*np.transpose(self.C) + SigmaV); #(4)
        xh = xh + L*(ztrue - zh); #(5)
        SigmaX = SigmaX - L*self.C*SigmaX; #(6)
        return [xh,SigmaX]


    def PropagateKalmanFilter(self,measurements,u,xh,SigmaX,SigmaW,SigmaV):
        ii = 0
        while(ii<len(measurements)):
            ret = self.KalmanFilter(measurements[ii],u,xh,SigmaX,SigmaW,SigmaV)
            xh = ret[0]
            SigmaX = ret[1]
            ii+=1
        return [xh,SigmaX]


    def CreateModel(self,A,B,C,D): # sets up generic Kalman Filter model for sigma-point kalman filter
        # Creates state matrix model given the inputs as matrices.
        # Produces error matrices of the correct dimension for the 
        # state matrix model.
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)
        self.xh = np.zeros(len(A))
        self.SigmaX = np.zeros(len(A),len(A[0])) # need modification to account if A is 1-D
        self.SigmaV = np.zeros(len(C),len(C[0])) # ditto above
        self.SigmaW = np.zeros(len(D),len(D[0])) # ditto above
        u = np.zeros(len(C))


class simpleOrbitFilter():
    x = np.array([1e3,1e3,1e3,1,-1,-1])
    F = np.array([])
    H = np.array([])
    SigmaX = np.array([])
    Q = np.eye(6)
    R = np.eye(3)
    Pk = np.diag([1e3,1e3,1e3,2,2,2])
    Sensor = locations.Location('UCCS',38.89588,-104.80232,1950)
    # def __init__():
    #     mu = utils.MU_E
    #     x = [1e4,1e4,1e4,0,0,0]
    #     ri = x[0]
    #     rj = x[1]
    #     rk = x[2]
    #     r = np.linalg.norm(x)
    #     F1 = [0,0,0,1,0,0]
    #     F2 = [0,0,0,0,1,0]
    #     F3 = [0,0,0,0,0,1]
    #     F4 = [-mu/r**3+3*mu*ri**2/r**5,3*ri*rj/r**5,3*mu*ri*rk/r**5,0,0,0]
    #     F5 = [3*mu*ri*rj/r**5,-mu/r**3+3*mu*rj**2/r**5,3*mu*rj*rk/r**5,0,0,0]
    #     F6 = [3*mu*ri*rk/r**5,3*mu*rj*rk/r**5,-mu/r**3+3*mu*rk**2/r**5,0,0,0]
    #     F = np.array([F1,F2,F3,F4,F5,F6])
    #     Q = np.eye(6)
    #     # Rsez = np.norm()


    def initState(self,x):
        # if x.shape[0] = 6:
        self.x = x


    def updateF(self,zh):
        mu = utils.MU_E
        ri = zh[0]
        rj = zh[1]
        rk = zh[2]
        r = np.linalg.norm(self.x)
        F1 = [0,0,0,1,0,0]
        F2 = [0,0,0,0,1,0]
        F3 = [0,0,0,0,0,1]
        F4 = [-mu/r**3+3*mu*ri**2/r**5,3*ri*rj/r**5,3*mu*ri*rk/r**5,0,0,0]
        F5 = [3*mu*ri*rj/r**5,-mu/r**3+3*mu*rj**2/r**5,3*mu*rj*rk/r**5,0,0,0]
        F6 = [3*mu*ri*rk/r**5,3*mu*rj*rk/r**5,-mu/r**3+3*mu*rk**2/r**5,0,0,0]
        self.F = np.array([F1,F2,F3,F4,F5,F6])
        # return self.F


    def updateH(self,azelrng):
        r = azelrng[2]
        rs = np.sin(azelrng[1])*np.cos(azelrng[0])*azelrng[2] # check that this is correct
        re = np.sin(azelrng[1])*np.sin(azelrng[0])*azelrng[2] # check that this is correct
        rz = np.cos(azelrng[1])*azelrng[2]
        H1 = [rs/r,re/r,rz/r]
        H2 = [-1/(rs**2*(re**2/rs**2+1)),rs/(re**2*(re**2/rs**2+1)),0]
        H3 = [rs*rz/(r**3*np.sqrt(-rz**2/r**2+1)),-re*rz/(r**3*np.sqrt(-rz**2/r**2+1)),rs*rz/(r**3*np.sqrt(-rz**2/r**2+1))]
        dobsdrsez = np.array([H1,H2,H3])
        lat = self.Sensor.latitude_rad
        lon = self.Sensor.longitude_rad
        dsezdeci = [[np.sin(lat)*np.cos(lon),np.sin(lat)*np.sin(lon),-np.cos(lat)],[-np.sin(lon),np.cos(lon),0],[np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]]
        H1 = np.cross(dobsdrsez,dsezdeci) # need to add zeros to pad matrix as needed
        H2 = np.zeros([3,6])
        H2[0:3,0:3] = H1
        self.H = H2
        return self.H


    def stepKalmanFilter(self,zk,dt=1):
        # predict
        self.updateF(zk)
        xk_1 = np.matmul(self.F,self.x)*dt
        I = np.eye(6)+self.F*dt+np.matmul(self.F,self.F)*dt/math.factorial(2)
        dxk = xk_1-self.x # check if this is accurate
        dxk_1 = np.matmul(I,dxk)
        Pk_1 = np.matmul(np.matmul(I,self.Pk),np.transpose(I))+self.Q
        # update
        try:
            Hk_1 = self.updateH(zk)
            bk_1 = zk - np.matmul(Hk_1,xk_1) # is this correct?
            Kk_1 = np.matmul(np.matmul(Pk_1,np.transpose(Hk_1)),np.linalg.inv(np.matmul(np.matmul(Hk_1,Pk_1),np.transpose(Hk_1))+self.R)) # need to check how this is done
            dxk_1 = dxk_1 + np.matmul(Kk_1,(bk_1-np.matmul(Hk_1,dxk_1)))
            xk_1 = xk_1 + dxk_1
            Pk_1 = np.matmul((I-np.matmul(Kk_1,Hk_1)),Pk_1)
            self.Pk = Pk_1
            self.xk = xk_1
        except:
            print("KF update failed")
        return xk_1,Pk_1


    def filter(self,data,timestep = 1):
        storX = []
        storPk = []
        jj = 0
        for jj in range(len(data[0])):
            x,pk = self.stepKalmanFilter(np.array([data[0][jj],data[1][jj],data[2][jj]]),timestep)
            storX.append(x)
            storPk.append(np.diag(pk))
        return storX,storPk

