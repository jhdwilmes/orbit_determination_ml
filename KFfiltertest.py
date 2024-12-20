import numpy as np
from filterpy.kalman import kalman_filter
from filterpy.kalman import sigma_points
from filterpy.kalman import unscented_transform
from filterpy.kalman import UnscentedKalmanFilter

from orbit_predictor import constants
from orbit_predictor import locations
from orbit_predictor import utils
from orbit_predictor import coordinate_systems

# import orbdtools
# from orbdtools import ArcObs
# from orbdtools import Body

import matplotlib as mpl
from matplotlib import pyplot as plt

import datetime

import OrbitDetermination as OD
import neural_filter
import orbitFunctions
import createDataLive
import kalman_filter as KF




LD = createDataLive.TLEdata()
LD.createRandomOrbit(100)
orbdet = OD.orbitDetermination()
dataset = '1'# 'ME_10s'
oblimit = 360
hillSphere = 35786 # distance for geostationary orbit
# rdata,soln,svtruth0 = neural_filter.importDataKF('datasets/tle_dataset_custom_'+dataset+'.csv',obslimit = oblimit,hillSphere=hillSphere)
# rdata,soln = neural_filter.importDataKF('datasets/tle_dataset_custom_'+dataset+'.csv',obslimit = oblimit,hillSphere=hillSphere)
# r = np.random.randint(0,len(rdata))
# ddata = np.transpose(rdata[r])
# zAzEl = np.transpose([360*(ddata[1]+0.5),360*(ddata[2]+0.5),ddata[7],ddata[8]])
# zAzElRng = np.transpose([ddata[1],ddata[2],ddata[3]]) # AzElRange
# zAzEl = np.transpose([ddata[1],ddata[2]]) # AzEl - does Jtime (ddata[0]) need to be included?
# zRaDec = np.transpose([ddata[4],ddata[5]]) # RAdec
# zRaDecRng = np.transpose([ddata[4],ddata[5],ddata[3]]) # RAdec,range (not exact measurement?)
# zh = 0*zAzEl[0]
dt = 10
o,w,t = LD.oneTLEdata(tstep=dt,tryagain=5,noise=0.1)
plt.figure()
plt.plot(np.transpose(w)[0])
plt.plot(np.transpose(w)[1])
plt.plot(np.transpose(w)[2])
plt.title('Orbit Input Positions')
plt.savefig('plots/KFtestInputObs.png')

o1 = np.transpose(o)
# zs = np.transpose([360*(ddata[1]+0.5),360*(ddata[2]+0.5),ddata[7],ddata[8]])
zAzElRng = np.transpose([o1[1],o1[2],o1[3]]) # AzElRange
zAzElRngAERt = np.transpose([o1[1],o1[2],o1[3],o1[8],o1[9]]) # AzElRangeAzElRate
zAzElRngRts = np.transpose([o1[1,0:-1],o1[2,0:-1],o1[3,0:-1],o1[8,0:-1],o1[9,0:-1],(o1[3,1:]-o1[3,0:-1])/dt])
zAzEl = np.transpose([o1[1],o1[2]]) # AzEl - does Jtime (ddata[0]) need to be included?
zRaDec = np.transpose([o1[4],o1[5]]) # RAdec
zRaDecRt = np.transpose([o1[4],o1[5],o1[10],o1[11]]) # RAdec
zRaDec2 = np.transpose([o1[6],o1[7]]) # RAdec
zRaDecRng = np.transpose([o1[6],o1[7],o1[3]]) # RAdec,range (not exact measurement?)

skf = KF.simpleOrbitFilter()

OF = orbitFunctions.orbitFunctions()
dim = 6
points = sigma_points.MerweScaledSigmaPoints(dim, alpha=.01, beta=2., kappa=-3)
pointr = sigma_points.MerweScaledSigmaPoints(dim, alpha=.1, beta=2., kappa=-3)
# dt = np.round((ddata[0][1]-ddata[0][0])*86400)
dt = np.median(o1[12,1:]-o1[12,0:-1]).seconds
print('Timestep (s):',dt)
UKFr = UnscentedKalmanFilter(dim_x=dim, dim_z=3, dt=dt, fx=OF.orbitFunction6, hx=OF.orbitSensorModel, points=pointr)
UKFr2 = UnscentedKalmanFilter(dim_x=dim, dim_z=5, dt=dt, fx=OF.orbitFunction6, hx=OF.orbitSensorModelRt, points=pointr)
UKFr3 = UnscentedKalmanFilter(dim_x=dim, dim_z=6, dt=dt, fx=OF.orbitFunction6, hx=OF.orbitSensorModelRts, points=pointr)
UKF  = UnscentedKalmanFilter(dim_x=dim, dim_z=2, dt=dt, fx=OF.orbitFunction6, hx=OF.orbitSensorModelAO2, points=points)
UKF2  = UnscentedKalmanFilter(dim_x=dim, dim_z=4, dt=dt, fx=OF.orbitFunction6, hx=OF.orbitSensorModelAO2r, points=points)
# UKFr.P *= 0.9
# UKF.P *= 0.9
UKFr.R = np.diag([0.02,0.02,99.9]) # measurement error?
# UKF.R = np.diag([9.5,9.5]) # measurement error?
UKF.R = np.diag([0.1,0.1]) # measurement error?
# UKF2.R = np.diag([9.5,9.5,9.1,9.1]) # measurement error?
UKF2.R = np.diag([0.1,0.1,1.0,1.0]) # measurement error?
UKFr2.R = np.diag([0.02,0.02,99.9,.03,.03]) # measurement error?
UKFr3.R = np.diag([0.02,0.02,99.9,.03,.03,5]) # measurement error?
# UKF.Q = 
uccs = neural_filter.locations.Location('UCCS',38.89588,-104.80232,1950)
# gausoln = orbdet.GaussOrbitDetermination(ddata[0],[ddata[1],ddata[2]],uccs,radecOrazel=1)
gausoln = orbdet.GaussOrbitDetermination(o1[0],[o1[1].astype(float),o1[2].astype(float)],uccs,radecOrazel=1)
gausoln2 = orbdet.GaussOrbitDetermination(o1[0],[o1[4].astype(float),o1[5].astype(float)],uccs,radecOrazel=0)
gausoln3 = orbdet.GaussOrbitDetermination(o1[0],[o1[6].astype(float),o1[7].astype(float)],uccs,radecOrazel=0) # this is the correct way to run the gaussian solution?

# earth = Body.from_name('Earth')
# arc_optical = ArcObs({'t':t,'radec':radec,'xyz_site':xyz_site})
# arc_iod = arc_optical.iod(earth)
# arc_iod.gauss(ellipse_only=False)
# print(arc_iod.df.to_string())


estsoln = [o1[0][1],np.array(w[1][0:3]),np.array(w[1][3:])]

speed = (np.array(w[2])-np.array(w[0]))/((o1[0][2]-o1[0][0])*86400)
impute = np.array([w[1][0],w[1][1],w[1][2],speed[0],speed[1],speed[2]])
orate = OF.orbitSensorModelAOrate(impute,np.append([o1[0][1]],zAzEl[1]))

oe = orbdet.StateVector2OrbitalElements2(gausoln)
oe2 = orbdet.StateVector2OrbitalElements2(gausoln2)
oe3 = orbdet.StateVector2OrbitalElements2(gausoln3) # this is the most accurate one?
oetrue = orbdet.StateVector2OrbitalElements2(estsoln)

UKFr.x = np.append(gausoln3[1],gausoln3[2])
UKF.x = np.append(gausoln3[1],gausoln3[2])
UKF2.x = np.append(gausoln3[1],gausoln3[2])
UKFr2.x = np.append(gausoln3[1],gausoln3[2])
UKFr3.x = np.append(gausoln3[1],gausoln3[2])

xsave = []
x2save = []
xrsave = []
xr2save = []
xr3save = []
Xsave = []
X2save = []
Xrsave = []
Xr2save = []
Xr3save = []
# Psave = []
# P2save = []
# Prsave = []

skf.initState(np.array([gausoln[1],gausoln[2]]).flatten())
x,pk=skf.filter(np.transpose(zRaDecRng),60)

print("State vector predictions:")
print("Truth:",estsoln)
print("Gaussian Alt-Az:",gausoln)
print("Gaussian sensor RA-Dec:",gausoln2)
print("Gaussian true RA-Dec:",gausoln3)
print('altitude and speed imputed:',np.linalg.norm(estsoln[1]),np.linalg.norm(estsoln[2]))
print('altitude and speed predicted:',np.linalg.norm(gausoln[1]),np.linalg.norm(gausoln[2]))
print('altitude and speed predicted:',np.linalg.norm(gausoln2[1]),np.linalg.norm(gausoln2[2]))
print('altitude and speed predicted:',np.linalg.norm(gausoln3[1]),np.linalg.norm(gausoln3[2]))
print('Orbit data - mean motion, eccentricity, inclination, right ascension of the ascending node, argument of perigee, true anomaly')
print('Gaussian orbit prediction:',86400/oe[0],oe[1],oe[2]*180/np.pi,oe[3]*180/np.pi,oe[4]*180/np.pi,oe[5]*180/np.pi)
print('Gaussian orbit prediction:',86400/oe2[0],oe2[1],oe2[2]*180/np.pi,oe2[3]*180/np.pi,oe2[4]*180/np.pi,oe2[5]*180/np.pi)
print('Gaussian orbit prediction:',86400/oe3[0],oe3[1],oe3[2]*180/np.pi,oe3[3]*180/np.pi,oe3[4]*180/np.pi,oe3[5]*180/np.pi) # the most accurate one?
print('\"Truth\" OE:',86400/oetrue[0],oetrue[1],oetrue[2]*180/np.pi,oetrue[3]*180/np.pi,oetrue[4]*180/np.pi,oetrue[5]*180/np.pi)

maxiter = np.min([len(zAzEl)-1,100])
ii = 0
for z in zAzEl[0:maxiter]:
    # OF.time = ddata[0][ii]
    OF.time = o1[0][ii]
    UKF.predict(dt=dt)
    UKF.update(z)
    UKF2.predict(dt=dt)
    UKF2.update(zRaDecRt[ii])
    UKFr2.predict(dt=dt)
    UKFr2.update(zAzElRngAERt[ii])
    UKFr3.predict(dt=dt)
    UKFr3.update(zAzElRngRts[ii])
    """
    idea - use a neural network reading the KF input data and covariance, and outputing range.  NN attempts to minimive KF covariance (error)
    second idea - compare results to PINN converting input data into orbit/range/etc.
    Likely need either PINN or reinforcement learning to make this work.
    """
    UKFr.predict(dt=dt)
    UKFr.update(zAzElRng[ii])
    # print(UKF.x, 'log-likelihood', UKF.log_likelihood)
    xsave.append(UKF.x)
    x2save.append(UKF2.x)
    xrsave.append(UKFr.x)
    xr2save.append(UKFr2.x)
    Xsave.append(np.diag(UKF.P))
    X2save.append(np.diag(UKF2.P))
    # Psave.append(UKF.P.flatten())
    # P2save.append(UKF2.P.flatten())
    # Prsave.append(UKFr.P.flatten())
    Xrsave.append(np.diag(UKFr.P))
    Xr2save.append(np.diag(UKFr2.P))
    xr3save.append(UKFr3.x)
    Xr3save.append(np.diag(UKFr3.P))
    ii+=1

# print('Timestep (s):',dt)

minr = 0
maxr = np.min([len(xsave),50])

xsave1 = np.transpose(xsave[minr:maxr])
Xsave1 = np.transpose(Xsave[minr:maxr])
x2save1 = np.transpose(x2save[minr:maxr])
X2save1 = np.transpose(X2save[minr:maxr])
xrsave1 = np.transpose(xrsave[minr:maxr])
Xrsave1 = np.transpose(Xrsave[minr:maxr])
xr2save1 = np.transpose(xr2save[minr:maxr])
Xr2save1 = np.transpose(Xr2save[minr:maxr])
xr3save1 = np.transpose(xr3save[minr:maxr])
Xr3save1 = np.transpose(Xr3save[minr:maxr])

x1 = np.transpose(x[minr:maxr])
w1 = np.transpose(w[minr:maxr])
pk1 = np.transpose(pk[minr:maxr])

plt.figure()
# plt.plot(xsave1[0],xsave1[1])
# plt.plot(xsave1[0],xsave1[2])
# plt.plot(xsave1[1],xsave1[2])
p=plt.subplot(3,1,1)
plt.plot(xsave1[0],'-b')
plt.plot(xsave1[1],'-g')
plt.plot(xsave1[2],'-r')
plt.plot(x2save1[0],'--b')
plt.plot(x2save1[1],'--g')
plt.plot(x2save1[2],'--r')
plt.plot(w1[0],':b.',alpha=0.6)
plt.plot(w1[1],':g.',alpha=0.6)
plt.plot(w1[2],':r.',alpha=0.6)
plt.title('Position Prediction')
p=plt.subplot(3,1,2)
plt.plot(xrsave1[0],'-b')
plt.plot(xrsave1[1],'-g')
plt.plot(xrsave1[2],'-r')
plt.plot(xr2save1[0],'--b')
plt.plot(xr2save1[1],'--g')
plt.plot(xr2save1[2],'--r')
plt.plot(xr3save1[0],'-.b')
plt.plot(xr3save1[1],'-.g')
plt.plot(xr3save1[2],'-.r')
# plt.plot(x1[0],'-.b',alpha=0.5)
# plt.plot(x1[1],'-.g',alpha=0.5)
# plt.plot(x1[2],'-.r',alpha=0.5)
plt.plot(w1[0],':b.',alpha=0.6)
plt.plot(w1[1],':g.',alpha=0.6)
plt.plot(w1[2],':r.',alpha=0.6)
# p.set_ylim(np.min(xrsave1),np.max(xrsave1))
p=plt.subplot(3,1,3)
plt.plot(x1[0],'-b')
plt.plot(x1[1],'-g')
plt.plot(x1[2],'-r')
plt.plot(w1[0],':b.',alpha=0.6)
plt.plot(w1[1],':g.',alpha=0.6)
plt.plot(w1[2],':r.',alpha=0.6)
plt.savefig('plots/KalmanTestPredictedPositions.png')

plt.figure()
p=plt.subplot(3,1,1)
plt.plot(xsave1[3],'-b')
plt.plot(xsave1[4],'-g')
plt.plot(xsave1[5],'-r')
plt.plot(x2save1[3],'--b')
plt.plot(x2save1[4],'--g')
plt.plot(x2save1[5],'--r')
plt.title('Velocity Prediction')
p=plt.subplot(3,1,2)
plt.plot(xrsave1[3],'-b')
plt.plot(xrsave1[4],'-g')
plt.plot(xrsave1[5],'-r')
plt.plot(xr2save1[3],'--b')
plt.plot(xr2save1[4],'--g')
plt.plot(xr2save1[5],'--r')
plt.plot(xr3save1[3],'-.b')
plt.plot(xr3save1[4],'-.g')
plt.plot(xr3save1[5],'-.r')
p=plt.subplot(3,1,3)
plt.plot(x1[3],'-b')
plt.plot(x1[4],'-g')
plt.plot(x1[5],'-r')
plt.savefig('plots/KalmanTestPredictedVelocity.png')

# R1 = Xrsave[-1]
oe1x = [o1[0][0],xrsave[0][0:3],xrsave[0][3:]]
o1e0 = orbdet.StateVector2OrbitalElements2(oe1x)
o1e0[0] = 86400/o1e0[0]
for ii in range(1,np.min([len(Xrsave)-1,50])):
    oe1x = [o1[0][ii],xrsave[ii][0:3],xrsave[ii][3:]]
    o1e = orbdet.StateVector2OrbitalElements2(oe1x)
    o1e[0] = 86400/o1e[0]
    # if np.linalg.norm(Xrsave[ii] - Xrsave[ii+1]) <= np.min([np.linalg.norm(xrsave[ii])*0.001,5]):
    #     print("KF Converged")
    #     break
    if abs(o1e[0] - o1e0[0]) < 0.1 and abs(o1e[1] - o1e0[1]) < 0.01 and abs(o1e[2] - o1e0[2]) < 0.5:
        break
    o1e0 = o1e
oe1x = [o1[0][ii],xr2save[0][0:3],xr2save[0][3:]]
o1e0 = orbdet.StateVector2OrbitalElements2(oe1x)
o1e0[0] = 86400/o1e0[0]
for jj in range(1,np.min([len(Xr2save)-1,50])):
    oe1x = [o1[0][jj],xr2save[jj][0:3],xr2save[jj][3:]]
    o1e = orbdet.StateVector2OrbitalElements2(oe1x)
    o1e[0] = 86400/o1e[0]
    # if np.linalg.norm(Xr2save[jj] - Xr2save[jj+1]) <= np.min([np.linalg.norm(xr2save[ii])*0.001,5]):
    #     print("KF2 Converged")
    #     break
    if abs(o1e[0] - o1e0[0]) < 0.1 and abs(o1e[1] - o1e0[1]) < 0.01 and abs(o1e[2] - o1e0[2]) < 0.5:
        break
    o1e0 = o1e
oe1x = [o1[0][ii],xr3save[0][0:3],xr3save[0][3:]]
o1e0 = orbdet.StateVector2OrbitalElements2(oe1x)
o1e0[0] = 86400/o1e0[0]
for kk in range(1,np.min([len(Xr3save)-1,50])):
    oe1x = [o1[0][kk],xr3save[kk][0:3],xr3save[kk][3:]]
    o1e = orbdet.StateVector2OrbitalElements2(oe1x)
    o1e[0] = 86400/o1e[0]
    if abs(o1e[0] - o1e0[0]) < 0.1 and abs(o1e[1] - o1e0[1]) < 0.01 and abs(o1e[2] - o1e0[2]) < 0.5:
        break
    o1e0 = o1e
oe1x = [o1[0][ii],xrsave[ii][0:3],xrsave[ii][3:]]
o1e = orbdet.StateVector2OrbitalElements2(oe1x)
o1e[0] = 86400/o1e[0]
print('KF orbit prediction:',o1e[0],o1e[1],o1e[2]*180/np.pi,o1e[3]*180/np.pi,o1e[4]*180/np.pi,o1e[5]*180/np.pi,'Iterations to converge:',ii)
oe2x = [o1[0][jj],xr2save[jj][0:3],xr2save[jj][3:]]
o2e = orbdet.StateVector2OrbitalElements2(oe2x)
o2e[0] = 86400/o2e[0]
print('KF2 orbit prediction:',o2e[0],o2e[1],o2e[2]*180/np.pi,o2e[3]*180/np.pi,o2e[4]*180/np.pi,o2e[5]*180/np.pi,'Iterations to converge:',jj)
oe3x = [o1[0][kk],xr3save[kk][0:3],xr3save[kk][3:]]
o3e = orbdet.StateVector2OrbitalElements2(oe3x)
o3e[0] = 86400/o3e[0]
print('KF3 orbit prediction:',o3e[0],o3e[1],o3e[2]*180/np.pi,o3e[3]*180/np.pi,o3e[4]*180/np.pi,o3e[5]*180/np.pi,'Iterations to converge:',kk)

plt.figure()
p=plt.subplot(3,1,1)
plt.plot(Xsave1[0],'-b',alpha=0.6)
plt.plot(Xsave1[1],'-g',alpha=0.6)
plt.plot(Xsave1[2],'-r',alpha=0.6)
plt.plot(X2save1[0],'--b',alpha=0.6)
plt.plot(X2save1[1],'--g',alpha=0.6)
plt.plot(X2save1[2],'--r',alpha=0.6)
plt.plot(abs(w1[0]-xsave1[0]),'b:')
plt.plot(abs(w1[1]-xsave1[1]),'g:')
plt.plot(abs(w1[2]-xsave1[2]),'r:')
plt.title('Covariance')
p.set_ylim(0,np.max(xsave1))
p=plt.subplot(3,1,2)
plt.plot(Xrsave1[0],'-b',alpha=0.6)
plt.plot(Xrsave1[1],'-g',alpha=0.6)
plt.plot(Xrsave1[2],'-r',alpha=0.6)
plt.plot(Xr2save1[0],'--b',alpha=0.6)
plt.plot(Xr2save1[1],'--g',alpha=0.6)
plt.plot(Xr2save1[2],'--r',alpha=0.6)
plt.plot(Xr3save1[0],'.b',alpha=0.6)
plt.plot(Xr3save1[1],'.g',alpha=0.6)
plt.plot(Xr3save1[2],'.r',alpha=0.6)
# plt.plot(pk1[0],'-.b',alpha=0.6)
# plt.plot(pk1[1],'-.g',alpha=0.6)
# plt.plot(pk1[2],'-.r',alpha=0.6)
plt.plot(abs(w1[0]-xrsave1[0]),'b:')
plt.plot(abs(w1[1]-xrsave1[1]),'g:')
plt.plot(abs(w1[2]-xrsave1[2]),'r:')
plt.plot(abs(w1[0]-xr2save1[0]),'b-.')
plt.plot(abs(w1[1]-xr2save1[1]),'g-.')
plt.plot(abs(w1[2]-xr2save1[2]),'r-.')
plt.plot(abs(w1[0]-xr3save1[0]),'b:.')
plt.plot(abs(w1[1]-xr3save1[1]),'g:.')
plt.plot(abs(w1[2]-xr3save1[2]),'r:.')
p.set_ylim(0,np.max(xrsave1))
p=plt.subplot(3,1,3)
plt.plot(pk1[0],'-b',alpha=0.6)
plt.plot(pk1[1],'-g',alpha=0.6)
plt.plot(pk1[2],'-r',alpha=0.6)
plt.plot(abs(w1[0]-x1[0]),'b:')
plt.plot(abs(w1[1]-x1[1]),'g:')
plt.plot(abs(w1[2]-x1[2]),'r:')

plt.savefig('plots/KalmanTestPredictedCovariance.png')

plt.show()

