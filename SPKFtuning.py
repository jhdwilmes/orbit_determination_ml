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
import time


LD = createDataLive.TLEdata()
LD.readTLEfile('tles/tle_11052024.txt')
LD.prunTLEs([.001,15])
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


dt = 5
o,w,t = LD.oneTLEdata(tstep=dt,tryagain=5,noise=0.004,selecttle=20)
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

OF = orbitFunctions.orbitFunctions()
dim = 6
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

Xsave = []
X2save = []

imax = 1000
maxiter = np.min([len(zAzEl)-1,100])

alphas = np.linspace(0.1,1,10)**5
betas = np.linspace(0.0,4,13)
kappas = np.linspace(-4,3,15)

t1 = time.time()

Zs = zAzElRng
Zdim = 3
orbitfunc = OF.orbitSensorModel
R = np.diag([.01,.01,99.9])
Rmod = [[.0001,0,0],[0,.0001,0],[0,0,.9]]

# Zs = zRaDec
# Zdim = 2
# orbitfunc = OF.orbitSensorModelAO2
# R = np.diag([.01,.01])
# Rmod = [[.0001,0],[0,.0001]]

# Zs = zRaDecRt
# Zdim = 4
# orbitfunc = OF.orbitSensorModelAO2r
# R = np.diag([.01,.01,.01,.01])
# Rmod = [[.0001,0,0,0],[0,.0001,0,0],[0,0,.0001,0],[0,0,0,.0001]]

Ps = []
Ks = []
Bs = []
As = []
Xs = []
alpha = .01
beta = 2
kappa = 0
params = [[.001,0,0],[-.001,.1,0],[0,.1,0],[0,-.1,0],[0,0,.1],[0,0,-.1]]
maxiter = 50
numparams = len(params)
numRs = len(Rmod)
for ii in range(maxiter):
    errors  = []
    errorsm = []
    for kk in range(numRs):
        points = sigma_points.MerweScaledSigmaPoints(dim, alpha=alpha, beta=beta, kappa=kappa) # need subtract function for angles?
        UKF  = UnscentedKalmanFilter(dim_x=dim, dim_z=Zdim, dt=dt, fx=OF.orbitFunction6, hx=orbitfunc, points=points)
        UKFm = UnscentedKalmanFilter(dim_x=dim, dim_z=Zdim, dt=dt, fx=OF.orbitFunction6, hx=orbitfunc, points=points)
        rupdate = []
        rupdatem = []
        for rr in range(len(Rmod)):
            rupdate.append(R[rr][rr] + Rmod[kk][rr])
            rupdatem.append(R[rr][rr] - Rmod[kk][rr])
        UKF.R  = np.diag(rupdate)
        UKFm.R = np.diag(rupdatem)
        UKF.x  = np.append(gausoln3[1],gausoln3[2])
        UKFm.x = np.append(gausoln3[1],gausoln3[2])
        tt = 0
        for z in Zs[0:maxiter]:
            OF.time = o1[0][tt]
            UKF.predict(dt=dt)
            UKF.update(z)
            UKFm.predict(dt=dt)
            UKFm.update(z)
            tt+=1
        errors.append(abs(np.dot(UKF.x  / np.linalg.norm(w[maxiter])**2, w[maxiter] / np.linalg.norm(w[maxiter])**2) - 1))
        errorsm.append(abs(np.dot(UKFm.x / np.linalg.norm(w[maxiter])**2, w[maxiter] / np.linalg.norm(w[maxiter])**2) - 1))
    errors  = np.array(errors)
    errorsm = np.array(errorsm)
    R = R + np.diag(1e-12/(errorsm - errors + (errors==errorsm)*1e22))
    for ll in range(len(np.diag(R))):
        if R[ll][ll] < 1e-3:
            R[ll][ll] = 1e-3
    errors  = []
    errorsm = []
    for jj in range(numparams):
        points = sigma_points.MerweScaledSigmaPoints(dim, alpha=alpha + params[jj][0], beta=beta + params[jj][1], kappa=kappa + params[jj][2]) # need subtract function for angles?
        UKF  = UnscentedKalmanFilter(dim_x=dim, dim_z=Zdim, dt=dt, fx=OF.orbitFunction6, hx=orbitfunc, points=points)
        UKF.R  = R
        UKF.x  = np.append(gausoln3[1],gausoln3[2])
        tt = 0
        for z in Zs[0:maxiter]:
            OF.time = o1[0][tt]
            UKF.predict(dt=dt)
            UKF.update(z)
            tt+=1
        errors.append(abs(np.dot(UKF.x  / np.linalg.norm(w[maxiter])**2, w[maxiter] / np.linalg.norm(w[maxiter])**2) - 1))
    alpha = alpha + 1e-18 / (errors[1] - errors[0] + (errors[1]==errors[0])*1e12)
    if alpha < 1e-4:
        alpha = 1e-4
    beta  = beta  + 1e-16 / (errors[3] - errors[2] + (errors[3]==errors[2])*1e12)
    if beta < 0:
        beta = 0
    kappa = kappa + 1e-16 / (errors[5] - errors[4] + (errors[5]==errors[4])*1e12)

t2 = time.time()
print("time utilized:",t2-t1)

Xs = np.array(Xs)
Ps = np.array(Ps)

print("Orbit SV:",w[0])
print("Orbit elements:",t)
print("Tuning noise characteristics:\n",R)
print("Tuned alpha:",alpha,"Beta:",beta,"Kappa:",kappa)

points = sigma_points.MerweScaledSigmaPoints(dim, alpha=alpha, beta=beta, kappa=kappa)
UKF = UnscentedKalmanFilter(dim_x=dim, dim_z=Zdim, dt=dt, fx=OF.orbitFunction6, hx=orbitfunc, points=points)
UKF.R = R
UKF.x = np.append(gausoln3[1],gausoln3[2])

xsez = []
Psez = []
for z in Zs[0:maxiter]:
    OF.time = o1[0][ii]
    UKF.predict(dt=dt)
    UKF.update(z)
    xsez.append(UKF.x)
    Psez.append(np.diag(UKF.P))
    ii+=1

xsez = np.array(xsez)
Psez = np.array(Psez)

plt.figure()
plt.subplot(2,1,1)
# plt.title("Tuned Kalman Filter Results")
plt.plot(xsez[:,0])
plt.plot(xsez[:,1])
plt.plot(xsez[:,2])
plt.ylabel('Position')
plt.legend(["X","Y","Z"])
plt.subplot(2,1,2)
plt.plot(xsez[:,3])
plt.plot(xsez[:,4])
plt.plot(xsez[:,5])
plt.ylabel('Velocity')
plt.savefig('plots/tunedSPKFpredictions2.png')