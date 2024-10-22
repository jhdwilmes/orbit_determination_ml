
import orbit_predictor
from orbit_predictor import predictors
from orbit_predictor import locations
from orbit_predictor import coordinate_systems
from orbit_predictor import constants
from orbit_predictor import groundtrack
from orbit_predictor import sources
from orbit_predictor import utils
import datetime
import numpy as np
import math

import orbit_predictor.keplerian

import OrbitDetermination
import orbitFunctions


class TLEdata():
    def __init__(self,orbitfile=''):
        self.location = locations.Location('UCCS',38.89588,-104.80232,1950)
        self.tles = []
        if len(orbitfile) > 0:
            try:
                self.readTLEfile(orbitfile)
            except:
                print('Provided TLE file not found.')
        self.OD = OrbitDetermination.orbitDetermination()
        self.OF = orbitFunctions.orbitFunctions()


    def _R1rotation(self,a):
        Q = np.array([[1,0,0],[0,np.cos(a),np.sin(a)],[0,-np.sin(a),np.cos(a)]])
        return Q


    def _R3rotation(self,a):
        Q = np.array([[np.cos(a),np.sin(a),0],[-np.sin(a),np.cos(a),0],[0,0,1]])
        return Q
    
    
    def _orbitRotation(self,a1,a2,a3):
        # Q1 = np.array([-np.sin(a1)*np.cos(a2)*np.sin(a3)+np.cos(a1)*np.cos(a3),-np.sin(a1)*np.cos(a2)*np.cos(a3)-np.cos(a1)*np.sin(a3),np.sin(a1)*np.sin(a2)])
        # Q2 = np.array([np.cos(a1)*np.cos(a2)*np.sin(a3)+np.sin(a1)*np.cos(a3),np.cos(a1)*np.cos(a2)*np.cos(a3)-np.sin(a1)*np.sin(a3),-np.cos(a1)*np.sin(a2)])
        # Q3 = np.array([np.sin(a2)*np.sin(a3),np.sin(a2)*np.cos(a3),np.cos(a2)])
        # Q = np.array([Q1,Q2,Q3])
        Q1 = self._R3rotation(a1)
        Q2 = self._R1rotation(a2)
        Q3 = self._R3rotation(a3)
        Q = np.dot(Q3,Q2)
        Q = np.dot(Q,Q1)
        return Q


    def _singleTLetime(self,nn,timelength=1440):
        t = self.tles[nn].tle.lines[0][18:32]
        Dt = datetime.datetime(2000+int(t[0:2]),1,1,0,0,0) + datetime.timedelta(days=int(t[2:5]),seconds=float(t[5:])*86400)
        return Dt


    def _createTLEtime(self,timelength=1440,randomize=1):
        starthour = 0
        if randomize:
            starthour = np.random.randint(0,23)
        Dt0 = datetime.datetime(2000,1,1,starthour,0,0)
        for ii in range(len(self.tles)):
            Dt = self._singleTLetime(ii)
            if Dt > Dt0:
                Dt0 = Dt
        Dt2 = Dt0 + datetime.timedelta(minutes=timelength)
        return Dt0,Dt2


    def _createTLEfromOE(self,oe,timestamp=np.nan):
        r2d = 180/np.pi
        if np.isnan(timestamp)==True:
            time = datetime.datetime.now()
        else:
            time = timestamp
        doy = (time - datetime.datetime(time.year,1,1,0,0,0)).days
        # a = sources.get_predictor_from_tle_lines(['1 00000C 00000AAA 00000.00000000 +.00000000 +00000-0 +00000-0 0 00000','2 00000 000.0000 000.0000 0000000 000.0000 000.0000 01.00000000000000'])
        #       0 2      9        1820  24       3335       4445   515354   60626465    0 2     8 inc    17 raan  26 ecc  34 AP    43 MA    52 MM           68 checksum
        txt = ['1 00000U 00000A   00000.00000000 +.00000000 +00000-0 +00000-0 0 00000','2 00000 000.0000 000.0000 0000000 000.0000 000.0000 01.00000000000000']
        tstring = '{:02d}'.format(int(time.year - 100*np.floor(time.year/100)))
        # txt[0][18:20] = '{:02d}'.format(int(time.year - 100*np.floor(time.year/100))) #f'{num:.3f}'
        txt[0][0:18]+tstring
        dstring = '{:03d}'.format(int(doy))
        # txt[0][20:23] = '{:03d}'.format(int(doy))
        # txt[0][24:32] = '{message:{fill}{align}{width}}'.format(message=str(int((time.timestamp()%1)*100000000)),fill='0',align='<',width=8)
        todstring = '{message:{fill}{align}{width}}'.format(message=str(int((time.timestamp()%1)*100000000)),fill='0',align='<',width=8)
        # txt[1][8:16]  = '{:03.4f}'.format(oe[1])
        i = oe[2]*r2d
        inc = '{:03d}'.format(int(i))+'.'+'{:04d}'.format(int((i % 1)*1000))
        r = oe[3]*r2d
        raan = '{:03d}'.format(int(r))+'.'+'{:04d}'.format(int((r % 1)*1000))
        ecc = '{:07d}'.format(int(oe[1]*10000000))
        a = oe[4]*r2d
        ap = '{:03d}'.format(int(a))+'.'+'{:04d}'.format(int((a % 1)*1000))
        MA = np.arctan2(np.sqrt(1-oe[1]**2)*np.sin(oe[5]),1+oe[1]*np.cos(oe[5]))
        m = MA*r2d
        ma = '{:03d}'.format(int(m))+'.'+'{:04d}'.format(int((m % 1)*1000))
        mm = '{:02d}'.format(int(oe[0]))+'.'+'{:08d}'.format(int((oe[0] % 1)*100000000))
        rev = '{:05d}'.format(np.random.randint(0,99999))
        line1 = txt[0][0:18]+tstring+dstring+'.'+todstring+txt[0][32:]
        line2 = txt[1][0:8]+inc+' '+raan+' '+ecc+' '+ap+' '+ma+' '+mm+rev+'0'
        tle = [line1,line2]
        activeTLE = sources.get_predictor_from_tle_lines(tle)
        return activeTLE


    def _createOEfromTLE(self,tle):
        # a = sources.get_predictor_from_tle_lines(['1 00000C 00000AAA 00000.00000000 +.00000000 +00000-0 +00000-0 0 00000','2 00000 000.0000 000.0000 0000000 000.0000 000.0000 01.00000000000000'])
        #       0 2      9        1820  24       3335       4445   515354   60626465    0 2     8 inc    17 raan  26 ecc  34 AP    43 MA    52 MM           68 checksum
        txt = ['1 00000U 00000A   00000.00000000 +.00000000 +00000-0 +00000-0 0 00000','2 00000 000.0000 000.0000 0000000 000.0000 000.0000 01.00000000000000']
        tstmp = datetime.datetime(int(tle[0][18:20])+2000,1,1,0,0,0) + datetime.timedelta(days=float(tle[0][20:32]))
        inc = float(tle[1][8:16])
        raan = float(tle[1][17:25])
        ecc = float(tle[1][26:33]) / 1e7
        ap = float(tle[1][34:42])
        ma = float(tle[1][43:51])
        mm = float(tle[1][52:67])
        # p, ecc, inc, raan, argp, ta
        # mm, ecc, inc, raan, argp, ma
        return [mm, ecc, inc, raan, ap, ma]


    def readTLEfile(self,filename='tles/tle_custom.txt'):
        a = sources.get_predictor_from_tle_lines(['1 00000C 00000AAA 00000.00000000 +.00000000 +00000-0 +00000-0 0 00000','2 00000 000.0000 000.0000 0000000 000.0000 000.0000 01.00000000000000'])
        # f = open('tles/tle_custom_NE.txt','r')
        f = open(filename,'r')
        tletext = f.readlines()
        f.close()
        tles = []
        ii=0
        line1 = ''
        line2 = ''
        linecount = 0
        while(ii<len(tletext)):
            if(tletext[ii][0:2]=='1 '):
                line1 = tletext[ii]
                linecount+=1
            elif(tletext[ii][0:2]=='2 '):
                line2 = tletext[ii]
                linecount+=1
            if(linecount == 2):
                linecount = 0
                tles.append(sources.get_predictor_from_tle_lines([line1,line2]))
            ii+=1
        self.tles = tles
        return tles


    def createRandomOrbit(self,number=1,maxAltSeed=20000):
        for ii in range(number):
            Rapo = np.random.rayleigh()*maxAltSeed+(6371+100) # apogee radius
            Rperi = Rapo*(1-np.random.rand()**3) # perigee radius
            if Rperi < 6471.0:
                Rperi = 6471.0
            a = 0.5*(Rapo+Rperi)
            ecc = 1-Rperi/a # eccentricity
            h = np.sqrt(a*constants.MU_E*(1-ecc**2))
            b = a*np.sqrt(1-ecc**2)
            inc = (np.random.rand()-0.5)*np.pi # inclination in radians
            LAN = np.random.rand()*2*np.pi # longitude of the ascending node
            AP = np.random.rand()*2*np.pi # argument of perigee
            TA = np.random.rand()*2*np.pi # true anomaly
            T = np.pi*2*np.sqrt(a**3/constants.MU_E)
            # Some trouble converting elements to SV
            # x0 = np.array([a*(ecc+np.cos(TA))/(1+ecc*np.cos(TA)),b*np.sqrt(1-ecc**2)/(1+ecc*np.cos(TA))*np.sin(TA),0]) # starting position
            # # x0 = np.array([a*(np.cos(TA)-ecc),a*np.sin(TA),0])
            # # v0 = constants.MU_E/h*np.array([-np.sin(TA),ecc+np.cos(TA),0])
            # v0 = constants.MU_E/h * np.array([-(np.sin(TA)+ecc),(np.cos(TA)),0])
            # Q = self._orbitRotation(LAN,inc,AP) # need to verify this is done correctly
            # x = np.dot(Q,np.transpose(x0))
            # v = np.dot(Q,np.transpose(v0))
            # self.x = np.append(x,v)
            # self.A = self.createOrbitA6(self.x)
            # oe = self.OD.StateVector2OrbitalElements2([utils.MU_E,x,v])
            # if oe[1] >= 0.9: # this is janky, but the eccentricity of the orbital elements should be the same as that calculated above
            #     oe[1] = ecc
            oe = [T,ecc,inc,LAN,AP,TA]
            oe[0] = 86400/oe[0]
            try:
                tle = self._createTLEfromOE(oe)
            except:
                print("Orbit failed",oe)
                ii-=1
                continue
            self.tles.append(tle)
        self._createTLEtime()
        return tle


    def propagateTLE(self,tle,t0,t1,tstep=60,dtrate=0.5,maxObs=0):
        tstep = datetime.timedelta(seconds=tstep)
        t = t0
        tprev = t0
        dt = t - tprev
        dsave = []
        # stepstaken = 0
        ecidata = []
        observations = []
        dtmod = 1/dtrate
        if maxObs < 3:
            maxObs = 1e6 # default maximum observations
        obs = 0
        while(t<t1 and obs < maxObs):
            t01 = t+datetime.timedelta(seconds=dtmod)
            pos = tle.get_position(t)
            pos1 = tle.get_position(t01)
            azel = self.location.get_azimuth_elev_deg(pos)
            azel1 = self.location.get_azimuth_elev_deg(pos1)
            dazel = np.subtract(azel1,azel)*dtmod*180/np.pi
            relPos = np.subtract(pos.position_ecef,self.location.position_ecef)
            relPos1 = np.subtract(pos1.position_ecef,self.location.position_ecef)
            position_eci = coordinate_systems.ecef_to_eci(pos.position_ecef,utils.gstime_from_datetime(t))
            position_eci1 = coordinate_systems.ecef_to_eci(pos1.position_ecef,utils.gstime_from_datetime(t01))
            veci = (np.array(position_eci1) - np.array(position_eci)) * dtrate
            relPos_eci = coordinate_systems.ecef_to_eci(relPos,utils.gstime_from_datetime(t))
            relPos_eci1 = coordinate_systems.ecef_to_eci(relPos1,utils.gstime_from_datetime(t01))
            radec = coordinate_systems.eci_to_radec(position_eci)
            #radec1 = coordinate_systems.eci_to_radec(position_eci1)
            radecRel = coordinate_systems.eci_to_radec(relPos_eci)
            radecRel1 = coordinate_systems.eci_to_radec(relPos_eci1)
            dradec = np.subtract(radecRel1,radecRel)*dtmod*180/np.pi
            if(azel[1] > 0):
                #print(ii,t,azel)
                rng = self.location.slant_range_km(pos.position_ecef)
                dt = t - tprev
                # 0 = satnum, 1 = date, 2 = time, 3 = delta time (since track start), 4 = az, 5 = el, 6 = range, 7 = true ra, 8 = true dec, 9 = obs ra, 10 = obs dec, 11 = ECI X, 12 = ECI Y, 13 = ECI Z, 14 = TLE period, 15 = azimuth rate, 16 = elevation rate, 17 = ra rate, 18 = dec rate
                # f.write(str(tles[nn].tle[0])+','+str(t.date())+','+str(t.time())+','+str(dt)+','+str(azel[0])+','+str(azel[1])+','+str(rng)+','+str(radec[0]*180/math.pi)+','+str(radec[1]*180/math.pi)+','+str(radecRel[0]*180/math.pi)+','+str(radecRel[1]*180/math.pi)+','+str(position_eci[0])+','+str(position_eci[1])+','+str(position_eci[2])+','+str(tles[nn].period)+','+str(dazel[0])+','+str(dazel[1])+','+str(dradec[0])+','+str(dradec[1])+'\n')
                j = self.OF.Datetime2Jtime(t)
                #                    0-jtime 1-az   2-el  3-rng 4-ra true       5-dec true           6-ra relative           7-dec relative          8-dAz    9-dEl    10-dRA    11-dDec   12-datetime
                observations.append([j,azel[0],azel[1],rng,radec[0]*180/math.pi,radec[1]*180/math.pi,radecRel[0]*180/math.pi,radecRel[1]*180/math.pi,dazel[0],dazel[1],dradec[0],dradec[1],t])
                deci = np.append(position_eci,veci)
                ecidata.append(deci)
                tprev = t
                obs += 1
            t = t+tstep
        # f.close()
        return observations, np.array(ecidata)


    def oneTLEdata(self,tstep=60,selectrandtle=True,t0=np.nan,t1=np.nan,debug=True,tryagain=0,noise=0,maxObs=0,minObs=4):
        if np.isnan(t0) and np.isnan(t1):
            t0,t1 = self._createTLEtime()
        elif np.isnan(t1):
            t1 = t0 + datetime.timedelta(minutes=1440)
        elif np.isnan(t0):
            t0 = t1 - datetime.timedelta(minutes=1440)
        #print(tles[0].get_position(t),tles[0].get_next_pass(uccs))
        #print(tles[0].passes_over(uccs,t0),uccs.get_azimuth_elev_deg(tles[0].get_position(t)))
        ii = 0

        ltle = len(self.tles)
        dtrate = 0.5

        dsave = []
        stepstaken = 0
        nn = ii
        if(selectrandtle):
            nn = np.random.randint(0,ltle)
        # print(ii,'/',ltle)
        tle = self.tles[nn]
        obs, ecidata = self.propagateTLE(tle,t0,t1,tstep,dtrate,maxObs)
        if len(obs) < minObs:
            if debug:
                print('Low obs number, consider trying different TLE or date range')
            for ii in range(tryagain):
                nn = np.random.randint(0,ltle)
                tle = self.tles[nn]
                obs, ecidata = self.propagateTLE(tle,t0,t1,tstep,dtrate,maxObs)
                if len(obs) >= minObs:
                    break
        if noise > 0:
            noisechar = noise * np.array([0,1/60,1/60,5.0,1/60,1/60,1/60,1/60,1/30,1/30,1/30,1/30])
            for ii in range(len(obs)):
                obs[ii][0:-1] = obs[ii][0:-1] + np.multiply(np.random.randn(len(noisechar)),noisechar)
        tleorbit = self._createOEfromTLE(tle.tle.lines)
        # truthdata = [ecidata, tleorbit]
        return obs, ecidata, tleorbit

