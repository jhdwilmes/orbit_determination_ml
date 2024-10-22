import numpy as np
import pandas as pd
import datetime
from orbit_predictor import predictors
from orbit_predictor import constants
from orbit_predictor import locations
from orbit_predictor import utils
from orbit_predictor import coordinate_systems
from orbit_predictor import keplerian


class orbitDetermination:
    #orbits = None


    def AzEl2HDec(self,az,el,obsLocation):
        """Converts Az-El (AKA Alt-Az) into H-Dec

        Args:
            az (float): azimuth (Rad)
            el (float): elevation (or altitude) (Rad)
            obsLocation (location object): location object

        Returns:
            [H,Dec]: [Hour angle, Declination] (float,float)
        """
        cosAz = np.cos(az)
        cosEl = np.cos(el)
        # sinAz = np.sin(az)
        sinEl = np.sin(el)
        cosLat = np.cos(obsLocation.latitude_rad)
        sinLat = np.sin(obsLocation.latitude_rad)
        Dec = np.arcsin(cosLat * cosAz * cosEl + sinLat * sinEl)
        H = np.arccos(cosLat * sinEl - sinLat * cosAz * cosEl) / (np.cos(Dec))
        return [H,Dec]


    def HDec2AzEl(self,H,Dec,obsLocation):
        """Converts H-Dec into Az-El (AKA alt-az)

        Args:
            H (float): hour angle (Rad)
            Dec (float): Declination (Rad)
            obsLocation (orbit_predictor location): orbit_predictor location object (latitude is the part that matters)

        Returns:
            [float,float]: azimuth, elevation (or altitude)
        """
        sinH = np.sin(H)
        cosH = np.cos(H)
        cosLat = np.cos(obsLocation.latitude_rad)
        sinLat = np.sin(obsLocation.latitude_rad)
        cosDec = np.cos(Dec)
        sinDec = np.sin(Dec)
        az = np.arctan2(-cosDec * sinH, -sinLat * cosDec * cosH + cosLat * sinDec)
        el = np.arccos((-cosDec * sinH) / np.sin(az))
        return [az,el]


    def siderealTime(self,Jtime,obsLocation):
        UT = 0 # need to update this to better reflect time rotation?  If Jtime includes partial day, then do not change from zero - only used if Jtime always ends in 0.5
        T0 = (Jtime-2451545)/36525
        TG0 = (100.4606184 + 36000.77004 * T0 + .000387933 * T0**2 - 2.583e-8 * T0**3) % 360 # rotation in degrees (low order, may need updates)
        siderealTime = ((TG0 + obsLocation.longitude_deg + 360.98564724 * UT/24) % 360) * np.pi / 180 # need to verify these results are correct
        return siderealTime


    def Qrotation(self,latitude,siderealTime):
        Q1 = np.array([-np.sin(siderealTime),-np.sin(latitude)*np.cos(siderealTime),np.cos(latitude)*np.cos(siderealTime)])
        Q2 = np.array([np.cos(siderealTime),-np.sin(latitude)*np.sin(siderealTime),np.cos(latitude)*np.sin(siderealTime)])
        Q3 = np.array([0,np.cos(latitude),np.sin(latitude)])
        QxX = np.array([Q1,Q2,Q3])
        return QxX


    def QrotateVector(self,vector,Jtime,obsLocation,direction=0):
        UT = 0 # need to update this to better reflect time rotation?  If Jtime includes partial day, then do not change from zero - only used if Jtime always ends in 0.5
        T0 = (Jtime-2451545)/36525
        TG0 = (100.4606184 + 36000.77004 * T0 + .000387933 * T0**2 - 2.583e-8 * T0**3) % 360 # rotation in degrees (low order, may need updates)
        siderealTime = ((TG0 + obsLocation.longitude_deg + 360.98564724 * UT/24) % 360) * np.pi / 180
        Q = self.Qrotation(obsLocation.latitude_rad,siderealTime)
        if direction==0:
            return np.dot(vector,Q)
        else:    
            return np.transpose(np.dot(Q,np.transpose(vector))) # make sure this rotation is in the correct direction
        

    def Jday2DT(self,jday):
        # modjday = jday - utils.DECEMBER_31TH_1999_MIDNIGHT_JD
        # jt = datetime.timedelta(days=modjday)
        # dt = jt + datetime.datetime(1999,12,31,0,0,0) # Check if this is the right date - may need to change to 2000,1,1,0,0,0
        dt = utils.datetime_from_jday(jday,0)
        return dt


    def Datetime2Jtime(self,dtime):
        jtim = utils.jday(dtime.year,dtime.month,dtime.day,dtime.hour,dtime.minute,dtime.second)
        return jtim


    def findRoot(self,a,b,c,x0=1e6,maxiter = 100,precision=1e-3):
        ii = 0
        xi1 = x0
        xi = xi1*2+2*precision
        while(ii<100 and abs(xi-xi1) > precision):
            xi = xi1
            xi1 = xi - (xi**8 + a*xi**6 + b*xi**3 + c)/(8*xi**7 + 6*a*xi**5 + 3*b*xi**2)
            ii+=1
        return xi1


    def GaussOrbitDetermination(self,timeset,angles,obsLocation,radecOrazel = 0):
        if len(timeset) < 3:
            print('Insufficient data to calculate orbit')
            return([0,np.array([0,0,0]),np.array([0,0,0])])
        # format times
        Rs = []
        times = []
        for t in timeset:
            t1 = t
            if type(t)!=type(datetime.datetime(1,1,1)):
                t1 = self.Jday2DT(t)
                times.append(t)
            elif type(t)==type(datetime.datetime(1,1,1)):
                times.append(self.Datetime2Jtime(t))
            Rs.append(coordinate_systems.ecef_to_eci(obsLocation.position_ecef,utils.gstime_from_datetime(t1)))
        # format angles
        if(radecOrazel == 0):
            ra = angles[0] * utils.pi / 180
            dec = angles[1] * utils.pi / 180
            rhos = np.transpose([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
        elif(radecOrazel == 1):
            az = angles[0] * utils.pi / 180 
            el = angles[1] * utils.pi / 180
            # rhos = np.transpose([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
            # rhos = self.QrotateVector(rhos,times[1],obsLocation,direction=1) # not sure if this is correct?
            h,dec = self.AzEl2HDec(az,el,obsLocation)
            siderealTime = self.siderealTime(times[1],obsLocation)
            ra = h + siderealTime
            rhos = np.transpose([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
            # need to account for rotation into topocentric coordinates
        else:
            print('Need RaDec or AzEl angle coordinates inputed as the "angles" variable, and radecOrazel set to 0 for RaDec or 1 for AzEl.')
            return 1
        T1 = times[0] - times[1] # tau_1 # assumed julian dates, meaning the result should be fractions of a day
        T3 = times[2] - times[1] # tau_3
        T = T3 - T1 # tau
        # convert time to seconds - note there might be numerical problems here if the time delta is too small
        T1 = T1 * 86400
        T3 = T3 * 86400
        T  = T * 86400
        # product vectors
        p1 = np.cross(rhos[1],rhos[2]) # Rho2 x Rho3
        p2 = np.cross(rhos[0],rhos[2]) # Rho1 x Rho3
        p3 = np.cross(rhos[0],rhos[1]) # Rho1 x Rho2
        D0 = np.dot(rhos[0],p1)
        # label sensor locations at each time
        R1 = Rs[0]
        R2 = Rs[1]
        R3 = Rs[2]
        # create matrix coefficients
        D11 = np.dot(R1,p1)
        D12 = np.dot(R1,p2)
        D13 = np.dot(R1,p3)
        D21 = np.dot(R2,p1)
        D22 = np.dot(R2,p2)
        D23 = np.dot(R2,p3)
        D31 = np.dot(R3,p1)
        D32 = np.dot(R3,p2)
        D33 = np.dot(R3,p3)
        # Range solution
        A = 1 / D0 * (-D12 * T3 / T + D22 + D32 * T1 / T)
        B = 1 / (6 * D0) * (D12 * (T3**2 - T**2) * T3 / T + D32 * (T**2 - T1**2) * T1 / T)
        E = np.dot(R2,rhos[1])
        R22 = np.dot(R2,R2)
        a = -(A**2 + 2 * A * E + R22)
        b = -2 * utils.MU_E * B * (A + E)
        c = -utils.MU_E**2 * B**2
        r2 = self.findRoot(a, b, c)
        # Velocity solution
        r1num = 6 * (D31 * T1 / T3 + D21 * T / T3) * r2**3 + utils.MU_E * D31 * (T**2 - T1**2) * T1 / T3
        r1denom = 6 * r2**3 + utils.MU_E * (T**2 - T3**2)
        rho1 = 1 / D0 * (r1num / r1denom - D11)
        rho2 = A + utils.MU_E * B / r2**3
        r3num = 6 * (D13 * T3 / T1 - D23 * T / T1) * r2**3 + utils.MU_E * D13 * (T**2 - T3**2) * T3 / T1
        r3denom = 6 * r2**3 + utils.MU_E * (T**2 - T1**2)
        rho3 = 1 / D0 * (r3num / r3denom - D33)
        x1 = R1 + rho1 * rhos[0]
        x2 = R2 + rho2 * rhos[1]
        x3 = R3 + rho3 * rhos[2]
        f1 = 1 - 0.5 * utils.MU_E * T1**2 / r2**3
        f3 = 1 - 0.5 * utils.MU_E * T3**2 / r2**3
        g1 = T1 - utils.MU_E / (6 * r2**3) * T1**3
        g3 = T3 - utils.MU_E / (6 * r2**3) * T3**3
        v2 = 1 / (f1 * g3 - f3 * g1) * (-f3 * x1 + f1 * x3)
        return [times[1], x2, v2]


    def StateVector2OrbitalElements(self,SV):
        OE = np.array(keplerian.rv2coe(utils.MU_E,SV[1],SV[2]))
        if np.isnan(OE[0]):
            OE[0] = 0
        return OE


    def OrbitElement2StateVector(self,OE):
        SV = np.array(keplerian.coe2rv(utils.MU_E,OE[0],OE[1],OE[2],OE[3],OE[4],OE[5]))
        # semi-major axis: a = OE[0]/(1-OE[1]**2)
        return SV
    

    def Period2SemiLatusRectum(self,T,ecc):
        a = ((T/(2*utils.pi))**2 * utils.MU_E)**(1/3)
        SML = a*(1-ecc**2)
        return SML


    def Period2SemiMajorAxis(self,T):
        return ((T/(2*utils.pi))**2 * utils.MU_E)**(1/3)
    

    def OrbitElement2StateVector2(self,OE):
        T = OE[0]
        a = ((T/(2*utils.pi))**2 * utils.MU_E)**(1/3)
        OE [0] = a*(1-OE[1]**2)
        SV = np.array(keplerian.coe2rv(utils.MU_E,OE[0],OE[1],OE[2],OE[3],OE[4],OE[5]))
        # semi-major axis: a = OE[0]/(1-OE[1]**2)
        return SV
    

    def StateVector2OrbitalElements2(self,SV):
        OE = np.array(keplerian.rv2coe(utils.MU_E,SV[1],SV[2]))
        a = OE[0]/(1-OE[1]**2)
        T = utils.pi*2*np.sqrt(a**3/utils.MU_E) # orbit period
        OE[0] = T
        if np.isnan(OE[0]):
            OE[0] = 0
        return OE


    def StateVector2OrbitalElements3(self,SV):
        OE = np.array(keplerian.rv2coe(utils.MU_E,SV[1],SV[2]))
        a = OE[0]/(1-OE[1]**2) # semi-major axis
        OE[0] = a
        if np.isnan(OE[0]):
            OE[0] = 0
        return OE


    def StateVector2OrbitalElements4(self,SV):
        OE = np.array(keplerian.rv2coe(utils.MU_E,SV[1],SV[2]))
        a = OE[0]/(1-OE[1]**2)
        T = utils.pi*2*np.sqrt(a**3/utils.MU_E) # orbit period
        OE[0] = 86400/T
        if np.isnan(OE[0]):
            OE[0] = 0
        return OE
    

    def OrbitElement2StateVector3(self,OE):
        a = OE[0]
        OE[0] = a*(1-OE[1]**2)
        SV = np.array(keplerian.coe2rv(utils.MU_E,OE[0],OE[1],OE[2],OE[3],OE[4],OE[5]))
        # semi-major axis: a = OE[0]/(1-OE[1]**2)
        return SV


    def OrbitElement2StateVector4(self,OE):
        M = OE[0]
        T = 86400/M
        a = ((T/(2*utils.pi))**2 * utils.MU_E)**(1/3)
        OE [0] = a*(1-OE[1]**2)
        SV = np.array(keplerian.coe2rv(utils.MU_E,OE[0],OE[1],OE[2],OE[3],OE[4],OE[5]))
        # semi-major axis: a = OE[0]/(1-OE[1]**2)
        return SV


    def AzElRng2SV(self,tazelrng,tazelrng1,loc):
        # tazelrng is time, az, el, rng
        locecef = loc.position_ecef
        rrel1 = tazelrng[3]*np.array([np.sin(tazelrng1[1]*np.pi/180)*np.cos(tazelrng1[2]*np.pi/180),np.cos(tazelrng[1]*np.pi/180)*np.cos(tazelrng1[2]*np.pi/180),np.sin(tazelrng1[2]*np.pi/180)]) 
        rrel = tazelrng[3]*np.array([np.sin(tazelrng[1]*np.pi/180)*np.cos(tazelrng[2]*np.pi/180),np.cos(tazelrng[1]*np.pi/180)*np.cos(tazelrng[2]*np.pi/180),np.sin(tazelrng[2]*np.pi/180)])
        Q1 = [-np.sin(loc.longitude_rad),-np.cos(loc.longitude_rad)*np.sin(loc.latitude_rad),np.cos(loc.longitude_rad)*np.cos(loc.latitude_rad)]
        Q2 = [np.cos(loc.longitude_rad),-np.sin(loc.longitude_rad)*np.sin(loc.latitude_rad),np.sin(loc.longitude_rad)*np.cos(loc.latitude_rad)]
        Q3 = [0,np.cos(loc.latitude_rad),np.sin(loc.latitude_rad)]
        Q = np.array([Q1,Q2,Q3])
        recefrel = np.matmul(Q,rrel)
        recef = recefrel + locecef
        recefrel1 = np.matmul(Q,rrel1)
        recef1 = recefrel1 + locecef
        v = (recef1 - recef) / ((tazelrng1[0] - tazelrng[0])*86400)
        dt = self.Jday2DT(tazelrng[0])
        #SVecef = [utils.timetuple_from_dt(dt),recef,v]
        SVecef = [tazelrng[0],recef,v]
        gst = utils.gstime_from_datetime(dt)
        SV = np.array([SVecef[0],np.array(coordinate_systems.ecef_to_eci(SVecef[1],gst)),np.array(coordinate_systems.ecef_to_eci(SVecef[2],gst)) + np.array(coordinate_systems.ecef_to_eci(np.array([SVecef[1][0],SVecef[1][1],0]),gst))*np.pi*2.0/86164.0])
        return SV


