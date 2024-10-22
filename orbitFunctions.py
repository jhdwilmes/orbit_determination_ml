import numpy as np
from orbit_predictor import constants
from orbit_predictor import locations
from orbit_predictor import utils
from orbit_predictor import coordinate_systems
import datetime


class orbitFunctions(object):
    """
    Algorithms based predominantly on material found in "Orbital Mechanics for Engineering Students, 3rd Edition" (Curtis, Howard)
    """
    A = np.array([]) # Model A matrix (physics)
    B = np.array([]) # Model B matrix (forcing)
    C = np.array([]) # Model C matrix (measurement)
    D = np.array([]) # Model D matrix (hysteresis)
    S = np.array([])
    Sensor = locations.Location('UCCS',38.89588,-104.80232,1950)
    dt = 60
    time = 0.
    r0 = 10000.


    def Datetime2Jtime(self,dtime):
        jtim = utils.jday(dtime.year,dtime.month,dtime.day,dtime.hour,dtime.minute,dtime.second)
        return jtim


    def Jtime2Datetime(self,Jtime):
        Jstart = 2451545.0
        Tstart = datetime.datetime(2000,1,1,12,0,0)
        dJtime = Jtime - Jstart
        dtime = Tstart + datetime.timedelta(days=dJtime)
        return dtime


    # def Jtime2sidereal(self,Jtime):
    #     # returns sidereal time
    #     dtime = self.Jtime2Datetime(Jtime)
    #     utils.sidereal_time(dtime.timetuple)


    # def ECItoRADEC(self,eci):
    #     return 0


    def Xrotation(self,angle,matrix = np.eye(3)):
        Q = np.array([[1,0,0],[0,np.cos(angle),np.sin(angle)],[0,-np.sin(angle),np.cos(angle)]])
        return np.matmul(Q,matrix)


    def Yrotation(self,angle,matrix = np.eye(3)):
        Q = np.array([[np.cos(angle),0,-np.sin(angle)],[0,1,0],[np.sin(angle),0,np.cos(angle)]])
        return np.matmul(Q,matrix)
    
    
    def Zrotation(self,angle,matrix = np.eye(3)):
        Q = np.array([[np.cos(angle),np.sin(angle),0],[-np.sin(angle),np.cos(angle),0],[0,0,1]])
        return np.matmul(Q,matrix)


    def Qrotation(self,latitude,siderealTime):
        Q1 = np.array([-np.sin(siderealTime),-np.sin(latitude)*np.cos(siderealTime),np.cos(latitude)*np.cos(siderealTime)])
        Q2 = np.array([np.cos(siderealTime),-np.sin(latitude)*np.sin(siderealTime),np.cos(latitude)*np.sin(siderealTime)])
        Q3 = np.array([0,np.cos(latitude),np.sin(latitude)])
        QxX = np.array([Q1,Q2,Q3])
        return QxX


    def siderealTime(self,Jtime,obsLocation):
        UT = 0 # need to update this to better reflect time rotation?  If Jtime includes partial day, then do not change from zero - only used if Jtime always ends in 0.5
        T0 = (Jtime-2451545)/36525
        TG0 = (100.4606184 + 36000.77004 * T0 + .000387933 * T0**2 - 2.583e-8 * T0**3) % 360 # rotation in degrees (low order, may need updates)
        siderealTime = ((TG0 + obsLocation.longitude_deg + 360.98564724 * UT/24) % 360) * np.pi / 180 # need to verify these results are correct
        return siderealTime


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


    def AzEl2RADec(self,az,el,obsLocation,siderealTime):
        """Converts Az-El (AKA alt-az) into RA-Dec

        Args:
            az (float): Azimuth (Rad)
            el (float): elevation or altitude (Rad)
            obsLocation (location object): orbit_predictor location object (latitude is the part that matters)
            siderealTime (float): sidereal time (Rad)

        Returns:
            [float,float]: [Right Ascension (Rad), Declination (Rad)]
        """
        radec = self.AzEl2HDec(az,el,obsLocation)
        radec[0] = siderealTime - radec[0]
        return radec
        

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


    def RADec2AzEl(self,RA,Dec,obsLocation,siderealTime):
        """Converts RA-Dec into Az-El (AKA alt-az)

        Args:
            RA (float): Right Ascension (relative to observer, Radians)
            Dec (float): Declination (relative to observer, Radians)
            obsLocation (orbit_predictor location): Location object (uses latitude_rad)
            siderealTime (float): Local sidereal time (Rad)

        Returns:
            [float,float]: azimuth, elevation (or altitude)
        """
        sinRA = np.sin(RA)
        cosRA = np.cos(RA)
        cosLat = np.cos(obsLocation.latitude_rad)
        sinLat = np.sin(obsLocation.latitude_rad)
        cosS = np.cos(siderealTime)
        sinS = np.sin(siderealTime)
        cosDec = np.cos(Dec)
        sinDec = np.sin(Dec)
        el = np.arcsin(cosLat * sinRA * cosDec + sinLat * sinDec)
        # az = np.arctan2(cosLat * cosS * sinDec - sinS * cosRA * cosDec - sinLat * cosS * sinRA * cosDec, cosS * cosRA * cosDec - sinLat * sinS * sinRA * sinDec + cosLat * sinS * sinDec)
        az = np.arccos((cosS * cosRA * cosDec - sinLat * sinS * sinRA * sinDec + cosLat * sinS * sinDec) / np.cos(el))
        return [az,el]


    def sensorECILocation(self,siderealTime,latitude,longitude): # returns sensor ECI position given sidereal time, latitude, and longitude
        theta = np.pi - latitude
        r = utils.R_E_MEAN_KM # can add more fidelity if desired, this is a low fidelity model
        Pnominal = np.array([r * np.cos(longitude) * np.sin(theta), r * np.sin(longitude) * np.sin(theta), r * np.sin(theta)]) # need to verify this is correct
        Pactual = self.Zrotation(siderealTime,Pnominal)
        return Pactual


    def relativeRAdecPos(self,siderealTime,sensorLat,sensorLon,satelliteECI):
        Psensor = self.sensorECILocation(siderealTime,sensorLat,sensorLon)
        X = satelliteECI - Psensor
        r = np.linalg.norm(X)
        Dec = np.arcsin(X[2] / r)
        RA = np.arccos((X[0] / r) / np.cos(Dec)) % (2 * np.pi)
        return RA, Dec


    def orbitSensorModel(self,xh):#,zh):
        if(xh.ndim>1):
            xh = xh.flatten()
        time = self.time # zh[0] # need to ensure this is correct
        t = utils.gstime_from_datetime(self.Jtime2Datetime(time))
        ecef = coordinate_systems.eci_to_ecef(xh[0:3],t)
        horizon = coordinate_systems.to_horizon(self.Sensor.latitude_rad,self.Sensor.longitude_rad,self.Sensor.position_ecef,ecef)
        azel = coordinate_systems.horizon_to_az_elev(horizon[0],horizon[1],horizon[2])
        azel = [azel[0]*180/np.pi, azel[1]*180/np.pi]
        rng = self.Sensor.slant_range_km(ecef)
        observation = np.array([azel[0],azel[1],rng])
        return observation


    def orbitSensorModelRt(self,xh):#,zh):
        if(xh.ndim>1):
            xh = xh.flatten()
        time = self.time # zh[0] # need to ensure this is correct
        t = utils.gstime_from_datetime(self.Jtime2Datetime(time))
        ecef = coordinate_systems.eci_to_ecef(xh[0:3],t)
        horizon = coordinate_systems.to_horizon(self.Sensor.latitude_rad,self.Sensor.longitude_rad,self.Sensor.position_ecef,ecef)
        azel = coordinate_systems.horizon_to_az_elev(horizon[0],horizon[1],horizon[2])
        azel = [azel[0]*180/np.pi, azel[1]*180/np.pi]
        rng = self.Sensor.slant_range_km(ecef)
        dt = 1 # timedelta in seconds for rate calculations
        xh1 = self.orbitFunction6(xh,dt)
        t1 = (t + dt/86400) % (np.pi * 2)
        ecef1 = coordinate_systems.eci_to_ecef(xh1[0:3],t1)
        horizon1 = coordinate_systems.to_horizon(self.Sensor.latitude_rad,self.Sensor.longitude_rad,self.Sensor.position_ecef,ecef1)
        azel1 = coordinate_systems.horizon_to_az_elev(horizon1[0],horizon1[1],horizon1[2])
        azelrate = [azel1[0]*180/np.pi - azel[0], azel1[1]*180/np.pi - azel[1]]
        observation = np.array([azel[0],azel[1],rng,azelrate[0],azelrate[1]])
        return observation


    def orbitSensorModelRts(self,xh):#,zh):
        if(xh.ndim>1):
            xh = xh.flatten()
        time = self.time # zh[0] # need to ensure this is correct
        t = utils.gstime_from_datetime(self.Jtime2Datetime(time))
        ecef = coordinate_systems.eci_to_ecef(xh[0:3],t)
        horizon = coordinate_systems.to_horizon(self.Sensor.latitude_rad,self.Sensor.longitude_rad,self.Sensor.position_ecef,ecef)
        azel = coordinate_systems.horizon_to_az_elev(horizon[0],horizon[1],horizon[2])
        azel = [azel[0]*180/np.pi, azel[1]*180/np.pi]
        rng = self.Sensor.slant_range_km(ecef)
        dt = 1 # timedelta in seconds for rate calculations
        xh1 = self.orbitFunction6(xh,dt)
        t1 = (t + dt/86400) % (np.pi * 2)
        ecef1 = coordinate_systems.eci_to_ecef(xh1[0:3],t1)
        rngrate = (self.Sensor.slant_range_km(ecef1) - rng) / dt
        horizon1 = coordinate_systems.to_horizon(self.Sensor.latitude_rad,self.Sensor.longitude_rad,self.Sensor.position_ecef,ecef1)
        azel1 = coordinate_systems.horizon_to_az_elev(horizon1[0],horizon1[1],horizon1[2])
        azelrate = [azel1[0]*180/np.pi - azel[0], azel1[1]*180/np.pi - azel[1]]
        observation = np.array([azel[0],azel[1],rng,azelrate[0],azelrate[1],rngrate])
        return observation


    def orbitSensorModelAO(self,xh,zh):
        time = zh[0] #self.time # zh[0] # need to ensure this is correct
        t = utils.gstime_from_datetime(self.Jtime2Datetime(time))
        ecef = coordinate_systems.eci_to_ecef(xh[0:3],t)
        horizon = coordinate_systems.to_horizon(self.Sensor.latitude_rad,self.Sensor.longitude_rad,self.Sensor.position_ecef,ecef)
        azel = coordinate_systems.horizon_to_az_elev(horizon[0],horizon[1],horizon[2])
        azel = [azel[0]*180/np.pi,azel[1]*180/np.pi]
        # rng = self.Sensor.slant_range_km(ecef)
        observation = np.array([azel[0],azel[1]])
        return observation


    def orbitSensorModelAO2(self,xh): # Satellite true RA/Dec measurement
        eci = xh[0:3]
        orbitAltitude = np.linalg.norm(eci)
        RA = np.arctan2(eci[1],eci[0])
        Dec = np.arcsin(eci[2] / orbitAltitude)
        truRaDec = np.array([RA,Dec])
        return truRaDec


    def orbitSensorModelAO2r(self,xh): # Satellite true RA/Dec measurement
        eci = xh[0:3]
        orbitAltitude = np.linalg.norm(eci)
        RA = np.arctan2(eci[1], eci[0])
        Dec = np.arcsin(eci[2] / orbitAltitude)
        # deci = xh[3:]
        # dRA = np.arctan2(deci[1], deci[0]) # not sure if this is correct - would like directly from angles though
        # dDec = np.arcsin(deci[2] / orbitAltitude)
        dt = 1
        xh1 = self.orbitFunction6(xh,dt)
        RA1 = np.arctan2(xh1[1],xh1[0])
        Dec1 = np.arcsin(xh1[2] / np.linalg.norm(xh1[0:3]))
        dRA = (RA1 - RA) / dt
        dDec = (Dec1 - Dec) / dt
        truRaDec = np.array([RA, Dec, dRA, dDec])
        return truRaDec


    def orbitSensorModelAO3(self,xh,zh):
        if(xh.ndim>1):
            xh = xh.flatten()
        time = zh[0] # need to ensure this is correct
        t = utils.gstime_from_datetime(self.Jtime2Datetime(time))
        ecef = coordinate_systems.eci_to_ecef(xh[0:3],t)
        horizon = coordinate_systems.to_horizon(self.Sensor.latitude_rad,self.Sensor.longitude_rad,self.Sensor.position_ecef,ecef)
        azel = coordinate_systems.horizon_to_az_elev(horizon[0],horizon[1],horizon[2])
        azel = [azel[0]*180/np.pi,azel[1]*180/np.pi]
        rng = self.Sensor.slant_range_km(ecef)
        c = np.cross(xh[0:3],horizon)/1e6
        cn = np.linalg.norm(c)
        observation = np.array([azel[0],azel[1]])
        # self.C = np.transpose([[0,0,0,0,0,0,1,1,1],[horizon[0]/ecef[0]*xh[0]/azel[0],horizon[1]/ecef[1]*xh[1]/azel[0],horizon[2]/ecef[2]*xh[2]/azel[0],0,0,0,0,0,0],[horizon[0]/ecef[0]*xh[0]/azel[1],horizon[1]/ecef[1]*xh[1]/azel[1],horizon[2]/ecef[2]*xh[2]/azel[1],0,0,0,0,0,0]])
        self.C = np.array([[c[0]/(cn*azel[0]),c[1]/(cn*azel[0]),c[2]/(cn*azel[0]),0,0,0,0,0,0],[c[0]/(cn*azel[1]),c[1]/(cn*azel[1]),c[2]/(cn*azel[1]),0,0,0,0,0,0]])
        return observation


    def orbitSensorModelAOrate(self,xh,zh): # angles only, but also includes rate information
        if(xh.ndim>1):
            xh = xh.flatten()
        # Algorithm TBD
        time = zh[0] # need to ensure this is correct
        t = utils.gstime_from_datetime(self.Jtime2Datetime(time))
        ecef = coordinate_systems.eci_to_ecef(xh[0:3],t)
        horizon = coordinate_systems.to_horizon(self.Sensor.latitude_rad,self.Sensor.longitude_rad,self.Sensor.position_ecef,ecef)
        azel = coordinate_systems.horizon_to_az_elev(horizon[0],horizon[1],horizon[2])
        azel = [azel[0]*180/np.pi,azel[1]*180/np.pi]
        dt = 1
        xh1 = self.orbitFunction6(xh,dt)
        t1 = (t + dt/86400) % (np.pi * 2)
        ecef1 = coordinate_systems.eci_to_ecef(xh1[0:3],t1)
        horizon1 = coordinate_systems.to_horizon(self.Sensor.latitude_rad,self.Sensor.longitude_rad,self.Sensor.position_ecef,ecef1)
        azel1 = coordinate_systems.horizon_to_az_elev(horizon1[0],horizon1[1],horizon1[2])
        # dx = xh1 - xh
        azelrate = [azel1[0]*180/np.pi - azel[0], azel1[1]*180/np.pi - azel[1]] # need to finish this
        # rng = self.Sensor.slant_range_km(ecef)
        # c = np.cross(xh[0:3],horizon)/1e6
        # cn = np.linalg.norm(c)
        observation = np.array([azel[0],azel[1],azelrate[0],azelrate[1]])
        # self.C = np.transpose([[0,0,0,0,0,0,1,1,1],[horizon[0]/ecef[0]*xh[0]/azel[0],horizon[1]/ecef[1]*xh[1]/azel[0],horizon[2]/ecef[2]*xh[2]/azel[0],0,0,0,0,0,0],[horizon[0]/ecef[0]*xh[0]/azel[1],horizon[1]/ecef[1]*xh[1]/azel[1],horizon[2]/ecef[2]*xh[2]/azel[1],0,0,0,0,0,0]])
        # self.C = np.array([[c[0]/(cn*azel[0]),c[1]/(cn*azel[0]),c[2]/(cn*azel[0]),0,0,0,0,0,0],[c[0]/(cn*azel[1]),c[1]/(cn*azel[1]),c[2]/(cn*azel[1]),0,0,0,0,0,0]])
        return observation


    def orbitSensorModel2(self,xh):#,zh):
        time = self.time # zh[0]
        radec = coordinate_systems.eci_to_radec(xh[0:3])
        observation = [radec[0],radec[1]]
        return observation


    def updateAorbit(self,xh):
        r = np.sqrt(xh[0]**2 + xh[1]**2 + xh[2]**2)
        cr = -constants.MU_E/r**3
        A7 = [cr,0,0,0,0,0,0,0,0]
        A8 = [0,cr,0,0,0,0,0,0,0]
        A9 = [0,0,cr,0,0,0,0,0,0]
        A1 = [0,0,0,1,0,0,0,0,0]
        A2 = [0,0,0,0,1,0,0,0,0]
        A3 = [0,0,0,0,0,1,0,0,0]
        A4 = [0,0,0,0,0,0,1,0,0]
        A5 = [0,0,0,0,0,0,0,1,0]
        A6 = [0,0,0,0,0,0,0,0,1]
        self.A = np.array([A1,A2,A3,A4,A5,A6,A7,A8,A9])


    def createOrbitAhat2(self,xh,dt=60):
        dt = self.dt
        r = np.sqrt(xh[0]**2 + xh[1]**2 + xh[2]**2)
        cr = -constants.MU_E/r**5
        A7 = [-2*cr*xh[0],xh[1]*cr,xh[2]*cr,0,0,0,0,0,0]
        A8 = [xh[0]*cr,-2*cr*xh[1],xh[2]*cr,0,0,0,0,0,0]
        A9 = [xh[0]*cr,xh[1]*cr,-2*cr*xh[2],0,0,0,0,0,0]
        #A7 = [-cr*xh[0],0,0,0,0,0,0,0,0]
        #A8 = [0,-cr*xh[1],0,0,0,0,0,0,0]
        #A9 = [0,0,-cr*xh[2],0,0,0,0,0,0]
        A1 = [0,0,0,1,0,0,0,0,0]
        A2 = [0,0,0,0,1,0,0,0,0]
        A3 = [0,0,0,0,0,1,0,0,0]
        A4 = [0,0,0,0,0,0,1,0,0]
        A5 = [0,0,0,0,0,0,0,1,0]
        A6 = [0,0,0,0,0,0,0,0,1]
        Ahat = np.array([A1,A2,A3,A4,A5,A6,A7,A8,A9])/dt
        return Ahat


    def createOrbitAhat9(self,xh):
        dt = self.dt
        r = np.sqrt(xh[0]**2 + xh[1]**2 + xh[2]**2)
        cr = -constants.MU_E/r**3
        cr2 = -constants.MU_E/r**5
        #A7 = [-2*cr*xh[0],xh[1]*cr,xh[2]*cr,0,0,0,0,0,0]
        #A8 = [xh[0]*cr,-2*cr*xh[1],xh[2]*cr,0,0,0,0,0,0]
        #A9 = [xh[0]*cr,xh[1]*cr,-2*cr*xh[2],0,0,0,0,0,0]
        A1 = [0,0,0,1,0,0,0,0,0]
        A2 = [0,0,0,0,1,0,0,0,0]
        A3 = [0,0,0,0,0,1,0,0,0]
        A4 = [0,0,0,0,0,0,-cr*xh[0],0,0]
        A5 = [0,0,0,0,0,0,0,-cr*xh[1],0]
        A6 = [0,0,0,0,0,0,0,0,-cr*xh[2]]
        A7 = [-2*cr2*xh[0],xh[1]*cr2,xh[2]*cr2,0,0,0,0,0,0]
        A8 = [xh[0]*cr2,-2*cr2*xh[1],xh[2]*cr2,0,0,0,0,0,0]
        A9 = [xh[0]*cr2,xh[1]*cr2,-2*cr2*xh[2],0,0,0,0,0,0]
        Ahat = np.array([A1,A2,A3,A4,A5,A6,A7,A8,A9])*dt
        return Ahat


    def createOrbitAhat6(self,xh,dt=60):
        dt = self.dt
        r = np.sqrt(xh[0]**2 + xh[1]**2 + xh[2]**2)
        cr = -constants.MU_E/r**3
        #A7 = [-2*cr*xh[0],xh[1]*cr,xh[2]*cr,0,0,0,0,0,0]
        #A8 = [xh[0]*cr,-2*cr*xh[1],xh[2]*cr,0,0,0,0,0,0]
        #A9 = [xh[0]*cr,xh[1]*cr,-2*cr*xh[2],0,0,0,0,0,0]
        A1 = [0,0,0,1,0,0]
        A2 = [0,0,0,0,1,0]
        A3 = [0,0,0,0,0,1]
        A4 = [cr,0,0,0,0,0]
        A5 = [0,cr,0,0,0,0]
        A6 = [0,0,cr,0,0,0]
        Ahat = np.array([A1,A2,A3,A4,A5,A6])*dt
        return Ahat


    def createOrbitA6(self,xh):
        r = np.sqrt(xh[0]**2 + xh[1]**2 + xh[2]**2)
        cr = -constants.MU_E/r**3
        #A7 = [-2*cr*xh[0],xh[1]*cr,xh[2]*cr,0,0,0,0,0,0]
        #A8 = [xh[0]*cr,-2*cr*xh[1],xh[2]*cr,0,0,0,0,0,0]
        #A9 = [xh[0]*cr,xh[1]*cr,-2*cr*xh[2],0,0,0,0,0,0]
        A1 = [0,0,0,1,0,0]
        A2 = [0,0,0,0,1,0]
        A3 = [0,0,0,0,0,1]
        A4 = [cr,0,0,0,0,0]
        A5 = [0,cr,0,0,0,0]
        A6 = [0,0,cr,0,0,0]
        A = np.array([A1,A2,A3,A4,A5,A6])
        return A


    def createOrbitChat(self,xh):
        r = np.sqrt(xh[0]**2 + xh[1]**2 + xh[2]**2)
        cr = -constants.MU_E/r**5


    def orbitFunction9(self,xh,dt):#,u,xw): # full 9-dimensional orbital dynamics model
        # dt = self.dt
        r = np.sqrt(xh[0]**2 + xh[1]**2 + xh[2]**2)
        cr = constants.MU_E/r**3
        cr2 = constants.MU_E/r**5
        A1 = [0,0,0,1,0,0,0,0,0]
        A2 = [0,0,0,0,1,0,0,0,0]
        A3 = [0,0,0,0,0,1,0,0,0]
        A4 = [0,0,0,0,0,0,-cr*xh[0],0,0]
        A5 = [0,0,0,0,0,0,0,-cr*xh[1],0]
        A6 = [0,0,0,0,0,0,0,0,-cr*xh[2]]
        A7 = [-2*cr2*xh[0],xh[1]*cr2,xh[2]*cr2,0,0,0,0,0,0]
        A8 = [xh[0]*cr2,-2*cr2*xh[1],xh[2]*cr2,0,0,0,0,0,0]
        A9 = [xh[0]*cr2,xh[1]*cr2,-2*cr2*xh[2],0,0,0,0,0,0]
        A = np.array([A1,A2,A3,A4,A5,A6,A7,A8,A9])
        xh1 = np.matmul(A,xh)*dt + xh #+ xw
        return xh1


    def orbitFunction6(self,xh,dt):#,u,xw,dt): # 6-variable version of the orbital dynamics model
        # dt = self.dt
        r = np.sqrt(xh[0]**2 + xh[1]**2 + xh[2]**2)
        cr = -constants.MU_E/r**3
        A1 = [0,0,0,1,0,0]
        A2 = [0,0,0,0,1,0]
        A3 = [0,0,0,0,0,1]
        A4 = [cr,0,0,0,0,0]
        A5 = [0,cr,0,0,0,0]
        A6 = [0,0,cr,0,0,0]
        A = np.array([A1,A2,A3,A4,A5,A6])
        xh1 = np.matmul(A,xh)*dt + xh #+ xw
        return xh1


    def ellipseModel(self):
        self.A = np.array([])


    def initObservation(self,z): # need to verify the time format for this
        deg2rad = np.pi/180
        a = z[0]*deg2rad
        e = z[1]*deg2rad
        # tazelrng = np.array([self.r0*np.sin(e)*np.cos(a),self.r0*np.sin(e)*np.sin(a),self.r0*np.cos(e)])
        # tazelrng = [0,e,a,self.r0]
        # self.Sensor.position_ecef()
        loc = self.Sensor
        locecef = loc.position_ecef
        rrel = self.r0*np.array([np.sin(a)*np.cos(e),np.cos(a)*np.cos(e),np.sin(e)]) # verify this?
        Q1 = [-np.sin(loc.longitude_rad),-np.cos(loc.longitude_rad)*np.sin(loc.latitude_rad),np.cos(loc.longitude_rad)*np.cos(loc.latitude_rad)]
        Q2 = [np.cos(loc.longitude_rad),-np.sin(loc.longitude_rad)*np.sin(loc.latitude_rad),np.sin(loc.longitude_rad)*np.cos(loc.latitude_rad)]
        Q3 = [0,np.cos(loc.latitude_rad),np.sin(loc.latitude_rad)]
        Q = np.array([Q1,Q2,Q3])
        recefrel = np.matmul(Q,rrel)
        recef = recefrel + locecef
        reci = utils.ecef_to_eci(recef,self.time) # may just need rotation based on sidereal time?
        return recef


