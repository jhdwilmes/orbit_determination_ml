
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


nombre = 'custom_ME_10s'
selectrandtle = 1
maxlimit = 500
tstep = datetime.timedelta(seconds=10)
uccs = locations.Location('UCCS',38.89588,-104.80232,1950)

a = sources.get_predictor_from_tle_lines(['1 00000C 00000AAA 00000.00000000 +.00000000 +00000-0 +00000-0 0 00000','2 00000 000.0000 000.0000 0000000 000.0000 000.0000 01.00000000000000'])
# f = open('tles/tle_custom_NE.txt','r')
f = open('tles/tle_custom_ME.txt','r')
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


t0 = datetime.datetime(2023,3,11,0,0,0)
t1 = datetime.datetime(2023,3,12,0,0,0)
t = t0
#print(tles[0].get_position(t),tles[0].get_next_pass(uccs))
#print(tles[0].passes_over(uccs,t0),uccs.get_azimuth_elev_deg(tles[0].get_position(t)))
ii = 0
f = open('datasets/tle_dataset_'+str(nombre)+'.csv','w+')

ltle = len(tles)
dtrate = 0.2
dtmod = 1/dtrate

while(ii<ltle and ii < maxlimit):
    t = t0
    tprev = t0
    dt = t - tprev
    dsave = []
    stepstaken = 0
    nn = ii
    if(selectrandtle):
        nn = np.random.randint(0,ltle)
    print(ii,'/',ltle)
    while(t<t1):
        t01 = t+datetime.timedelta(seconds=dtrate)
        pos = tles[nn].get_position(t)
        pos1 = tles[nn].get_position(t01)
        azel = uccs.get_azimuth_elev_deg(pos)
        azel1 = uccs.get_azimuth_elev_deg(pos1)
        dazel = np.subtract(azel1,azel)*dtmod*180/np.pi
        relPos = np.subtract(pos.position_ecef,uccs.position_ecef)
        relPos1 = np.subtract(pos1.position_ecef,uccs.position_ecef)
        position_eci = coordinate_systems.ecef_to_eci(pos.position_ecef,utils.gstime_from_datetime(t))
        #position_eci1 = coordinate_systems.ecef_to_eci(pos1.position_ecef,utils.gstime_from_datetime(t))
        relPos_eci = coordinate_systems.ecef_to_eci(relPos,utils.gstime_from_datetime(t))
        relPos_eci1 = coordinate_systems.ecef_to_eci(relPos1,utils.gstime_from_datetime(t01))
        radec = coordinate_systems.eci_to_radec(position_eci)
        #radec1 = coordinate_systems.eci_to_radec(position_eci1)
        radecRel = coordinate_systems.eci_to_radec(relPos_eci)
        radecRel1 = coordinate_systems.eci_to_radec(relPos_eci1)
        dradec = np.subtract(radecRel1,radecRel)*dtmod*180/np.pi
        if(azel[1] > 0):
            #print(ii,t,azel)
            rng = uccs.slant_range_km(pos.position_ecef)
            dt = t - tprev
            # 0 = satnum, 1 = date, 2 = time, 3 = delta time (since track start), 4 = az, 5 = el, 6 = range, 7 = true ra, 8 = true dec, 9 = obs ra, 10 = obs dec, 11 = ECI X, 12 = ECI Y, 13 = ECI Z, 14 = TLE period, 15 = azimuth rate, 16 = elevation rate, 17 = ra rate, 18 = dec rate
            f.write(str(tles[nn].tle[0])+','+str(t.date())+','+str(t.time())+','+str(dt)+','+str(azel[0])+','+str(azel[1])+','+str(rng)+','+str(radec[0]*180/math.pi)+','+str(radec[1]*180/math.pi)+','+str(radecRel[0]*180/math.pi)+','+str(radecRel[1]*180/math.pi)+','+str(position_eci[0])+','+str(position_eci[1])+','+str(position_eci[2])+','+str(tles[nn].period)+','+str(dazel[0])+','+str(dazel[1])+','+str(dradec[0])+','+str(dradec[1])+'\n')
            tprev = t
        t = t+tstep
    ii+=1
f.close()


