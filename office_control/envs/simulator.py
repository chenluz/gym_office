# the actions can be any kinds of control to building
import requests
from requests.auth import HTTPBasicAuth
import json
import base64
import pytz, datetime
import time
import math
import pickle
from sklearn.kernel_ridge import KernelRidge
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
## ref: https://github.com/CenterForTheBuiltEnvironment/comfort_tool/blob/master/contrib/comfort_models.py
############need to change for different users #############
Ts_min = 25
Ts_max = 37
Ta_min = 16.5
Ta_max = 32
Rh_min = 12
Rh_max = 44
Rh_out_min = 24
Rh_out_max = 85
Ta_out_min = 5.77
Ta_out_max = 22
Ta_action_min = 18.6
Ta_action_max = 24.8
Ta_out_action_min = 5.77
Ta_out_action_max = 10.1
############need to change for different users #############
A_min = 0
A_max = 3

class outdoorSet():
    """
    read the outdoor tempeature and relative humidity from csv
    """
    def __init__(self):  
        ##### process outdoor air temperature and relative humidity 
        data_outdoor = pd.read_csv("user4/environment-outdoor-2018-3-13.csv", names = ["Time", "Out_Temperature", "Out_Humidity"])
        data_outdoor.index = pd.to_datetime(data_outdoor.Time)
        self.data_outdoor = data_outdoor.resample('300s').mean().dropna()
    
    def get_out(self, step_count):
        index = self.data_outdoor.index
        Out_Temperature = self.data_outdoor.get_value(index[step_count], "Out_Temperature")
        Out_Humidity = self.data_outdoor.get_value(index[step_count], "Out_Humidity")
        return (Out_Temperature, Out_Humidity)


class airEnviroment():
    """
    Cacluate air temperature based on action take from kernerl_regression_model

    """
    def __init__(self):  
        self.temp_model = pickle.load(open("user4/envSimulator_Action4_user4_Air Temperature_SVR.sav", 'rb'))
        self.humi_model = pickle.load(open("user4/temperature_user4_Humidity_kernel.sav", 'rb'))

    def get_air_temp(self,action, pre_Ta, pre_out_Ta):  
        _input = self.process_state_air(action, pre_Ta, pre_out_Ta)
        # input should be indoor, action outdoor
        Ta = self.temp_model.predict([_input]).flatten()[0]
        result = Ta*(Ta_action_max - Ta_action_min) + Ta_action_min
        if action == 3:
            result = 8/pre_Ta + pre_Ta + 0.01*pre_out_Ta
        elif action == 2:
            result = 6/pre_Ta + pre_Ta + 0.01*pre_out_Ta
        elif action == 1:
            result = 2/pre_Ta + pre_Ta + 0.01*pre_out_Ta
        else: 
            result = pre_Ta - 0.01*(10 - pre_out_Ta)
        return result

    def process_state_air(self, action, pre_Ta, pre_out_Ta):
        action = (action - A_min)/(A_max - A_min)
        pre_Ta = (pre_Ta - Ta_action_min)/(Ta_action_max - Ta_action_min)
        pre_out_Ta = (pre_out_Ta - Ta_out_action_min)/(Ta_out_action_max - Ta_out_action_min)
        return [pre_Ta, action, pre_out_Ta]

    def get_air_humidity(self, Ta, out_Rh):
        
        _input = self.process_state_humidity(Ta, out_Rh)
        # input should be indoor, action outdoor
        Rh = self.humi_model.predict([_input]).flatten()[0]
        result = Rh*(Rh_max - Rh_min) + Rh_min
        return result

    def process_state_humidity(self, Ta, out_Rh):
        Ta = (Ta - Ta_min)/(Ta_max - Ta_min)
        out_Rh = (out_Rh - Rh_min)/(Rh_max - Rh_min)
        return [Ta, out_Rh]


class airVelocity():
    """
    Cacluate the air velocity distribution based on Fan scale and location of the room
    """

    def __init__(self):  
        pass

    

    def get_air_velocity(self, fan, location = "B1"):
        """ get the air velocity based on Fan scale and location of the room 
        Parameters
        ----------
        fan: int, the setpoint of the fan
        location: str, location lable from the air velocity analysis study

        Return
        ----------
        speed: float, air velocity at location B1

        """
        speed = 0.2929*fan
        return speed               
                               


class skinTemperature():
    """
    Calculate skin temperature based on environmental variables
    """

    def __init__(self):  
        self.skin_model = pickle.load(open("user4/Skin_user4_Skin Temperature_SVR.sav", 'rb'))
    

    def skin_SVR(self, cur_Ta, cur_Rh):
        """
        A Support Vecotr Regression using 
        air temperature and air humidity 
        to predict skin temperature

        """
        
        _input =self.process_state(cur_Ta, cur_Rh)
        Ts = self.skin_model.predict([_input]).flatten()[0]
        result = Ts*(Ts_max - Ts_min) + Ts_min
        return result


    def process_state(self, cur_Ta, cur_Rh):
        cur_Ta = (cur_Ta - Ta_min)/(Ta_max - Ta_min)
        cur_Rh = (cur_Rh - Rh_min)/(Rh_max - Rh_min)
        return [cur_Ta, cur_Rh]


    def comfPierceSET(self, ta, tr, rh, clo, vel=0.1, met = 1.1, wme = 0, BODYWEIGHT = 69.9, BODYSURFACEAREA = 1.8258):
        """
        Function to find the saturation vapor pressure, used frequently
        throughtout the comfPierceSET function.

        Return
        ----------
        TempSkin: float, mean skin temperature of a human body

        """
        res = None

        def findSaturatedVaporPressureTorr(T):
            # calculates Saturated Vapor Pressure (Torr) at Temperature T  (C)
            return math.exp(18.6686 - 4030.183 / (T + 235.0))

        # Key initial variables.
        VaporPressure = (rh * findSaturatedVaporPressureTorr(ta)) / 100
        AirVelocity = max(vel, 0.1)
        KCLO = 0.25
       
        METFACTOR = 58.2
        SBC = 0.000000056697  # Stefan-Boltzmann constant (W/m2K4)
        CSW = 170
        CDIL = 120
        CSTR = 0.5

        TempSkinNeutral = 33.7  # setpoint (neutral) value for Tsk
        TempCoreNeutral = 36.49  # setpoint value for Tcr
        # setpoint for Tb (.1*TempSkinNeutral + .9*TempCoreNeutral)
        TempBodyNeutral = 36.49
        SkinBloodFlowNeutral = 6.3  # neutral value for SkinBloodFlow

        # INITIAL VALUES - start of 1st experiment
        TempSkin = TempSkinNeutral
        TempCore = TempCoreNeutral
        SkinBloodFlow = SkinBloodFlowNeutral
        MSHIV = 0.0
        ALFA = 0.1
        ESK = 0.1 * met

        # Start new experiment here (for graded experiments)
        # UNIT CONVERSIONS (from input variables)

        # This variable is the pressure of the atmosphere in kPa and was taken
        # from the psychrometrics.js file of the CBE comfort tool.
        p = 101325.0 / 1000

        PressureInAtmospheres = p * 0.009869
        LTIME = 60
        TIMEH = LTIME / 60.0
        RCL = 0.155 * clo

        FACL = 1.0 + 0.15 * clo  # INCREASE IN BODY SURFACE AREA DUE TO CLOTHING
        LR = 2.2 / PressureInAtmospheres  # Lewis Relation is 2.2 at sea level
        RM = met * METFACTOR
        M = met * METFACTOR

        if clo <= 0:
            WCRIT = 0.38 * pow(AirVelocity, -0.29)
            ICL = 1.0
        else:
            WCRIT = 0.59 * pow(AirVelocity, -0.08)
            ICL = 0.45

        CHC = 3.0 * pow(PressureInAtmospheres, 0.53)
        CHCV = 8.600001 * pow((AirVelocity * PressureInAtmospheres), 0.53)
        CHC = max(CHC, CHCV)

        # initial estimate of Tcl
        CHR = 4.7
        CTC = CHR + CHC
        RA = 1.0 / (FACL * CTC)  # resistance of air layer to dry heat transfer
        TOP = (CHR * tr + CHC * ta) / CTC
        TCL = TOP + (TempSkin - TOP) / (CTC * (RA + RCL))

        # ========================  BEGIN ITERATION
        #
        # Tcl and CHR are solved iteratively using: H(Tsk - To) = CTC(Tcl - To),
        # where H = 1/(Ra + Rcl) and Ra = 1/Facl*CTC

        TCL_OLD = TCL
        TIME = range(LTIME)
        flag = True
        for TIM in TIME:
            if flag == True:
                while abs(TCL - TCL_OLD) > 0.01:
                    TCL_OLD = TCL
                    CHR = 4.0 * SBC * pow(((TCL + tr) / 2.0 + 273.15), 3.0) * 0.72
                    CTC = CHR + CHC
                    # resistance of air layer to dry heat transfer
                    RA = 1.0 / (FACL * CTC)
                    TOP = (CHR * tr + CHC * ta) / CTC
                    TCL = (RA * TempSkin + RCL * TOP) / (RA + RCL)
            flag = False
            DRY = (TempSkin - TOP) / (RA + RCL)
            HFCS = (TempCore - TempSkin) * (5.28 + 1.163 * SkinBloodFlow)
            ERES = 0.0023 * M * (44.0 - VaporPressure)
            CRES = 0.0014 * M * (34.0 - ta)
            SCR = M - HFCS - ERES - CRES - wme
            SSK = HFCS - DRY - ESK
            TCSK = 0.97 * ALFA * BODYWEIGHT
            TCCR = 0.97 * (1 - ALFA) * BODYWEIGHT
            DTSK = (SSK * BODYSURFACEAREA) / (TCSK * 60.0)  # deg C per minute
            DTCR = SCR * BODYSURFACEAREA / (TCCR * 60.0)  # deg C per minute
            TempSkin = TempSkin + DTSK
            TempCore = TempCore + DTCR
            TB = ALFA * TempSkin + (1 - ALFA) * TempCore
            SKSIG = TempSkin - TempSkinNeutral
            WARMS = (SKSIG > 0) * SKSIG
            COLDS = ((-1.0 * SKSIG) > 0) * (-1.0 * SKSIG)
            CRSIG = (TempCore - TempCoreNeutral)
            WARMC = (CRSIG > 0) * CRSIG
            COLDC = ((-1.0 * CRSIG) > 0) * (-1.0 * CRSIG)
            BDSIG = TB - TempBodyNeutral
            WARMB = (BDSIG > 0) * BDSIG
            COLDB = ((-1.0 * BDSIG) > 0) * (-1.0 * BDSIG)
            SkinBloodFlow = ((SkinBloodFlowNeutral + CDIL * WARMC)
                / (1 + CSTR * COLDS))
            if SkinBloodFlow > 90.0: SkinBloodFlow = 90.0
            if SkinBloodFlow < 0.5: SkinBloodFlow = 0.5
            REGSW = CSW * WARMB * math.exp(WARMS / 10.7)
            if REGSW > 500.0: REGSW = 500.0
            ERSW = 0.68 * REGSW
            REA = 1.0 / (LR * FACL * CHC)  # evaporative resistance of air layer
            # evaporative resistance of clothing (icl=.45)
            RECL = RCL / (LR * ICL)
            EMAX = ((findSaturatedVaporPressureTorr(TempSkin) - VaporPressure) /
                (REA + RECL))
            PRSW = ERSW / EMAX
            PWET = 0.06 + 0.94 * PRSW
            EDIF = PWET * EMAX - ERSW
            ESK = ERSW + EDIF
            if PWET > WCRIT:
                PWET = WCRIT
                PRSW = WCRIT / 0.94
                ERSW = PRSW * EMAX
                EDIF = 0.06 * (1.0 - PRSW) * EMAX
                ESK = ERSW + EDIF
            if EMAX < 0:
                EDIF = 0
                ERSW = 0
                PWET = WCRIT
                PRSW = WCRIT
                ESK = EMAX
            ESK = ERSW + EDIF
            MSHIV = 19.4 * COLDS * COLDC
            M = RM + MSHIV
            ALFA = 0.0417737 + 0.7451833 / (SkinBloodFlow + .585417)


        # Define new heat flow terms, coeffs, and abbreviations
        STORE = M - wme - CRES - ERES - DRY - ESK  # rate of body heat storage
        HSK = DRY + ESK  # total heat loss from skin
        RN = M - wme  # net metabolic heat production
        ECOMF = 0.42 * (RN - (1 * METFACTOR))
        if ECOMF < 0.0: ECOMF = 0.0  # from Fanger
        EREQ = RN - ERES - CRES - DRY
        EMAX = EMAX * WCRIT
        HD = 1.0 / (RA + RCL)
        HE = 1.0 / (REA + RECL)
        W = PWET
        PSSK = findSaturatedVaporPressureTorr(TempSkin)
        # Definition of ASHRAE standard environment... denoted "S"
        CHRS = CHR
        if met < 0.85:
            CHCS = 3.0
        else:
            CHCS = 5.66 * pow((met - 0.85), 0.39)
            if CHCS < 3.0: CHCS = 3.0

        CTCS = CHCS + CHRS
        RCLOS = 1.52 / ((met - wme / METFACTOR) + 0.6944) - 0.1835
        RCLS = 0.155 * RCLOS
        FACLS = 1.0 + KCLO * RCLOS
        FCLS = 1.0 / (1.0 + 0.155 * FACLS * CTCS * RCLOS)
        IMS = 0.45
        ICLS = IMS * CHCS / CTCS * (1 - FCLS) / (CHCS / CTCS - FCLS * IMS)
        RAS = 1.0 / (FACLS * CTCS)
        REAS = 1.0 / (LR * FACLS * CHCS)
        RECLS = RCLS / (LR * ICLS)
        HD_S = 1.0 / (RAS + RCLS)
        HE_S = 1.0 / (REAS + RECLS)

        # SET* (standardized humidity, clo, Pb, and CHC)
        # determined using Newton's iterative solution
        # FNERRS is defined in the GENERAL SETUP section above

        DELTA = .0001
        dx = 100.0
        X_OLD = TempSkin - HSK / HD_S  # lower bound for SET
        while abs(dx) > .01:
            ERR1 = (HSK - HD_S * (TempSkin - X_OLD) - W * HE_S
                * (PSSK - 0.5 * findSaturatedVaporPressureTorr(X_OLD)))
            ERR2 = (HSK - HD_S * (TempSkin - (X_OLD + DELTA)) - W * HE_S
                * (PSSK - 0.5 * findSaturatedVaporPressureTorr((X_OLD + DELTA))))
            X = X_OLD - DELTA * ERR1 / (ERR2 - ERR1)
            dx = X - X_OLD
            X_OLD = X
        return TempSkin



class feedback():
    def __init__(self):  
        self.model = Sequential()
        self.model.add(Dense(3, input_dim=3, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(20, activation='relu'))
        self.model.add(Dense(7, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                  optimizer='adam', 
                  metrics=['accuracy'])
        self.model.load_weights('user4/user4_PMV_Real_Satisfaction_ANN.h5')


    def Satisfaction_neural(self, cur_Ts, cur_Ta, cur_Rh):
        X = self.process_state(cur_Ts, cur_Ta, cur_Rh).reshape(-1, 3)
        min_lable = -3
        classes = self.model.predict(X)
        result = [np.argmax(values) + min_lable for values in classes][0]
        return result

    def process_state(self, cur_Ts, cur_Ta, cur_Rh):
        cur_Ts = (cur_Ts - Ts_min)/(Ts_max - Ts_min)
        cur_Ta = (cur_Ta - Ta_min)/(Ta_max - Ta_min)
        cur_Rh = (cur_Rh - Rh_min)/(Rh_max - Rh_min)
        # pre_Ts = (pre_Ts - Ts_min)/(Ts_max - Ts_min)
        # pre_Ta = (pre_Ta - Ta_min)/(Ta_max - Ta_min)
        # pre_Rh = (pre_Rh - Rh_min)/(Rh_max - Rh_min)
        return np.array([cur_Ts, cur_Ta, cur_Rh])


    def comfPMV(self, ta, tr, rh,  clo, met =1.1, vel=0.1, wme = 0):
        """
        ref:https://github.com/CenterForTheBuiltEnvironment/comfort_tool/blob/master/contrib/comfort_models.py
        returns [pmv, ppd]
        ta, air temperature (C), must be float, e.g. 25.0
        tr, mean radiant temperature (C),must be float, e.g. 25.0
        vel, relative air velocity (m/s), must be float, e.g. 25.0
        rh, relative humidity (%) Used only this way to input humidity level
        met, metabolic rate (met)
        clo, clothing (clo)
        wme, external work, normally around 0 (met)
        """

        pa = rh * 10 * math.exp(16.6536 - 4030.183 / (ta + 235))

        icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
        m = met * 58.15  # metabolic rate in W/M2
        w = wme * 58.15  # external work in W/M2
        mw = m - w  # internal heat production in the human body
        if (icl <= 0.078):
            fcl = 1 + (1.29 * icl)
        else:
            fcl = 1.05 + (0.645 * icl)

        # heat transf. coeff. by forced convection
        hcf = 12.1 * math.sqrt(vel)
        taa = ta + 273
        tra = tr + 273
        tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)

        p1 = icl * fcl
        p2 = p1 * 3.96
        p3 = p1 * 100
        p4 = p1 * taa
        p5 = (308.7 - 0.028 * mw) + (p2 * math.pow(tra / 100, 4))
        xn = tcla / 100
        xf = tcla / 50
        eps = 0.00015

        n = 0
        while abs(xn - xf) > eps:
            xf = (xf + xn) / 2
            hcn = 2.38 * math.pow(abs(100.0 * xf - taa), 0.25)
            if (hcf > hcn):
                hc = hcf
            else:
                hc = hcn
            xn = (p5 + p4 * hc - p2 * math.pow(xf, 4)) / (100 + p3 * hc)
            n += 1
            if (n > 150):
                print ('Max iterations exceeded')
                return 1


        tcl = 100 * xn - 273

        # heat loss diff. through skin
        hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
        # heat loss by sweating
        if mw > 58.15:
            hl2 = 0.42 * (mw - 58.15)
        else:
            hl2 = 0
        # latent respiration heat loss
        hl3 = 1.7 * 0.00001 * m * (5867 - pa)
        # dry respiration heat loss
        hl4 = 0.0014 * m * (34 - ta)
        # heat loss by radiation
        hl5 = 3.96 * fcl * (math.pow(xn, 4) - math.pow(tra / 100, 4))
        # heat loss by convection
        hl6 = fcl * hc * (tcl - ta)

        ts = 0.303 * math.exp(-0.036 * m) + 0.028
        pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
        ppd = 100.0 - 95.0 * math.exp(-0.03353 * pow(pmv, 4.0)
            - 0.2179 * pow(pmv, 2.0))

        r = []
        r.append(pmv)
        r.append(ppd)

        return r

# outdoor = outdoorSet()
# temp, humid = outdoor.get_out(20)
# air = airEnviroment()
# skin = skinTemperature()
# vote = feedback()
# print(temp)
# print(air.get_air_temp(3, 20, temp)) # not reasonable
# print(air.get_air_humidity(20, humid))
# print(skin.skin_SVR(25.2, 30.69))
# print(vote.Satisfaction_neural(32.33, 25.2, 38.69))