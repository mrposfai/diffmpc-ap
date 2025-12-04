import numpy as np
from numba import njit
import OPT_GLOBALS

@njit("float64[:](float64[:],float64[:],float64[:])", cache=True)
def model(states, inputs, parameters):

    NOMINAL_FR_threshold = 9.0
    NOMINAL_F01_threshold = 4.5

    """ Define Parameters """
    (kb1, kb2, kb3, EGP0, ke, F01, AG, tmaxI, tmaxG, p3, p4, p5, beta, a, BW, HRrest, HRmax, VG, VI,
    VT_HRR, k12, ka1, ka2, ka3, aSI, b, c, dSI, tsub, p1, p2, p6, p7, p8, tmaxGFast) = parameters.T

    # Hard lower and upper constraint on Q1
    states[6] = np.max(np.array([states[6],VG]))
    states[6] = np.min(np.array([states[6],VG*32]))

    """ Define State variables """
    S1, S2, I, x1, x2, x3, Q1, Q2, IG, D1Slow, D2Slow, D1Fast, D2Fast = states.T

    # Nonlinear insulin sensitivity mapping
    SIModelRatioHypo = (aSI * np.tanh(Q1 / VG / b + dSI) + c) / (aSI * np.tanh(5 / b + dSI) + c)
    SIModelRatioHyper = 1 - 0.018 * (Q1 / VG - 5.55)

    """ Overwriting parameters"""
    kb1_nonlin = kb1 * SIModelRatioHypo * SIModelRatioHyper
    kb2_nonlin = kb2 * SIModelRatioHypo * SIModelRatioHyper
    kb3_nonlin = kb3 * SIModelRatioHypo * SIModelRatioHyper


    """ Define Inputs """
    uFastCarbs, uSlowCarbs, uHR, uInsulin, _ = inputs.T

    
    # Insulin System
    dS1dt = uInsulin - S1 / tmaxI
    dS2dt = (S1 - S2) / tmaxI
    dIdt = S2 / (VI * tmaxI) - ke * I

    # Insulin Action System
    dx1dt = I * kb1_nonlin - x1 * ka1
    dx2dt = I * kb2_nonlin - x2 * ka2
    dx3dt = I * kb3_nonlin - x3 * ka3

    # Glucose compartment
    # Nervous system glucose uptake
    F01_nonlin = BW * F01
    if (Q1 / VG) < NOMINAL_F01_threshold:
        F01_nonlin = F01_nonlin * (Q1 / VG) / NOMINAL_F01_threshold

    # Renal glucose clearance
    FR = 0.0
    if (Q1 / VG) >= NOMINAL_FR_threshold:
        FR = 0.003 * ((Q1 / VG) - NOMINAL_FR_threshold) * VG

    # Gut glucose absorption
    UGe = (D2Slow / tmaxG + D2Fast / tmaxGFast)

    # Endogenuous glucose production meditated by insulin and physical activity
    EGP_contribution = EGP0 * (1 - x3)
    EGP_contribution = np.max(np.array([EGP_contribution,0.0]))
    #EGP_contribution[EGP_contribution < 0] = 0

    # Glucose compartments
    dQ1dt = (-Q1 * x1 + k12 * Q2 - F01_nonlin - FR + UGe + EGP_contribution)
    dQ2dt = (Q1 * x1 - k12 * Q2 - x2 * Q2)

    dIGdt = ((1 / tsub) * ((Q1 / VG) - IG))

    # Slow carbohydrate absorption system
    dD1Slowdt = (AG * (uSlowCarbs) - (D1Slow / tmaxG))
    dD2Slowdt = ((D1Slow - D2Slow) / tmaxG)

    # Fast carbohydrate absorption system
    dD1Fastdt = (AG * (uFastCarbs) - (D1Fast / tmaxGFast))
    dD2Fastdt = ((D1Fast - D2Fast) / tmaxGFast)

    next_state = states + np.array((dS1dt, dS2dt, dIdt, dx1dt, dx2dt, dx3dt, dQ1dt, dQ2dt, dIGdt, dD1Slowdt, dD2Slowdt,dD1Fastdt, dD2Fastdt))

    return next_state

@njit("float64(float64[:],float64[:],float64[:,:],float64[:])", cache=False)
def run_mpc(insulin, parameters, inputs, init_state):
    (kb1, kb2, kb3, EGP0, ke, F01, AG, tmaxI, tmaxG, p3, p4, p5, beta, a, BW, HRrest, HRmax, VG, VI,
    VT_HRR, k12, ka1, ka2, ka3, aSI, b, c, dSI, tsub, p1, p2, p6, p7, p8, tmaxGFast) = parameters.T
    inputs[:,3] = np.repeat(insulin,OPT_GLOBALS.TS)
    states = np.zeros((inputs.shape[0],init_state.shape[0]))
    states[0] = init_state
    for i in range(inputs.shape[0]-1,):
        states[i+1] = model(states=states[i],inputs=inputs[i],parameters=parameters)

    return np.sum(np.power(states[:,6]/VG-OPT_GLOBALS.SETPOINT/OPT_GLOBALS.MMOL_TO_MGDL,2))+OPT_GLOBALS.RU*np.sum(np.power(insulin,2))