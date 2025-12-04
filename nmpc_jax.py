import jax
import jax.numpy as jnp
import OPT_GLOBALS
import numpy as np

@jax.jit
def model_jax(states, inputs, parameters):

    NOMINAL_FR_threshold = 9.0
    NOMINAL_F01_threshold = 4.5


    (kb1, kb2, kb3, EGP0, ke, F01, AG, tmaxI, tmaxG,
    p3, p4, p5, beta, a, BW, HRrest, HRmax,
    VG, VI, VT_HRR,
    k12, ka1, ka2, ka3, aSI, b, c, dSI,
    tsub, p1, p2, p6, p7, p8, tmaxGFast) = parameters


    # Hard lower and upper constraint on Q1
    states = states.at[6].set(jax.lax.select(states[6]>VG*32.0,VG*32.0,states[6]))
    states = states.at[6].set(jax.lax.select(states[6]<VG,VG,states[6]))


    """ Define State variables """
    S1, S2, I, x1, x2, x3, Q1, Q2, IG, D1Slow, D2Slow, D1Fast, D2Fast = states


    # Nonlinear insulin sensitivity mapping
    SIModelRatioHypo = (aSI * jnp.tanh(Q1 / VG / b + dSI) + c) / (aSI * jnp.tanh(5 / b + dSI) + c)
    SIModelRatioHyper = 1 - 0.018 * (Q1 / VG - 5.55)

    """ Overwriting parameters"""
    kb1_nonlin = kb1 * SIModelRatioHypo * SIModelRatioHyper
    kb2_nonlin = kb2 * SIModelRatioHypo * SIModelRatioHyper
    kb3_nonlin = kb3 * SIModelRatioHypo * SIModelRatioHyper


    """ Define Inputs """
    uFastCarbs, uSlowCarbs, uHR, uInsulin, _ = inputs

    
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
    hypo_binmap = (Q1 / VG) < NOMINAL_F01_threshold
    F01_nonlin = jax.lax.select(hypo_binmap, F01_nonlin*(Q1 / VG) / NOMINAL_F01_threshold, F01_nonlin)

    # Renal glucose clearance
    FR = 0.0
    FR_binmap = (Q1 / VG) >= NOMINAL_FR_threshold
    FR = jax.lax.select(FR_binmap, 0.003 * ((Q1 / VG) - NOMINAL_FR_threshold) * VG, 0.0)

    # Gut glucose absorption
    UGe = (D2Slow / tmaxG + D2Fast / tmaxGFast)

    # Endogenuous glucose production meditated by insulin and physical activity
    EGP_contribution = EGP0 * (1.0 - x3)
    EGP_contribution = jax.lax.select(EGP_contribution<0.0,0.0,EGP_contribution)
    #EGP_contribution[EGP_contribution < 0] = 0

    # Glucose compartments
    dQ1dt = (-Q1 * x1 + k12 * Q2 - F01_nonlin - FR + UGe + EGP_contribution)
    dQ2dt = (Q1 * x1 - k12 * Q2 - x2 * Q2)

    dIGdt = ((1.0 / tsub) * ((Q1 / VG) - IG))

    # Slow carbohydrate absorption system
    dD1Slowdt = (AG * (uSlowCarbs) - (D1Slow / tmaxG))
    dD2Slowdt = ((D1Slow - D2Slow) / tmaxG)

    # Fast carbohydrate absorption system
    dD1Fastdt = (AG * (uFastCarbs) - (D1Fast / tmaxGFast))
    dD2Fastdt = ((D1Fast - D2Fast) / tmaxGFast)

    return jnp.array((dS1dt, dS2dt, dIdt, dx1dt, dx2dt, dx3dt, dQ1dt, dQ2dt, dIGdt, dD1Slowdt, dD2Slowdt,dD1Fastdt, dD2Fastdt))

@jax.jit
def model_jax_mpc(states, inputs, parameters):

    NOMINAL_FR_threshold = 9.0
    NOMINAL_F01_threshold = 4.5


    (kb1, kb2, kb3, EGP0, ke, F01, AG, tmaxI, tmaxG,
    p3, p4, p5, beta, a, BW, HRrest, HRmax,
    VG, VI, VT_HRR,
    k12, ka1, ka2, ka3, aSI, b, c, dSI,
    tsub, p1, p2, p6, p7, p8, tmaxGFast) = parameters


    # Hard lower and upper constraint on Q1
    states = states.at[6].set(jax.lax.select(states[6]>VG*32.0,VG*32.0,states[6]))
    states = states.at[6].set(jax.lax.select(states[6]<VG,VG,states[6]))

    """ Define State variables """
    S1, S2, I, Q1, Q2, E1, E2, x1, x2, x3, D1Slow, D2Slow = states


    # Nonlinear insulin sensitivity mapping
    SIModelRatioHypo = (aSI * jnp.tanh(Q1 / VG / b + dSI) + c) / (aSI * jnp.tanh(5 / b + dSI) + c)
    SIModelRatioHyper = 1.0 - 0.018 * (Q1 / VG - 5.55)

    """ Overwriting parameters"""
    kb1_nonlin = kb1 * SIModelRatioHypo * SIModelRatioHyper
    kb2_nonlin = kb2 * SIModelRatioHypo * SIModelRatioHyper
    kb3_nonlin = kb3 * SIModelRatioHypo * SIModelRatioHyper


    """ Define Inputs """
    uInsulin, uSlowCarbs, _ = inputs

    
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
    hypo_binmap = (Q1 / VG) < NOMINAL_F01_threshold
    F01_nonlin = jax.lax.select(hypo_binmap, F01_nonlin*(Q1 / VG) / NOMINAL_F01_threshold, F01_nonlin)

    # Renal glucose clearance
    FR = 0.0
    FR_binmap = (Q1 / VG) >= NOMINAL_FR_threshold
    FR = jax.lax.select(FR_binmap, 0.003 * ((Q1 / VG) - NOMINAL_FR_threshold) * VG, 0.0)

    # Gut glucose absorption
    UGe = (D2Slow / tmaxG )

    # Endogenuous glucose production meditated by insulin and physical activity
    EGP_contribution = EGP0 * (1.0 - x3)
    EGP_contribution = jax.lax.select(EGP_contribution<0.0,0.0,EGP_contribution)
    #EGP_contribution[EGP_contribution < 0] = 0

    # Glucose compartments
    dQ1dt = (-Q1 * x1 + k12 * Q2 - F01_nonlin - FR + UGe + EGP_contribution)
    dQ2dt = (Q1 * x1 - k12 * Q2 - x2 * Q2)


    # Slow carbohydrate absorption system
    dD1Slowdt = (AG * (uSlowCarbs) - (D1Slow / tmaxG))
    dD2Slowdt = ((D1Slow - D2Slow) / tmaxG)

    # Fast carbohydrate absorption system

    return jnp.array((dS1dt, dS2dt, dIdt, dQ1dt, dQ2dt, 0.0, 0.0, dx1dt, dx2dt, dx3dt, dD1Slowdt, dD2Slowdt))


@jax.jit
def model_jax2(states, inputs, parameters):
    next_state = states + model_jax(states, inputs, parameters)
    return next_state, next_state

@jax.jit
def run_mpc_jax(insulin, parameters, inputs, init_state):

    inputs = inputs.at[3].set(jnp.repeat(insulin,OPT_GLOBALS.TS))
    _, states = jax.lax.scan(jax.tree_util.Partial(model_jax2, parameters=parameters),init_state, inputs.T)
    with_init = jnp.concatenate((init_state[None,:],states[:-1]),0)

    return jnp.sum(jnp.power(with_init[:,6]/parameters[17]-OPT_GLOBALS.SETPOINT/OPT_GLOBALS.MMOL_TO_MGDL,2))+OPT_GLOBALS.RU*jnp.sum(jnp.power(insulin,2))


def linear_model_update_jax(states, inputs, A, B):
    next_state = jnp.matmul(A,states[:,None])[:,0] + jnp.matmul(B,inputs[:,None])[:,0]
    return next_state, next_state

@jax.jit
def run_lmpc_jax(insulin, parameters, inputs, init_state, A, B):

    inputs = inputs.at[0].set(insulin)
    _, states = jax.lax.scan(jax.tree_util.Partial(linear_model_update_jax, A=A, B=B),jnp.zeros(12,), inputs.T)
    with_init = jnp.concatenate((init_state[None,:],init_state[None,:]+states[:-1]),0)

    return jnp.sum(jnp.power(with_init[:,3]/parameters[17]-OPT_GLOBALS.SETPOINT/OPT_GLOBALS.MMOL_TO_MGDL,2))+OPT_GLOBALS.RU*jnp.sum(jnp.power(insulin,2))