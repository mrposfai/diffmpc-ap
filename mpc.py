import numpy as np
import qpsolvers
import jax
from jax import numpy as jnp
from nmpc_jax import model_jax_mpc
import time
import OPT_GLOBALS

def build_mpc_qp_multistep(linear_model, N=36,cgm=0.0,VG=0.0):

    x0 = np.zeros_like(linear_model.x0)
    A = np.asarray(linear_model.get_A())
    B = np.asarray(linear_model.get_B_artificial())
    #print(B)
    C = np.asarray(linear_model.get_C())
    #print("BUILD QP")
    #print(C)
    D = np.zeros((6,4))
    Qy = 1.0
    Ru = OPT_GLOBALS.RU

    x0 = np.atleast_2d(x0).reshape(-1, 1)       # (nx,1)
    #print(x0)
    #print(cgm)
    y_sp = np.atleast_2d((OPT_GLOBALS.SETPOINT - cgm)/OPT_GLOBALS.MMOL_TO_MGDL).reshape(-1, 1)   # (ny,1)

    opt_indices=0
    opt_indices = np.atleast_1d(opt_indices)
    # Extract the columns corresponding to the optimized inputs
    B_opt = B[:, opt_indices]  # (nx, nu_opt)
    B_unopt = B[:, 1:]

    nx = A.shape[0]
    ny = C.shape[0]
    nu_opt = B_opt.shape[1]
    nu_unopt = 3

    # --- Build lifted prediction matrices for optimized inputs only:
    #     Y = Fx * x0 + Fu_opt * U_opt
    Fx = np.zeros((N * ny, nx))
    Fu_opt = np.zeros((N * ny, N * nu_opt))
    Fu_unopt = np.zeros((N * ny, N * nu_unopt))

    # Precompute A^k
    A_powers = [np.eye(nx)]
    for k in range(1, N):
        A_powers.append(A_powers[-1] @ A)

    for k in range(N):
        row_start = k * ny
        row_end = (k + 1) * ny

        # effect of initial state
        Fx[row_start:row_end, :] = C @ A_powers[k]

        # effect of optimized inputs at steps 0..k
        for j in range(k):
            col_start = j * nu_opt
            col_end = (j + 1) * nu_opt
            Fu_opt[row_start:row_end, col_start:col_end] = (
                C @ (A_powers[k - 1 - j] @ B_opt)
            )

        for j in range(k):
            col_start = j * nu_unopt
            col_end = (j + 1) * nu_unopt
            Fu_unopt[row_start:row_end, col_start:col_end] = (
                C @ (A_powers[k - 1 - j] @ B_unopt )
            )


    # --- Block-diagonal weights over horizon
    Qbar = np.kron(np.eye(N), Qy)    # (N*ny, N*ny)
    Rbar = np.kron(np.eye(N), Ru)    # (N*nu_opt, N*nu_opt)

    # --- Stack setpoint over horizon
    Ysp = np.tile(y_sp, (N, 1))      # (N*ny, 1)

    # --- Cost: 0.5 * U_opt^T P U_opt + q^T U_opt

    constant_inputs = np.zeros((N*3,1))
    constant_inputs[2::3,0] = 1.0

    Fx_x0_minus_Ysp =  Fu_unopt @ constant_inputs - Ysp
    #Fx_x0_minus_Ysp = Fx @ x0 - Ysp
    sampler = np.repeat(np.eye(OPT_GLOBALS.PRED_STEPS),OPT_GLOBALS.TS,axis=0)
    P = sampler.T@(Fu_opt.T @ Qbar @ Fu_opt + Rbar)@sampler
    q = (sampler.T@(Fu_opt.T @ Qbar @ Fx_x0_minus_Ysp)).reshape(-1)

    # --- No inequality constraints
    G = None
    h = None

    # --- No equality constraints
    Aeq = None
    beq = None

    # --- Box bounds only on optimized inputs
    u_min_opt = 0.0
    u_max_opt = OPT_GLOBALS.MAX_RATE
    u_min_opt = np.asarray(u_min_opt).reshape(-1)
    u_max_opt = np.asarray(u_max_opt).reshape(-1)

    lb = np.tile(u_min_opt, OPT_GLOBALS.PRED_STEPS)   # (N*nu_opt,)
    ub = np.tile(u_max_opt, OPT_GLOBALS.PRED_STEPS)   # (N*nu_opt,)

    return P, q, G, h, Aeq, beq, lb, ub

class Bounds:

    def __init__(self, u_min=None, u_max=None, x_min=None, x_max=None, y_min=None, y_max=None, du_min=None, du_max=None):
        self.u_min = u_min
        self.u_max = u_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.du_min = du_min
        self.du_max = du_max


class MPC_Objective_Function_Params:

    def __init__(self, prediction_horizon=None, Q_x=None, Q_e=None, Q_u=None, Q_du=None, Q_PIC=None, PIC_pos=None, ysp=None):
        self.prediction_horizon = prediction_horizon
        self.Q_x = Q_x
        self.Q_e = Q_e
        self.Q_u = Q_u
        self.Q_du = Q_du
        self.Q_PIC = Q_PIC
        self.PIC_pos = PIC_pos
        self.ysp = ysp


class Linear_State_Space_Model:
    # T is the sample time which by default is set to 5 min
    def __init__(self, A=None, B=None, C=None, D=None, x0=None, u0=None, f0=None, T=5):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.x0 = x0
        self.u0 = u0
        self.f0 = f0
        self.T = T

    def get_state_num(self):
        return self.A.shape[0]

    def get_input_num(self):
        return self.B.shape[1]

    def get_input_num_artificial(self):
        return self.B.shape[1] + 1

    def get_output_num(self):
        return self.C.shape[0]

    def get_A(self):
        return self.A

    def get_B(self):
        return self.B

    def get_B_artificial(self):
        return np.concatenate((self.B, np.reshape(self.T * self.f0, (-1, 1))), axis=1)

    def get_C(self):
        return self.C

    def get_D(self):
        return self.D

def get_hovorka_basal_equilibrium(parameters, basal_blucose):
        """ Define Parameters """
        (kb1, kb2, kb3, EGP0, ke, F01, AG, tmaxI, tmaxG, p3, p4, p5, beta, a, BW, HRrest, HRmax, VG, VI,
        VT_HRR, k12, ka1, ka2, ka3, aSI, b, c, dSI, tsub, p1, p2, p6, p7, p8, tmaxGFast) = parameters.T

        F01_nonlin = BW * F01

        G = basal_blucose

        SIModelRatioHypo = (aSI * np.tanh(G / b + dSI) + c) / (aSI * np.tanh(5 / b + dSI) + c)
        SIModelRatioHyper = 1 - 0.018 * (G - 5.55)

        """ Overwriting parameters"""
        kb1nonlin = kb1 * SIModelRatioHypo * SIModelRatioHyper
        kb2nonlin = kb2 * SIModelRatioHypo * SIModelRatioHyper
        kb3nonlin = kb3 * SIModelRatioHypo * SIModelRatioHyper

        basal_equilibrium = (VI*ka1**(1/2)*ke*(ka1*EGP0**2*k12**2*ka2**2*kb3nonlin**2
                                               + 2*ka1*EGP0**2*k12*ka2*ka3*kb2nonlin*kb3nonlin
                                               + ka1*EGP0**2*ka3**2*kb2nonlin**2
                                               - 2*ka1*EGP0*F01_nonlin*k12*ka2*ka3*kb2nonlin*kb3nonlin
                                               - 2*ka1*EGP0*F01_nonlin*ka3**2*kb2nonlin**2
                                               + 4*G*VG*kb1nonlin*EGP0*k12*ka2*ka3**2*kb2nonlin
                                               + ka1*F01_nonlin**2*ka3**2*kb2nonlin**2
                                               - 4*G*VG*kb1nonlin*F01_nonlin*k12*ka2*ka3**2*kb2nonlin)**(1/2)
                             + EGP0*VI*ka1*ka3*kb2nonlin*ke - F01_nonlin*VI*ka1*ka3*kb2nonlin*ke
                             - EGP0*VI*k12*ka1*ka2*kb3nonlin*ke)/(2*(EGP0*ka1*kb2nonlin*kb3nonlin + G*VG*ka3*kb1nonlin*kb2nonlin))

        return basal_equilibrium

def get_hovorka_linear_model(x0, u0, curr_cgm, parameters, T=5):
    
    # u is the input vector, [insulin, meal, energy_expenditure]
    A, B, C, D = state_space_initializer(x0, parameters, T)
    #f0 = get_hovorka_model_f0(x0, u0, parameters)
    f0 = np.array(model_jax_mpc(jnp.array(x0),u0,parameters))

    linear_model = Linear_State_Space_Model(A=A, B=B, C=C, D=D, x0=x0, u0=u0, f0=f0, T=T)
    linear_model.VG = parameters[17]

    return linear_model


def state_space_initializer(x0, parameters, T=5):
    # T is the sampling rate of CGM values in minutes

    (kb1, kb2, kb3, EGP0, Ke, F01, AG, TmaxI, TmaxG, p3, p4, p5, beta, a, BW, HRrest, HRmax, VG, VI,
    VT_HRR, K12, ka1, ka2, ka3, aSI, b, c, dSI, tsub, p1, p2, p6, p7, p8, tmaxGFast) = parameters.T

    gain = 1
    TmaxE = 1
    Ag = 1

    #S1, S2, I, x1, x2, x3, Q1, Q2, IG, D1Slow, D2Slow, D1Fast, D2Fast = x0.T
    (S1,S2,I,Q1,Q2,_,_,x1,x2,x3,D1Slow,D2Slow) = x0
    R2 = 0

    G = Q1 / VG
    sw1 = 1 if G >= 4.5 else 0
    sw2 = 1 if G >= 9.0 else 0

    # p = [0.0053, 0.0257] # 2 parameter exercise model
    p = [0, 0.1253]  # 1 parameter exercise model
    alpha = p[0]
    beta = p[1]

    x0_mpc = jnp.array(x0)
    #Ac = jax.jacobian(model_jax_mpc,argnums=0)(x0_mpc,jnp.zeros((5,)),parameters)

    Ac = np.array(jax.jacobian(model_jax_mpc,argnums=0)(x0_mpc,jnp.zeros((3,)),parameters))
    Bc = np.array(jax.jacobian(model_jax_mpc,argnums=1)(x0_mpc,jnp.zeros((3,)),parameters))

    num_states = Ac.shape[0]
    num_inputs = Bc.shape[1]
    output_state_pos = 4

    A = np.eye(num_states) + T * Ac
    B = T * Bc
    C = np.array([np.concatenate((np.zeros(output_state_pos - 1), [1.0/VG], np.zeros(num_states - output_state_pos)))])
    D = np.zeros((1, num_inputs))

    return A, B, C, D


def mpc_execute(linear_model: Linear_State_Space_Model, parameters, model_states, cgm, basal_rate, inputs):

    # the first and second elements in inputs are carb and the fourth element is insulin
    last_input = np.zeros(3,)#[inputs[0][3], inputs[0][0] + inputs[0][1], 1]

    # basal rate is in mU/min

    # we do not consider exercise so MET is always 1
    # last input = [insulin, meal, 1]

    # cgm is a vector containing the last 6 cgm values

    n = 12
    m = 4
    p = 1
    T = 5

    # parameters
    (kb1, kb2, kb3, EGP0, Ke, F01, AG, TmaxI, TmaxG, p3, p4, p5, beta, a, BW, HRrest, HRmax, VG, VI,
    VT_HRR, K12, ka1, ka2, ka3, aSI, b, c, dSI, tsub, p1, p2, p6, p7, p8, tmaxGFast) = parameters.T

    # states
    #S1, S2, I, x1, x2, x3, Q1, Q2, IG, D1Slow, D2Slow, D1Fast, D2Fast = model_states.T
    S1,S2,I,Q1,Q2,_,_,x1,x2,x3,D1Slow,D2Slow = model_states

    PIC = I

    current_cgm = cgm
    #print("Current_cgm:"+str(current_cgm))

    ysp_mg_dl = OPT_GLOBALS.SETPOINT  # mg/dl
    ysp_deviation_mmol_l = (ysp_mg_dl - current_cgm) / OPT_GLOBALS.MMOL_TO_MGDL  # mmol/l

    risk_index = 1

    PIC_basal = basal_rate / (VI * Ke)
    gamma = PIC / PIC_basal

    bounds = Bounds()

    R0 = OPT_GLOBALS.RU
    Q0 = 1.0
    Q_x = np.zeros((n, n))
    Q_e = Q0 * risk_index
    Q_u = np.diag([R0, 0, 0, 0])
    Q_du = np.diag([0, 0, 0, 0])
    Q_PIC = np.diag([0, 0])

    PIC_pos = np.concatenate(([0, 0, 1], np.zeros(9)))
    mpc_obj_fcn_params = MPC_Objective_Function_Params(prediction_horizon=OPT_GLOBALS.PRED_LEN, Q_x=Q_x, Q_e=Q_e, Q_u=Q_u, Q_du=Q_du, Q_PIC=Q_PIC, PIC_pos=PIC_pos, ysp=ysp_deviation_mmol_l)

    # setting bounds
    bounds.x_min = np.full(n, -np.inf)
    bounds.x_max = np.full(n, np.inf)
    bounds.y_min = np.full(p, -np.inf)
    bounds.y_max = np.full(p, np.inf)
    bounds.du_min = np.full(m, -np.inf)
    bounds.du_max = np.full(m, np.inf)

    bounds.u_min = np.zeros((m, OPT_GLOBALS.PRED_LEN))
    bounds.u_max = np.zeros((m, OPT_GLOBALS.PRED_LEN))

    bounds.u_min[0, :] = 0.0
    bounds.u_max[0, :] = OPT_GLOBALS.MAX_RATE

    if last_input[0] < basal_rate:
        last_basal_rate = last_input[0]
    else:
        last_basal_rate = basal_rate

    bounds.u_min[1, :] = bounds.u_max[1, :] = 0.0
    bounds.u_min[2, :] = bounds.u_max[2, :] = 0.0

    bounds.u_min[3, :] = bounds.u_max[3, :] = 1.0

    H, f, A, b, Aeq, beq, lb, ub = mpc_set(linear_model=linear_model, mpc_obj_fcn_params=mpc_obj_fcn_params, bounds=bounds)
    init_vals = bounds.u_min+(bounds.u_max-bounds.u_min)/2.0
    init_vals[0,:] = 0.0#basal_rate
    t_start_lin = time.time()
    P2, q2, G2, h2, Aeq2, beq2, lb2, ub2 = build_mpc_qp_multistep(linear_model,OPT_GLOBALS.PRED_LEN,cgm,VG)
    x_solution2 = qpsolvers.solve_qp(P=P2, q=q2, G=G2, h=h2, lb=lb2, ub=ub2, solver='clarabel', initvals=np.zeros(OPT_GLOBALS.PRED_STEPS,),verbose=False)
    t_end_lin = time.time()
    #x_solution = qpsolvers.solve_qp(P=H, q=f, G=A, h=b, A=Aeq, b=beq, lb=lb, ub=ub, solver='clarabel', initvals=init_vals,verbose=False)
    x_solution = None
    #print("Shape:"+str(x_solution.shape))

    # check if the qpsolver exit flag is successful, if not continue with the last calculated basal
    if x_solution is None and x_solution2 is None:
        return None, None, None, -1
    if x_solution is not None and x_solution2 is None:
        sample_time_decision_var_num = m + p + p + 6 + n + m +m
        u_opt = x_solution.reshape((OPT_GLOBALS.PRED_LEN, sample_time_decision_var_num))[:, -8:-5].T
        u_horizon = u_opt + np.array([
            last_input[0],
            last_input[1],
            last_input[2]
        ]).reshape(-1, 1)
        opt_insulin_trajectory = u_horizon[0, :]
        return opt_insulin_trajectory, x_solution.reshape((OPT_GLOBALS.PRED_LEN, sample_time_decision_var_num)), None, -1
    if x_solution is None and x_solution2 is not None:
        return None, None, x_solution2, t_end_lin - t_start_lin
    if x_solution is not None and x_solution2 is not None:
        sample_time_decision_var_num = m + p + p + 6 + n + m +m
        u_opt = x_solution.reshape((OPT_GLOBALS.PRED_LEN, sample_time_decision_var_num))[:, -8:-5].T
        u_horizon = u_opt + np.array([
            last_input[0],
            last_input[1],
            last_input[2]
        ]).reshape(-1, 1)
        opt_insulin_trajectory = u_horizon[0, :]
        return opt_insulin_trajectory, x_solution.reshape((OPT_GLOBALS.PRED_LEN, sample_time_decision_var_num)), x_solution2, t_end_lin - t_start_lin


def mpc_set(linear_model: Linear_State_Space_Model, mpc_obj_fcn_params: MPC_Objective_Function_Params, bounds: Bounds):

    yss = 0
    PIC_ss = 0

    nx = 12
    nu = 4
    ny = 1

    eq_constraints_num = nu+ny+ny+4+nx+nu

    # 4 decision variables are added for actual inputs
    sample_time_decision_var_num = nu + ny + ny + 6 + nx + nu + nu

    weight_matrix = np.zeros((sample_time_decision_var_num, sample_time_decision_var_num))
    #weight_matrix[0:nu, 0:nu] = mpc_obj_fcn_params.Q_du  # du
    weight_matrix[nu+ny:nu+ny+ny, nu+ny:nu+ny+ny] = mpc_obj_fcn_params.Q_e  # e
    #weight_matrix[nu+ny+ny+4:nu+ny+ny+6, nu+ny+ny+4:nu+ny+ny+6] = mpc_obj_fcn_params.Q_PIC  # PIC deviation
    #weight_matrix[nu+ny+ny+6:nu+ny+ny+nx+6, nu+ny+ny+6:nu+ny+ny+nx+6] = mpc_obj_fcn_params.Q_x  # x (states)
    weight_matrix[nu+ny+ny+nx+nu+6:nu+ny+ny+nx+nu+nu+6, nu+ny+ny+nx+nu+6:nu+ny+ny+nx+nu+nu+6] = mpc_obj_fcn_params.Q_u  # u_actual

    H = np.kron(np.eye(OPT_GLOBALS.PRED_LEN), weight_matrix)
    f = np.zeros((sample_time_decision_var_num * OPT_GLOBALS.PRED_LEN, 1))

    A_inequality = None
    b_inequality = None

    # equality constraints
    A0_equality = np.concatenate([
        np.concatenate([np.zeros(nu), [1], np.zeros(ny+4+2+nx+nu+nu)]).reshape(1, -1),  # y
        np.concatenate([np.zeros(nu), [1], [-1], np.zeros(4+2+nx+nu+nu)]).reshape(1, -1),  # e
        np.concatenate([np.zeros(nu), [0], [0], [1, 0, 0, 0], np.zeros(2+nx+nu+nu)]).reshape(1, -1),  # PIC upper hard constraint
        np.concatenate([np.zeros(nu), [0], [0], [0, 1, 0, 0], np.zeros(2+nx+nu+nu)]).reshape(1, -1),  # PIC lower hard constraint
        np.concatenate([np.zeros(nu), [0], [0], [0, 0, 1, 0], np.zeros(2+nx+nu+nu)]).reshape(1, -1),  # PIC soft constraint
        np.concatenate([np.zeros(nu), [0], [0], [0, 0, 0, 1], np.zeros(2+nx+nu+nu)]).reshape(1, -1),  # PIC soft constraint
        np.concatenate([np.zeros((nx, nu+2*ny+4+2)), np.eye(nx), np.zeros((nx, nu+nu))], axis=1),  # x
        np.concatenate([-np.eye(nu), np.zeros((nu, 2*ny+4+2+nx)), np.eye(nu), np.zeros((nu, nu))], axis=1),  # du
        np.concatenate([np.zeros((nu, nu+ny+ny+6+nx)), -np.eye(nu), np.eye(nu)], axis=1)  # u_actual
    ], axis=0)

    A0_equality = np.concatenate([A0_equality, np.zeros((A0_equality.shape[0], sample_time_decision_var_num * (OPT_GLOBALS.PRED_LEN - 1)))], axis=1)

    A11_equality_block = np.concatenate([
        np.zeros((ny+ny+4, sample_time_decision_var_num)),  # y, e, and 4 PIC upper and lower constraints
        np.concatenate([np.zeros((nx, nu+2*ny+4+2)), linear_model.get_A(), linear_model.get_B_artificial(), np.zeros((nx, nu))], axis=1),  # x
        np.concatenate([np.zeros((nu, sample_time_decision_var_num - nu - nu)), -np.eye(nu), np.zeros((nu, nu))], axis=1),  # du
        np.zeros((nu, sample_time_decision_var_num))  # u_actual
        ], axis=0)

    A11_equality = np.kron(np.eye(OPT_GLOBALS.PRED_LEN - 1), A11_equality_block)
    A11_equality = np.concatenate([A11_equality, np.zeros(((OPT_GLOBALS.PRED_LEN - 1) * eq_constraints_num, sample_time_decision_var_num))], axis=1)

    A12_equality_block = np.concatenate([
            np.concatenate([np.zeros(nu), [1], np.zeros(ny+4+2), -linear_model.get_C().reshape(-1), np.zeros(nu+nu)]).reshape(1, -1),  # y
            np.concatenate([np.zeros(nu), [1], [-1], np.zeros(4+2+nx+nu+nu)]).reshape(1, -1),  # e
            np.concatenate([np.zeros(nu), [0], [0], [1, 0, 0, 0], np.zeros(2+nx+nu+nu)]).reshape(1, -1),  # PIC upper hard constraint
            np.concatenate([np.zeros(nu), [0], [0], [0, 1, 0, 0], np.zeros(2+nx+nu+nu)]).reshape(1, -1),  # PIC lower hard constraint
            np.concatenate([np.zeros(nu), [0], [0], [0, 0, 1, 0], np.zeros(2+nx+nu+nu)]).reshape(1, -1),  # PIC soft constraint
            np.concatenate([np.zeros(nu), [0], [0], [0, 0, 0, 1], np.zeros(2+nx+nu+nu)]).reshape(1, -1),  # PIC soft constraint
            np.concatenate([np.zeros((nx, nu+2*ny+4+2)), -np.eye(nx), np.zeros((nx, nu+nu))], axis=1),  # x
            np.concatenate([-np.eye(nu), np.zeros((nu, 2*ny+4+2+nx)), np.eye(nu), np.zeros((nu, nu))], axis=1),  # du
            np.concatenate([np.zeros((nu, nu+ny+ny+6+nx)), -np.eye(nu), np.eye(nu)], axis=1)   # u_actual
    ], axis=0)

    A12_equality = np.kron(np.eye(OPT_GLOBALS.PRED_LEN - 1), A12_equality_block)
    A12_equality = np.concatenate([np.zeros(((OPT_GLOBALS.PRED_LEN - 1) * eq_constraints_num, sample_time_decision_var_num)), A12_equality], axis=1)
    A1_equality = A11_equality + A12_equality
    A_equality = np.concatenate([A0_equality, A1_equality], axis=0)

    # the value of input u around which the system is linearized
    u0 = np.concatenate([linear_model.u0, [0]])

    b0_equality = np.concatenate([
                            [0,  # y
                            mpc_obj_fcn_params.ysp,  # e
                            0 * yss + 0 - PIC_ss,  # PIC upper hard constraint
                            0 * yss + 0 - PIC_ss,  # PIC lower hard constraint
                            0 * yss + 0 - PIC_ss,  # PIC soft constraint
                            0 * yss + 0 - PIC_ss],  # PIC soft constraint
                            np.zeros(nx),  # x
                            np.array([0, 0, 0, 1]),  # du (last input value is 1 in deviation format)
                            u0
                            ]).reshape(-1, 1)

    b1_equality_block = np.concatenate([
                            [0,  # y
                            mpc_obj_fcn_params.ysp,  # e
                            0 * yss + 0 - PIC_ss,  # PIC upper hard constraint
                            0 * yss + 0 - PIC_ss,  # PIC lower hard constraint
                            0 * yss + 0 - PIC_ss,  # PIC soft constraint
                            0 * yss + 0 - PIC_ss],  # PIC soft constraint
                            np.zeros(nx),  # x
                            np.zeros(nu),  # du
                            u0
                            ]).reshape(-1, 1)

    b1_equality = np.tile(b1_equality_block, (OPT_GLOBALS.PRED_LEN - 1, 1))
    b_equality = np.concatenate([b0_equality, b1_equality], axis=0)

    first_input_indices = np.arange(sample_time_decision_var_num - nu - nu, sample_time_decision_var_num * OPT_GLOBALS.PRED_LEN, sample_time_decision_var_num)
    u_index = np.hstack([first_input_indices, first_input_indices + 1, first_input_indices + 2, first_input_indices + 3])
    u_index = np.sort(u_index)

    lb_ = np.concatenate([bounds.du_min, bounds.y_min, np.full(ny, -np.inf), np.full(6, -np.inf), bounds.x_min, np.zeros(nu), np.full(nu, -np.inf)]).reshape(-1, 1)
    lb = np.tile(lb_, (OPT_GLOBALS.PRED_LEN, 1))
    lb[u_index] = bounds.u_min.T.reshape(-1, 1)

    ub_ = np.concatenate([bounds.du_max, bounds.y_max, np.full(ny, np.inf), np.full(6, np.inf), bounds.x_max, np.zeros(nu), np.full(nu, np.inf)]).reshape(-1, 1)
    ub = np.tile(ub_, (OPT_GLOBALS.PRED_LEN, 1))
    ub[u_index] = bounds.u_max.T.reshape(-1, 1)

    return H, f, A_inequality, b_inequality, A_equality, b_equality, lb, ub


def linear_model_update(states, inputs, linear_model):
    return np.matmul(linear_model.get_A(),states[:,None])[:,0] + np.matmul(linear_model.get_B_artificial(),inputs[:,None])[:,0]



def run_lmpc(insulin, linear_model, inputs, init_state, VG):
    inputs[:,0] = insulin
    states = np.zeros((inputs.shape[0],init_state.shape[0]))
    for i in range(inputs.shape[0]-1,):
        states[i+1] = linear_model_update(states[i],inputs[i],linear_model)
    
    states = states + init_state

    return np.sum(np.power(states[:,3]/VG-OPT_GLOBALS.SETPOINT/OPT_GLOBALS.MMOL_TO_MGDL,2))+OPT_GLOBALS.RU*np.sum(np.power(insulin,2))