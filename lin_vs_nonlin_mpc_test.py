import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy import optimize
import time
import pickle
from matplotlib import pyplot as plt
from mpc import *
from nmpc import *
from nmpc_jax import *
import OPT_GLOBALS
from tqdm import tqdm
from jaxopt import ScipyBoundedMinimize
import datetime
jax.config.update('jax_enable_x64', True)
import warnings
warnings.filterwarnings("ignore", message="Converted P to scipy.sparse.csc.csc_matrix")

def concentration_mgdl_to_mmolL(mgdL):
    return mgdL/OPT_GLOBALS.MMOL_TO_MGDL

# This line loads a simulation result which we will use as the testing scenarios and states.
with open("./SimulationResults/gri_fit/model.pkl","rb") as f:
    model_data = pickle.load(f)

# Extract parameters and states from the loaded model data
parameters = model_data.parameters.as_array
basal_infusion = get_hovorka_basal_equilibrium(parameters, concentration_mgdl_to_mmolL(100.0))
states = model_data.states.as_array[:,0:13,:]
print(states.shape)

# Simulation settings
PATIENTS_TO_TEST = parameters.shape[0]  # Full length of patients: parameters.shape[0], Quick test: (1,) [#]
TEST_END_TIME = states.shape[2]         # Full length of simulation: states.shape[2], Quick test: 600  [minutes]

# Prediction horizon length, we will tests different lengths
pred_len = OPT_GLOBALS.PRED_LEN
pred_steps = int(OPT_GLOBALS.PRED_LEN/OPT_GLOBALS.TS)

meal_arr = np.zeros((1,pred_len))
inputs = np.concatenate((np.zeros((1,pred_len)),meal_arr,np.zeros((1,pred_len)),np.zeros((1,pred_len)),np.zeros((1,pred_len))),axis=0).T

bounds = np.zeros((pred_steps,2))
bounds[:,1] = OPT_GLOBALS.MAX_RATE

init_ins = basal_infusion[:,None]*np.ones((basal_infusion.shape[0],pred_len))
init_state = states[0,:,0]

# This calls once the optimization to get the functions compiled so compilation does not interve with benchmarking
res = optimize.minimize(run_mpc,np.zeros((pred_steps,)),(parameters[0],np.copy(inputs),init_state),method='L-BFGS-B',bounds=bounds,tol=OPT_GLOBALS.TOL,options={"maxiter":OPT_GLOBALS.MAXITER})
final_ins = res.x

# JAX setup for the Auto diff. NMPC
bounds_jax = (0.0, 1000.0)
inputs_jax = jnp.concatenate((jnp.zeros((1,pred_len)),meal_arr,jnp.zeros((1,pred_len)),jnp.zeros((1,pred_len)),jnp.zeros((1,pred_len))),axis=0)
parameters_jax = jnp.array(model_data.parameters.as_array)
basal_infusion_jax = jnp.array(get_hovorka_basal_equilibrium(parameters_jax, concentration_mgdl_to_mmolL(100.0)))
states_jax = jnp.array(model_data.states.as_array[:,:13,:])
init_state_jax = states_jax[0,:,0]
init_ins_jax = basal_infusion_jax[:,None]*jnp.ones((basal_infusion_jax.shape[0],pred_len))
opt = optax.lbfgs()
lbfgsb = ScipyBoundedMinimize(fun=run_mpc_jax, method="l-bfgs-b",maxiter=OPT_GLOBALS.MAXITER, tol=OPT_GLOBALS.TOL,jit=True)
final_params,info = lbfgsb.run(jnp.zeros((pred_steps,)), bounds=bounds_jax, parameters=parameters_jax[0], inputs=inputs_jax, init_state=jnp.copy(init_state_jax))

# Initialize controller errors array
ctrl_error_step = OPT_GLOBALS.PRED_LEN // 5 # 5 glucose errors per simulation
# ctrl_errors: 1st dim: controller type (0: SCIPY NMPC, 1: Jax NMPC, 2: LMPC), 2nd dim: error type (0: error, 1: cost), 3rd dim: time points (0-4), 4th dim: simulation index
ctrl_errors = np.zeros((3, 2, 5, TEST_END_TIME // 60))  # glucose errors for all controllers from the nominal setpoint

# Initialize lists to store simulation states, optimized insulins, errors, optimization times, and iterations
saved_states_nl         = []            # simulation states from Finite diff nmpc with QP mpc insulins
saved_states_l          = []            # simulation states from QP mpc
saved_states_nlnl       = []            # simulation states from Finite diff nmpc
saved_states_jax        = []            # simulation states from Auto diff nmpc
failed_sim_states       = []            # states from failed simulations
failed_sim_insulins     = []            # insulins from failed simulations
results_lin             = []            # optimized insulins by QP mpc
results_nl              = []            # optimized insulins by Finite diff nmpc
results_jax             = []            # optimized insulins by Auto diff nmpc
costs                   = []            # optimization costs for all controllers
lin_mod_errors          = []            # linear model errors compared to the nonlinear model
time_of_optimizations   = [[],[],[]]    # time taken for optimizations for all controllers
nit_of_optimizations    = [[],[]]       # number of iterations for optimizations for all controllers

# We will test the MPCs on all 20 virtual patients on a lot of different initial states.
for vpi in range(PATIENTS_TO_TEST):# Full length of patients: parameters.shape[0], Quick test: 1,
    VG = parameters[vpi][17]
    cgm_array = states[vpi, 6, :] / VG * OPT_GLOBALS.MMOL_TO_MGDL  # basis one minute
    for si in tqdm(range(0, TEST_END_TIME, 60), desc=f"VP {vpi+1}/{PATIENTS_TO_TEST}"):

        # init_ins[vpi]: is the basal insulin array for the given patient and it is used as initial condition for the optimization algorithm
        # parameters[vpi]: is the parameters of the given patient, care has to taken to the order of the parameters. See model function: (kb1, kb2, kb3, EGP0, ke, F01, AG, tmaxI, tmaxG, p3, p4, p5, beta, a, BW, HRrest, HRmax, VG, VI, VT_HRR, k12, ka1, ka2, ka3, aSI, b, c, dSI, tsub, p1, p2, p6, p7, p8, tmaxGFast) = parameters.T
        # inputs: arrays which contains the cho values, insulin injections, heart rate values (not used)
        # states[vpi,:,si]: is the initial state for the model for the current test scenario.

        # Finite diff. LBFGS - Nonlinear
        t_start_scipy = time.time()
        res = optimize.minimize(run_mpc,np.zeros((pred_steps)),(parameters[vpi],np.copy(inputs),states[vpi,:,si]),method='L-BFGS-B',bounds=bounds,tol=OPT_GLOBALS.TOL,options={"maxiter":OPT_GLOBALS.MAXITER})
        t_end_scipy = time.time()
        res.x = np.repeat(res.x,OPT_GLOBALS.TS)
        results_nl.append(res.x)
        scipy_nit = res.nit
        test_inputs = np.copy(inputs)
        test_inputs[:,3] = res.x
        sim_test_states = np.zeros((inputs.shape[0],init_state.shape[0]))
        sim_test_states[0] = states[vpi,:,si]
        for i in range(inputs.shape[0]-1,):
            sim_test_states[i+1] = model(states=sim_test_states[i],inputs=test_inputs[i],parameters=parameters[vpi])

        # Autodiff LBFGS - Nonlinear
        t_start_jax = time.time()
        final_params,info = lbfgsb.run(jnp.zeros((pred_steps,)), bounds=bounds_jax, parameters=parameters_jax[vpi], inputs=inputs_jax, init_state=jnp.copy(states_jax[vpi,:,si]))
        t_end_jax = time.time()
        final_params = jnp.repeat(final_params,OPT_GLOBALS.TS)
        jax_nit = info.iter_num
        test_inputs_jax = jnp.array(np.copy(inputs_jax.T))
        test_inputs_jax = test_inputs_jax.at[:,3].set(final_params)
        results_jax.append(np.array(final_params))

        _, sim_test_states_jax = jax.lax.scan(jax.tree_util.Partial(model_jax2, parameters=parameters_jax[vpi]),jnp.copy(states_jax[vpi,:,si]), test_inputs_jax)
        sim_test_states_jax = np.concatenate(([states_jax[vpi,:,si]],np.array(sim_test_states_jax)[:-1,:]))

        # QP - Linear
        basal_rate = init_ins[vpi][0]
        cgm = OPT_GLOBALS.MMOL_TO_MGDL*states[vpi, 6, si]/ VG
        S1, S2, I, x1, x2, x3, Q1, Q2, IG, D1Slow, D2Slow, D1Fast, D2Fast = states[vpi, :, si]
        x0_mpc = np.array([S1,S2,I,Q1,Q2,0.0,0.0,x1,x2,x3,D1Slow,D2Slow])
        linear_model = get_hovorka_linear_model(x0_mpc, [inputs[0][3], inputs[0][0] + inputs[0][1], 1], cgm, parameters[vpi], T=1)

        res_lin_extended, qp_solution, res_lin, t_qp = mpc_execute(linear_model, parameters[vpi], x0_mpc, cgm, basal_rate, np.copy(inputs))
        time_of_optimizations[0].append(t_end_scipy - t_start_scipy)
        time_of_optimizations[1].append(t_end_jax - t_start_jax)
        time_of_optimizations[2].append(t_qp)
        nit_of_optimizations[0].append(scipy_nit)
        nit_of_optimizations[1].append(jax_nit)

        # Check if the linear MPC simulation was successful
        if res_lin is None:
            print(f"Simulation failed at patient {vpi+1} simulation {si+1}")
            print("States:", states[vpi,:,si])
            print("Insulin:", res.x)
            # Save insulin and glucose sequences of scipy and jax results
            failed_sim_states.append(states[vpi,:,si])
            failed_sim_insulins.append(res.x)
            # Log the failed simulation to a file
            with open("failed_simulations.log", "a") as log_file:
                log_file.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Simulation failed at patient {vpi+1} simulation {si+1}\n")
                log_file.write(f"States: {states[vpi,:,si]}\n")
                log_file.write(f"Insulin: {res.x}\n")
                log_file.write("-" * 80 + "\n")
            continue # skip this cycle
        else:
            res_lin = np.repeat(res_lin,OPT_GLOBALS.TS)

        test_inputs = np.copy(inputs)
        test_inputs[:,3] = res_lin
        sim_test_states_lin = np.zeros((inputs.shape[0],init_state.shape[0]))
        sim_test_states_lin[0] = states[vpi,:,si]
        for i in range(inputs.shape[0]-1,):
            sim_test_states_lin[i+1] = model(states=sim_test_states_lin[i],inputs=test_inputs[i],parameters=parameters[vpi])

        test_inputs = np.copy(inputs)
        test_inputs[:,3] = res_lin
        sim_test_states_lin2 = np.zeros((inputs.shape[0],init_state.shape[0]))

        sim_test_states_lin2[0] = states[vpi,:,si]
        f0 = model_jax(jnp.array(sim_test_states_lin2[0]),jnp.zeros((5,)),parameters[vpi])
        Ax = np.array(jax.jacobian(model_jax,argnums=0)(jnp.array(sim_test_states_lin2[0]),test_inputs[0],parameters[vpi]))
        Bx = np.array(jax.jacobian(model_jax,argnums=1)(jnp.array(sim_test_states_lin2[0]),test_inputs[0],parameters[vpi]))
        for i in range(inputs.shape[0]-1,):
            approx = f0+np.matmul(Ax,(sim_test_states_lin2[i]-sim_test_states_lin2[0])[:,None])[:,0]
            sim_test_states_lin2[i+1] = sim_test_states_lin2[i]+approx+np.matmul(Bx,test_inputs[i][:,None])[:,0]

        results_lin.append(res_lin)

        # plt.figure()
        # plt.plot(sim_test_states[:,6]/VG,label="SCIPY NL")
        # plt.plot(sim_test_states_lin[:,6]/VG,label="QP NL")
        # plt.plot(sim_test_states_jax[:,6]/VG,label="JAX NL",linestyle="--")
        # plt.plot(sim_test_states_lin2[:,6]/VG,label="QP2 L")
        # plt.legend()
        # plt.show()
       
        lin_mod_error = np.abs(sim_test_states_lin2[:,6]/VG - sim_test_states_lin[:,6]/VG)
        lin_mod_errors.append(lin_mod_error)

        saved_states_nl.append(np.concatenate((sim_test_states_lin,sim_test_states_lin[:,[6]]/VG,res_lin[:,None]),axis=1))
        saved_states_l.append(np.concatenate((sim_test_states_lin2,sim_test_states_lin2[:,[6]]/VG,res_lin[:,None]),axis=1))
        saved_states_nlnl.append(np.concatenate((sim_test_states,sim_test_states[:,[6]]/VG,res.x[:,None]),axis=1))
        saved_states_jax.append(np.concatenate((np.array(sim_test_states_jax),np.array(sim_test_states_jax[:,[6]])/VG,np.array(final_params)[:,None]),axis=1))

        # Calculate control errors 5 times per simulation
        for e in range(5,):
            #print(ctrl_error_step)
            ctrl_errors[0, 0, e, si//60] += np.abs(sim_test_states[ctrl_error_step * (e + 1) - 1,6]/VG - OPT_GLOBALS.SETPOINT / OPT_GLOBALS.MMOL_TO_MGDL)
            ctrl_errors[1, 0, e, si//60] += np.abs(sim_test_states_jax[ctrl_error_step * (e + 1) - 1,6]/VG - OPT_GLOBALS.SETPOINT / OPT_GLOBALS.MMOL_TO_MGDL)
            ctrl_errors[2, 0, e, si//60] += np.abs(sim_test_states_lin[ctrl_error_step * (e + 1) -1,6]/VG - OPT_GLOBALS.SETPOINT / OPT_GLOBALS.MMOL_TO_MGDL)

            ctrl_errors[0, 1, e, si//60] += np.sum(np.power(sim_test_states[ctrl_error_step * (e + 1) -1,6]/VG-OPT_GLOBALS.SETPOINT/OPT_GLOBALS.MMOL_TO_MGDL,2))+OPT_GLOBALS.RU*np.sum(np.power(res.x,2))
            ctrl_errors[1, 1, e, si//60] += jnp.sum(jnp.power(sim_test_states_jax[ctrl_error_step * (e + 1) - 1,6]/VG-OPT_GLOBALS.SETPOINT/OPT_GLOBALS.MMOL_TO_MGDL,2))+OPT_GLOBALS.RU*jnp.sum(jnp.power(final_params,2))
            ctrl_errors[2, 1, e, si//60] += np.sum(np.power(sim_test_states_lin[ctrl_error_step * (e + 1) -1,6]/VG-OPT_GLOBALS.SETPOINT/OPT_GLOBALS.MMOL_TO_MGDL,2))+OPT_GLOBALS.RU*np.sum(np.power(res_lin,2))

    np.save("data/saved_states_nl",np.array(saved_states_nl))
    np.save("data/saved_states_l",np.array(saved_states_l))
    np.save("data/saved_states_nlnl",np.array(saved_states_nlnl))
    np.save("data/saved_states_jax",np.array(saved_states_jax))

ctrl_errors /= parameters.shape[0]

# Save errors and states to npy files for plotting and analysis
np.save('data/lin_mod_errors.npy', lin_mod_errors)
np.save('data/results_scipy.npy', results_nl)
np.save('data/results_jax.npy', results_jax)
np.save('data/results_lin.npy', results_lin)
np.save('data/ctrl_errors.npy', ctrl_errors)
np.save('data/nit_of_optimizations.npy', np.array(nit_of_optimizations))
np.save('data/time_of_optimizations.npy', np.array(time_of_optimizations))

print("States, insulin, errors saved to npy files.")
