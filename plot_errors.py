import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
import OPT_GLOBALS
import numpy as np
import scipy.stats as stats
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

from mpc import *
from nmpc import *
from nmpc_jax import *

box_width = 0.6
matplotlib.rcParams.update({'font.size': 12})

# Create box plot for controller errors
def box_plot_controller_errors(ctrl_errors, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6, 5))
    # Prepare data for box plots - reshape ctrl_errors to get pairs for each time point
    # ctrl_errors shape: (2, 5, num_simulations)
    # We want to create box plots for each of the 5 time points, with pairs of NMPC and LMPC

    positions = []
    data_to_plot = []
    labels = []

    for i in range(5):
        # NMPC errors at time point i
        positions.append(i * 3 + box_width+0.05)
        data_to_plot.append(ctrl_errors[0, i, :])
        labels.append(f'Finite diff. NMPC-T{i+1}')
        
        # Jax NMPC errors at time point i
        positions.append(i * 3 + 2*(box_width+0.05))
        data_to_plot.append(ctrl_errors[1, i, :])
        labels.append(f'Auto diff. NMPC-T{i+1}')
        
        # LMPC errors at time point i
        positions.append(i * 3 + 3*(box_width+0.05))
        data_to_plot.append(ctrl_errors[2, i, :])
        labels.append(f'QP MPC-T{i+1}')

    # Create box plots
    bp = ax.boxplot(data_to_plot, positions=positions, widths=box_width, patch_artist=True,
                    showmeans=True, meanline=True,showfliers=False)

    # Color the boxes differently for NMPC and LMPC
    colors = ['blue', 'yellow', 'grey']
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % 3])

    # Set x-axis labels
    ax.set_xticks([i * 3 + 1.5 for i in range(5)])
    ax.set_xticklabels([f'{(i+1)*ctrl_error_step * 5} min' for i in range(5)])

    # Add legend
    legend_elements = [Patch(facecolor='blue', label='Finite diff. NMPC'),
                    Patch(facecolor='yellow', label='Auto diff. NMPC'),
                    Patch(facecolor='grey', label='QP MPC')]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim([0,None])
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

# Plot linear model errors with percentile bands
def plot_linear_model_errors(lin_mod_errors_array):
    mean_errors = np.mean(lin_mod_errors_array, axis=0)
    p10_errors = np.percentile(lin_mod_errors_array, 10, axis=0)
    p90_errors = np.percentile(lin_mod_errors_array, 90, axis=0)

    # Plot linear model errors with percentile bands
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(len(mean_errors))
    plt.plot(time_steps, mean_errors, label='Mean Error', linewidth=2)
    plt.fill_between(time_steps, p10_errors, p90_errors, alpha=0.3, label='10-90% Percentile')
    plt.xticks(time_steps[::12], (time_steps[::12] * 5).astype(int))
    plt.xlabel('Time [min]')
    plt.ylabel('Linear Model Error [mmol/L]')
    plt.title('Linear Model Error vs Nonlinear Model')
    plt.legend()
    plt.grid(True)
    plt.show()

# Create box plot for optimization times
def box_plot_optimization_times(time_of_optimizations):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for box plots
    # time_of_optimizations shape: (3, test_end_time // 60)
    # 0: SCIPY NMPC, 1: Jax NMPC, 2: LMPC

    data_to_plot = [
        time_of_optimizations[0, :],  # SCIPY NMPC
        time_of_optimizations[1, :],  # Jax NMPC
        time_of_optimizations[2, :]   # LMPC
    ]

    labels = ['SCIPY NMPC', 'Jax NMPC', 'LMPC']
    colors = ['blue', 'yellow', 'grey']

    # Create box plots
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color the boxes
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)

    ax.set_ylabel('Optimization Time (s)')
    ax.set_title('Optimization Time Comparison: NMPC vs Jax NMPC vs LMPC')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

# Create box plot for number of iterations
def box_plot_optimization_iterations(nit_of_optimizations):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for box plots
    # nit_of_optimizations shape: (2, test_end_time // 60)
    # 0: SCIPY NMPC, 1: Jax NMPC
    # Note: LMPC doesn't have iteration counts

    data_to_plot = [
        nit_of_optimizations[0, :],  # SCIPY NMPC
        nit_of_optimizations[1, :]   # Jax NMPC
    ]

    labels = ['SCIPY NMPC', 'Jax NMPC']
    colors = ['blue', 'yellow']

    # Create box plots
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color the boxes
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)

    ax.set_ylabel('Number of Iterations')
    ax.set_title('Optimization Iterations Comparison: NMPC vs Jax NMPC')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

pred_len = OPT_GLOBALS.PRED_LEN
meal_arr = np.zeros((1,pred_len))
inputs = np.concatenate((np.zeros((1,pred_len)),meal_arr,np.zeros((1,pred_len)),np.zeros((1,pred_len)),np.zeros((1,pred_len))),axis=0).T
ctrl_error_step = inputs.shape[0] // 5
ctrl_errors = np.load('data/ctrl_errors.npy')
# 0.dim Controllers
# 1.dim (Delta BG, Cost)
# 2.dim Steps on the pred horizon
# 3.dim Samples

agg_costs = ctrl_errors[:,1,:3,:]
agg_costs = agg_costs.reshape((*agg_costs.shape[:-2], -1))
agg_costs = agg_costs[:,agg_costs[2,:]<100.0]

plt.figure()
plt.boxplot(agg_costs.T)
plt.show()
print(agg_costs.shape)

#agg_costs = agg_costs[:,agg_costs[2]<100]
#agg_costs = agg_costs[:,:3]

m, s, n = np.mean(agg_costs,(1)), np.std(agg_costs,(1), ddof=1), agg_costs.shape[1]
t = stats.t.ppf(0.95, df=n-1)

e = t * (s / np.sqrt(n))
res_errs = np.stack((m,(m - e), (m + e))).T
print(res_errs)



#A = np.array(agg_costs[0])
#B = np.array(agg_costs[0])
#C = np.array(controller_C_errors)

# Optional transformation:
# A = np.log(A + 1e-8)
# B = np.log(B + 1e-8)
# C = np.log(C + 1e-8)
#agg_costs = np.log(agg_costs+1e-8)


stat, p = friedmanchisquare(*agg_costs)
print("Friedman test:", stat, p)


posthoc = sp.posthoc_wilcoxon(agg_costs, p_adjust='holm')
print(posthoc)

sim_test_states = np.load('data/sim_test_states_scipy.npy')
sim_test_states_jax = np.load('data/sim_test_states_jax.npy')
lin_mod_errors = np.load('data/lin_mod_errors.npy')
results_nl = np.load('data/results_scipy.npy')
results_jax = np.load('data/results_jax.npy')
results_lin = np.load('data/results_lin.npy')
time_of_optimizations = np.load('data/time_of_optimizations.npy')
nit_of_optimizations = np.load('data/nit_of_optimizations.npy')
lin_mod_errors_array = np.load('data/lin_mod_errors.npy')
plot_linear_model_errors(lin_mod_errors_array)

box_plot_optimization_times(time_of_optimizations)

box_plot_optimization_iterations(nit_of_optimizations)

box_plot_controller_errors(ctrl_errors[:,0,:,:], 
                           'Controller Error Comparison', 
                           'Prediction horizon (minutes)', 
                           'Control Error (mmol/L) \n setpoint = {:.1f} mmol/L'.format(OPT_GLOBALS.SETPOINT / OPT_GLOBALS.MMOL_TO_MGDL))

box_plot_controller_errors(np.square(ctrl_errors[:,0,:,:]), 
                           'Controller Error Comparison', 
                           'Prediction horizon (minutes)', 
                           'Squared Control Error')

box_plot_controller_errors(ctrl_errors[:,1,:,:], 
                           'Controller Error Comparison', 
                           'Prediction horizon (minutes)', 
                           'Controller Cost')