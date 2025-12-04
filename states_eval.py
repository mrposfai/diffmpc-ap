import numpy as np
from matplotlib import pyplot as plt
import OPT_GLOBALS
import matplotlib
from matplotlib.patches import Patch
import statsmodels.api as sm
import pandas as pd

box_width = 0.6


def bar_plot_controller_errors(ctrl_errors, title, xlabel, ylabel):
    # Create box plot for controller errors
    fig, ax = plt.subplots(figsize=(7, 5))
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
    ax.set_xticklabels([f'{(i+1)*ctrl_error_step} min' for i in range(5)])

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

matplotlib.rcParams.update({'font.size': 12})


states_l = np.load('data/saved_states_l_5.npy')
states_nl = np.load('data/saved_states_nl_5.npy')
states_nlnl = np.load('data/saved_states_nlnl_5.npy')
states_jax = np.load('data/saved_states_jax_5.npy')


# states_l = np.load('data/saved_states_l_1.npy')
# states_nl = np.load('data/saved_states_nl_1.npy')
# states_nlnl = np.load('data/saved_states_nlnl_1.npy')
# states_jax = np.load('data/saved_states_jax_1.npy')
time_opt_1 = np.load("data/time_of_optimizations_ryzen54600h_1.npy")
time_opt_5 = np.load("data/time_of_optimizations_ryzen54600h_5.npy")
nit_5 = np.load("data/nit_of_optimizations_5.npy")
nit_1 = np.load("data/nit_of_optimizations_1.npy")
# 0:SCIPY, 1:JAY, 2:QP
print(np.mean(time_opt_1,axis=1))
print(np.std(time_opt_1,axis=1))
print(np.min(time_opt_1,axis=1))
print(np.max(time_opt_1,axis=1))

print("--")
print(np.mean(time_opt_5,axis=1))
print(np.std(time_opt_5,axis=1))
print(np.min(time_opt_5,axis=1))
print(np.max(time_opt_5,axis=1))

print("--")
print(np.mean(nit_1,axis=1))
print(np.mean(nit_5,axis=1))

ctrl_error_step = states_l.shape[1] // 5
lin_mod_errors_array = OPT_GLOBALS.MMOL_TO_MGDL*np.abs(states_l-states_nl)[:,:,-2]
print(lin_mod_errors_array.shape)
mean_errors = np.mean(lin_mod_errors_array, axis=0)
p10_errors = np.percentile(lin_mod_errors_array, 10, axis=0)
p90_errors = np.percentile(lin_mod_errors_array, 90, axis=0)

# Plot linear model errors with percentile bands
plt.figure(figsize=(8, 5))
time_steps = np.arange(len(mean_errors))
plt.plot(time_steps, mean_errors, label='Mean Error', linewidth=2)
plt.fill_between(time_steps, p10_errors, p90_errors, alpha=0.3, label='10-90% Percentile')
plt.xticks(time_steps[::60], (time_steps[::60]).astype(int))
plt.xlabel('Time [min]')
plt.ylabel('Absolute difference [mg/dL]')
#plt.title('Linear Model Error vs Nonlinear Model')
plt.xlim([0,180])
plt.legend()
plt.grid(True)
#plt.show()


avg_pred_err = lin_mod_errors_array.mean(1)
init_bg = states_nl[:,0,-2]
init_I = states_nl[:,0,2]
init_D2 = states_nl[:,0,10]


mod = sm.OLS(pd.DataFrame({"pred_diff":avg_pred_err}), pd.DataFrame({"intercept":np.ones_like(init_bg),"init_bg":OPT_GLOBALS.MMOL_TO_MGDL*init_bg,"init_I":init_I,"init_D2":init_D2}))

res = mod.fit()

print(res.summary())


print(states_l.shape)


rmse = np.sqrt(np.mean(np.square(states_l[:,:,-2]-states_nl[:,:,-2]),axis=-1))
medidx = np.abs(rmse - np.median(rmse)).argmin()

print(rmse)

plt.figure()
plt.scatter(states_nl[:,0,2],rmse)
#plt.ylim([0,50])


idx = medidx
fig, ax = plt.subplots(2,1,figsize=(6,6))
ax[0].plot(OPT_GLOBALS.MMOL_TO_MGDL*states_l[idx,:,-2],label="QP - Linear pred.",color="gray")
ax[0].plot(OPT_GLOBALS.MMOL_TO_MGDL*states_nl[idx,:,-2],label="QP - Nonlinear pred.",color="black")
ax[0].plot(OPT_GLOBALS.MMOL_TO_MGDL*states_nlnl[idx,:,-2],linestyle="-",label="Finite diff.",color="blue")
ax[0].plot(OPT_GLOBALS.MMOL_TO_MGDL*states_jax[idx,:,-2],linestyle="--",label="Auto diff.",color="orange")
ax[0].set_ylabel("Predicted glucose (mg/dL)")
ax[0].legend()
ax[1].plot(states_l[idx,:,-1],label="QP",color="gray")
ax[1].plot(states_nlnl[idx,:,-1],linestyle="-",label="Finite diff.",color="blue")
ax[1].plot(states_jax[idx,:,-1],linestyle="--",label="Auto diff.",color="orange")
ax[1].set_ylabel("Optimized insulin (mU/min)")
ax[1].set_xlabel("Time (minutes)")
plt.show()
#plt.ylim([0,10])


plt.figure()
plt.plot(states_nl[np.argmax(rmse),:,-1])
plt.plot(states_nlnl[np.argmax(rmse),:,-1])
plt.plot(states_jax[np.argmax(rmse),:,-1],linestyle="--")

costs = [[],[],[]]
#costs[0] = []
#costs[1] = []
#costs[2] = []

print(np.sum(np.power(states_nl[np.argmax(rmse),:,-2]-OPT_GLOBALS.SETPOINT/OPT_GLOBALS.MMOL_TO_MGDL,2))+OPT_GLOBALS.RU*np.sum(np.power(states_nl[np.argmax(rmse),:,-1],2)))
print(np.sum(np.power(states_nlnl[np.argmax(rmse),:,-2]-OPT_GLOBALS.SETPOINT/OPT_GLOBALS.MMOL_TO_MGDL,2))+OPT_GLOBALS.RU*np.sum(np.power(states_nlnl[np.argmax(rmse),:,-1],2)))
print(np.sum(np.power(states_jax[np.argmax(rmse),:,-2]-OPT_GLOBALS.SETPOINT/OPT_GLOBALS.MMOL_TO_MGDL,2))+OPT_GLOBALS.RU*np.sum(np.power(states_jax[np.argmax(rmse),:,-1],2)))

for i in range(states_nl.shape[0]):
    costs[2].append(np.power(states_nl[i,:,-2]-OPT_GLOBALS.SETPOINT/OPT_GLOBALS.MMOL_TO_MGDL,2)+OPT_GLOBALS.RU*np.power(states_nl[i,:,-1],2))
    costs[0].append(np.power(states_nlnl[i,:,-2]-OPT_GLOBALS.SETPOINT/OPT_GLOBALS.MMOL_TO_MGDL,2)+OPT_GLOBALS.RU*np.power(states_nlnl[i,:,-1],2))
    costs[1].append(np.power(states_jax[i,:,-2]-OPT_GLOBALS.SETPOINT/OPT_GLOBALS.MMOL_TO_MGDL,2)+OPT_GLOBALS.RU*np.power(states_jax[i,:,-1],2))


costs = np.array(costs)
print(costs.shape)
chunked_costs = costs.reshape(costs.shape[0],costs.shape[1],5, 36).mean(axis=3).transpose((0,2,1))
print(chunked_costs.shape)
bar_plot_controller_errors(chunked_costs, 
                           'Controller Error Comparison', 
                           'Prediction horizon bin (minutes)', 
                           'Controller Cost')