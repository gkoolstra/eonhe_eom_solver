from matplotlib import pyplot as plt
import platform, os, sys, time
import numpy as np
from scipy.optimize import minimize, approx_fprime
import h5py
from tqdm import tqdm
from termcolor import colored, cprint

if 'Windows' in platform.system():
    sys.path.append(r'C:\Users\slab\Documents\Code')
    sys.path.append(r'D:\BEMPP_shared\Modules')
    import interpolate_slow
else:
    sys.path.append("/Users/gkoolstra/Documents/Code")
    from BEMHelper import interpolate_slow

from Common import common, kfit
from TrapAnalysis import trap_analysis, import_data, artificial_anneal as anneal

# Parameters:
box_length = 40E-6
N_electrons = 150
N_rows = 3
row_spacing = 0.20E-6
N_cols = 50
col_spacing = 0.20E-6
resVs = np.arange(2.00, 0.04, -0.02)

helium_height = 1.00E-6
screening_length = 2 * 0.80E-6
fitdomain = (-0.75, 0.75)

epsilon = 1e-10
use_gradient = True
gradient_tolerance = 1E-1

annealing_steps = [1.0] * 5
simulation_name = "M018V6_resonator_Vsweep_%d_electrons" % N_electrons
save_path = r"/Users/gkoolstra/Desktop/res_electron_simulation"
sub_dir = time.strftime("%y%m%d_%H%M%S_{}".format(simulation_name))
save = True
create_movie = True

# Load the data from the dsp file:
path = r'/Users/gkoolstra/Documents/Code/M018_Yggdrasil/simulation/exported_maxwell_data/Resonator only/ResonatorBiasSymmetricPotential_1100nm.dsp'
elements, nodes, elem_solution, bounding_box = import_data.load_dsp(path)

xdata, ydata, Udata = interpolate_slow.prepare_for_interpolation(elements, nodes, elem_solution)

x0 = -2.0 # Starting point for y
k = 251 # This defines the sampling
xeval = anneal.construct_symmetric_y(x0, k)

fig0 = plt.figure(figsize=(5.,3.))
xinterp, yinterp, Uinterp = interpolate_slow.evaluate_on_grid(xdata, ydata, Udata, xeval=xeval,
                                                     yeval=np.linspace(-1,1,151), clim=(0.00, 1.00), plot_axes='xy', linestyle='None',
                                                     cmap=plt.cm.viridis, plot_data=True,
                                                     **common.plot_opt("darkorange", msize=6))

xinterp, yinterp, Uinterp = interpolate_slow.evaluate_on_grid(xdata, ydata, Udata, xeval=xeval,
                                                              yeval=helium_height*1E6, clim=(0.00, 1.00),
                                                              plot_axes='xy', linestyle='None',
                                                              cmap=plt.cm.viridis, plot_data=False,
                                                              **common.plot_opt("darkorange", msize=6))

# Mirror around the y-axis
xsize = len(Uinterp[0])
Uinterp_symmetric = np.zeros(2*xsize)
Uinterp_symmetric[:xsize] = Uinterp[0]
Uinterp_symmetric[xsize:] = Uinterp[0][::-1]

x_symmetric = np.zeros(2 * xsize)
x_symmetric[:xsize] = xinterp[0]
x_symmetric[xsize:] = -xinterp[0][::-1]

fig1 = plt.figure(figsize=(5.,3.))
common.configure_axes(12)
plt.plot(x_symmetric, -Uinterp_symmetric, '.k')
plt.ylim(-0.8, 0.1)
plt.xlim(-2, 2)
plt.ylabel("$U_{\mathrm{ext}}$ (eV)")
plt.xlabel("$x$ (mm)")

ax = plt.gca()
# ax.set_axis_bgcolor('none')
fr, ferr = kfit.fit_poly(x_symmetric, -Uinterp_symmetric, mode='even', fitparams=[0, 1E4, 1E4], domain=fitdomain)
plt.plot(x_symmetric, kfit.polyfunc_even(x_symmetric, *fr), color='r', lw=2.0)

t = trap_analysis.TrapSolver()
f, sigmaf = t.get_electron_frequency([fr[0], -fr[1]], [ferr[0], -ferr[1]])
plt.title(r"$\omega_e/2\pi$ = %.2f $\pm$ %.2f GHz"%(f/1E9, np.abs(sigmaf)/1E9))

x_box = np.linspace(-1.8E-6, 1.8E-6, 501)
y_box = np.linspace(-box_length/2, box_length/2, 11)
X_box, Y_box = np.meshgrid(x_box, y_box)

if N_cols * col_spacing > box_length:
    raise ValueError("Placing electrons outside of the box length as initial condition is not allowed! Increase N_rows!")

electron_initial_positions = anneal.get_rectangular_initial_condition(N_electrons, N_rows=N_rows, N_cols=N_cols,
                                                                      x0=0.0E-6, y0=0.0E-6, dx=col_spacing, dy=row_spacing)

xinit, yinit = anneal.r2xy(electron_initial_positions)

def coordinate_transformation(r):
    x, y = anneal.r2xy(r)
    y_new = EP.map_y_into_domain(y)
    r_new = anneal.xy2r(x, y_new)
    return r_new

# This is where the actual solving starts...
os.mkdir(os.path.join(save_path, sub_dir))
time.sleep(1)
os.mkdir(os.path.join(save_path, sub_dir, "Figures"))

if save:
    common.save_figure(fig0, save_path=os.path.join(save_path, sub_dir))
    common.save_figure(fig1, save_path=os.path.join(save_path, sub_dir))

# Save the data to a single file
f = h5py.File(os.path.join(os.path.join(save_path, sub_dir), "Results.h5"), "w")
f.create_dataset("electrostatic_potential", data=-Uinterp_symmetric)
f.create_dataset("xpoints", data=x_symmetric*1E-6)
f.create_dataset("ypoints", data=y_box)
f.create_dataset("Vres", data=resVs)
f.create_dataset("use_gradient", data=use_gradient)
f.create_dataset("gradient_tolerance", data=gradient_tolerance)

conv_mon_save_path = os.path.join(save_path, sub_dir, "Figures")

for k, Vres in tqdm(enumerate(resVs)):
    EP = anneal.ResonatorSolver(x_symmetric*1E-6, -Vres*Uinterp_symmetric, box_length=box_length, smoothing=0.0,
                                include_screening=True, screening_length=screening_length)

    if use_gradient:
        jac = EP.grad_total
        grad_Uopt = EP.grad_total
    else:
        jac = None
        grad_Uopt = lambda x: approx_fprime(x, EP.Vtotal, epsilon)

    ConvMon = anneal.ConvergenceMonitor(Uopt=EP.Vtotal, grad_Uopt=EP.grad_total, N=10,
                                        Uext=EP.V,
                                        xext=x_box, yext=y_box, verbose=False, eps=epsilon,
                                        save_path=conv_mon_save_path, figsize=(4, 6),
                                        coordinate_transformation=coordinate_transformation)

    minimizer_options = {'method' : 'L-BFGS-B',
                         'jac': EP.grad_total,
                         'options': {'disp': False, 'gtol': gradient_tolerance, 'eps': epsilon},
                         'callback': None}

    res = minimize(EP.Vtotal, electron_initial_positions, **minimizer_options)

    if res['status'] > 0:
        cprint("WARNING: Step %d (Vres = %.2f V) did not converge!"%(k,Vres), "red")

    # This maps the electron positions within the simulation domain
    res['x'] = coordinate_transformation(res['x'])

    if len(annealing_steps) > 0:
        #res = EP.parallel_perturb_and_solve(EP.Vtotal, len(annealing_steps), annealing_steps[0], res, minimizer_options)
        res = EP.sequential_perturb_and_solve(EP.Vtotal, len(annealing_steps), annealing_steps[0], res, minimizer_options)
        res['x'] = coordinate_transformation(res['x'])

    resonator_ns_area = anneal.get_electron_density_by_area(anneal.xy2r(res['x'][::2], res['x'][1::2]))
    resonator_ns_pos = anneal.get_electron_density_by_position(anneal.xy2r(res['x'][::2], res['x'][1::2]))
    #print(
    #    "Electron density on resonator = %.2e (by area) or %.2e (by position)" % (resonator_ns_area, resonator_ns_pos))

    t0 = time.time()
    PP = anneal.PostProcess(save_path=conv_mon_save_path)
    PP.save_snapshot(res['x'], xext=x_box, yext=y_box, Uext=EP.V,
                     figsize=(4,6), common=common, title="Vres = %.2f V"%Vres,
                     clim=(-0.75*max(resVs), 0))

    f.create_dataset("step_%04d/electron_final_coordinates" % k, data=res['x'])
    f.create_dataset("step_%04d/electron_initial_coordinates" % k, data=electron_initial_positions)
    f.create_dataset("step_%04d/solution_converged" % k, data=True if res['status'] == 0 else False)
    f.create_dataset("step_%04d/energy" % k, data=res['fun'])
    f.create_dataset("step_%04d/jacobian" % k, data=res['jac'])
    f.create_dataset("step_%04d/electrons_in_trap" % k, data=PP.get_trapped_electrons(res['x']))
    print("Saved data for V = %.2f V in %.2f seconds" % (Vres, time.time() - t0))

    # Use the solution from the current time step as the initial condition for the next timestep!
    electron_initial_positions = res['x']

f.close()

if create_movie:
    # Create a movie
    ConvMon.create_movie(fps=10,
                         filenames_in=time.strftime("%Y%m%d")+"_figure_%05d.png",
                         filename_out="resonator_%d_electrons.mp4" % (N_electrons))

    # Move the file from the Figures folder to the sub folder
    os.rename(os.path.join(save_path, sub_dir, "Figures/resonator_%d_electrons.mp4" % (N_electrons)),
              os.path.join(save_path, sub_dir, "resonator_%d_electrons.mp4" % (N_electrons)))

x, y = anneal.r2xy(res['x'])
final_func_val = res['fun']
n_iterations = res['nit']

if 0:
    if len(np.shape(ConvMon.jac)) > 1:
        figgy = plt.figure(figsize=(6, 4))
        common.configure_axes(12)
        # LInf-norm
        plt.plot(ConvMon.iter, np.amax(np.abs(ConvMon.jac[:, ::2]), axis=1),
                 '.-r', label=r'$L^\infty$-norm $\nabla_x U_\mathrm{opt}$')
        plt.plot(ConvMon.iter, np.amax(np.abs(ConvMon.jac[:, 1::2]), axis=1),
                 '.-b', label=r'$L^\infty$-norm $\nabla_y U_\mathrm{opt}$')
        # L2-norm
        plt.plot(ConvMon.iter, np.sum(np.abs(ConvMon.jac[:, ::2]) ** 2.0, axis=1) ** (1 / 2.),
                 '.-c', label=r'$L^2$-norm $\nabla_x U_\mathrm{opt}$')
        plt.plot(ConvMon.iter, np.sum(np.abs(ConvMon.jac[:, 1::2]) ** 2.0, axis=1) ** (1 / 2.),
                 '.-m', label=r'$L^2$-norm $\nabla_y U_\mathrm{opt}$')

        plt.title("%d electrons at Vres = %.2f V" % (N_electrons, Vres))

        plt.xlabel("Iterations")
        plt.ylabel(r"$\nabla_{x,y} U_\mathrm{opt}$")
        plt.yscale('log')
        plt.xlim(0, res['nit'])
        plt.grid()
        plt.legend(loc=0, ncol=2, prop={'size': 9})

        common.save_figure(figgy, save_path=os.path.join(save_path, sub_dir))

if 1:
    y_in_domain = EP.map_y_into_domain(y)

    figgy = plt.figure(figsize=(4, 6))
    common.configure_axes(12)
    plt.pcolormesh(x_box * 1E6, y_box * 1E6, EP.V(X_box, Y_box), cmap=plt.cm.Spectral_r, vmax=0.0)
    EP.draw_resonator_pins(0.72, 0.5, 0.5, **{'color': 'black', 'alpha': 0.25})
    plt.plot(xinit * 1E6, yinit * 1E6, 'o', color='palegreen', alpha=0.5)
    plt.plot(x * 1E6, y_in_domain * 1E6, 'o', color='deepskyblue', alpha=1.0)
    plt.xlim(np.min(x_box)*1E6, np.max(x_box)*1E6)
    plt.xlabel("$x$ ($\mu$m)")
    plt.ylabel("$y$ ($\mu$m)")
    plt.colorbar()

    plt.show()

    num_unbounded_electrons = anneal.check_unbounded_electrons(res['x'],
                                                               xdomain=(np.min(x_box), np.max(x_box)),
                                                               ydomain=(np.min(y_box), np.max(y_box)))

    print("Number of unbounded electrons = %d" % num_unbounded_electrons)