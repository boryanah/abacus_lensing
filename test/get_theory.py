# /global/homes/b/boryanah/anaconda3/envs/desc/lib/python3.7/site-packages/pyccl/tracers.py
# for the CMB tracer
# /global/homes/b/boryanah/anaconda3/envs/desc/lib/python3.7/site-packages/pyccl/boltzmann.py
# for cosmology of summit

import sys, os, glob

import asdf
import pyccl as ccl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

np.random.seed(300)

"""
default settings are: ph002, z_l =0.5, z_s = 1.025; with RSD though I think this only matters for theory; we vary the tracers
#python get_theory.py LRG; python get_theory.py ELG
python get_theory.py LRG 0.5 1.025 1 # galaxy-shear
python get_theory.py LRG 0.5 1.0 1 # relevant for C_ell kg
python get_theory.py LRG 1.025 1089.276682 1 # relevant for C_ell kg
#python get_theory.py ELG 0.5 1.025 1 # galaxy-shear
#python get_theory.py ELG 0.5 1.0 1 # relevant for C_ell kg
python get_theory.py ELG 0.8 1.4 1 # relevant for C_ell kg and galaxy-shear
python get_theory.py ELG 1.025 1089.276682 1 # relevant for C_ell kg
#python get_theory.py LRG 0.5 1089.276682
#python get_theory.py ELG 0.5 1089.276682
QUESTION: is source clustering interesting??
ask if I am doing the Pk_CB correctly
My understanding is I need to be multiplying by Omega_cb back in the kappa production stage; and then compare to theory kappa unmodified
perl gives slightly better results (pyccl version?);
pk_tmp = None does better for huge on large scales and worse on small scales (can't tell if double accounting); it pulls things up?
"""
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# define Cosmology object
want_rsd = True
rsd_str = "_rsd" if want_rsd else ""
h = 0.6736
Omega_nu = 0.0006442/h**2
N_nu_mass = 3 # nr species
Neff = (N_nu_mass * 0.71611**4 * (4./11.)**(-4./3.))
Neff = 3.04
cosmo_dic = {
    'h': h,
    'Omega_c': 0.12/h**2,
    'Omega_b': 0.02237/h**2,
    'A_s': 2.083e-9, # og
#    'sigma8': 0.811355, # emu
    'n_s': 0.9649,
    'T_CMB': 2.7255,
    'Neff': 2.0328, # og
    'm_nu': 0.06, # og
    'm_nu_type': 'single', # og
#    'Neff': Neff, # emu
#    'm_nu': 0.0, # emu
#    'm_nu_type': 'equal', # emu
    'w0': -1.,
    'wa': 0.,
    'transfer_function': 'boltzmann_class',
#    'matter_power_spectrum': 'emu' # requires sigma8 and not single
}
cosmo = ccl.Cosmology(**cosmo_dic)
load_neighbors = int(sys.argv[4]) #True
n_neighs = 6 # n_neighs-1 on each side # default 3, 11, 0.2
# file params
sim_name = "AbacusSummit_base_c000_ph002" #sys.argv[1] #"AbacusSummit_base_c000_ph006" #"AbacusSummit_base_c000_ph000"
#sim_name = "AbacusSummit_huge_c000_ph201"
tracer = sys.argv[1] #"ELG", "LRG"
z_l = float(sys.argv[2]) #0.5
#z_l = 0.8
#z_s = 0.5
z_s = float(sys.argv[3]) #1.025
#z_s = 1089.276682
if z_s < 10.:
    want_shear = True 
else:
    want_shear = False
if z_l == 0.5:
    Delta_z = 0.15
elif z_l == 0.8:
    Delta_z = 0.3
elif z_l == 1.025:
    Delta_z = 0.3
    
# directories
# gqc only available at z = 0.8
#cat_dir = "/global/cscratch1/sd/boryanah/AbacusHOD_scratch/mocks_lc_output/"; dir_name = "lensing"; sub_dir_s = sub_dir_l = ""
cat_dir = "/global/cfs/cdirs/desi/cosmosim/AbacusLensing/mocks/"; dir_name = ""; sub_dir_s = "halos/"; sub_dir_l = "DESI/" # z-evolved HODs
redshift_s = f"/z{z_s:.3f}/{dir_name}/"
redshift_l = f"/z{z_l:.3f}/{dir_name}/"
fn_s = "catalog_halos.asdf"
fn_l = f"catalog_DESI_{tracer}.asdf" # z-evolved HODs
if tracer == "LRG":
    # gqc only available at z = 0.8
    #fn_l = f"catalog_xi2d_{tracer.lower()}_main_z{z_l:.1f}_velbias_B_s_test_hod1.asdf"
    if z_l == 0.5:
        bias_ampl = 1.5  # 1.42
    else:
        bias_ampl = 1.39 #1.35 #1.5
    #0.95 # theoretical or matching mag surveys maybe lensing # bias
else:
    # gqc only available at z = 0.8
    #fn_l = f"catalog_xi2d_{tracer.lower()}_z{z_l:.1f}_velbias_confbeta_test_hod1.asdf"
    if z_l == 0.8:
        bias_ampl = 0.72 # TESTING og is 0.75 #0.7 #0.8
    else:
        bias_ampl = 0.75 #0.7 #0.8
file_name_s = cat_dir + sub_dir_s + sim_name + redshift_s + fn_s
file_name_l = cat_dir + sub_dir_l + sim_name + redshift_l + fn_l

if not os.path.exists(file_name_s):
    want_shear = False

# integration parameters
if 'base' in sim_name:
    z_min = 0.1
    z_max = 2.45104189964307 # 1800 sq deg
else:
    z_min = 0.1
    z_max = 2.176501504645313 # full sky

# correction neutrinos
Omega_m = Omega_nu + cosmo_dic['Omega_b'] + cosmo_dic['Omega_c']
Omega_cb = cosmo_dic['Omega_b'] + cosmo_dic['Omega_c']
factor = Omega_cb/Omega_m
print("Omega_m", Omega_m)
print("factor^2", factor**2)

# map specs
nside = 16384
lmax = nside*2
ell = np.arange(lmax)

if 'huge' in sim_name:
    # TESTING!!!!!
    if False:  #z_s < 1.1: # should be more exact?????
        # read in CLASS power spectra
        z_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.1]) # 99 is last
        a_arr = 1./(1+z_arr)
        class_dir = os.path.expanduser("~/repos/AbacusSummit/Cosmologies/abacus_cosm000/")
        ks, Pk = np.loadtxt(class_dir+'abacus_cosm000.z%d_pk_cb.dat'%(1),unpack=True)
        Pk_a_s = np.zeros((len(a_arr),len(ks)))
        for i in range(len(a_arr)):
            print(i)
            Pk_a_s[i,:] = np.loadtxt(class_dir+'abacus_cosm000.z%d_pk_cb.dat'%(i+1))[:,1]
        # change the order cause that's what CCL prefers
        i_sort = np.argsort(a_arr)
        a_arr = a_arr[i_sort]
        Pk_a_s = Pk_a_s[i_sort,:]
        lpk_arr = np.log(Pk_a_s/h**3)
        #a_arr = np.array([1./(1+z) for z in np.linspace(0, 1.1, 1000)[::-1]]) # doesn't make a difference
        #lpk_arr = np.log(np.array([ccl.nonlin_matter_power(cosmo, ks*h, a) for a in a_arr])) # same as default
        pk_tmp = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(ks*h), pk_arr=lpk_arr, is_logp=True)
        pk_tmp = pk_tmp.apply_halofit(cosmo)#, pk_linear=pk_tmp) # very important (runs only on perlmutter pyccl version)
        #factor = 1. # TESTING!!!!!
    else:
        pk_tmp = None    
    
    cmbl_s = ccl.CMBLensingTracer(cosmo, z_source=z_s, z_min=z_min, z_max=z_max)
    cls_kappa_th = ccl.angular_cl(cosmo, cmbl_s, cmbl_s, ell, p_of_k_a=pk_tmp)/factor**2 # kappa_s kappa_s (usual)
    np.savez(f"data/kappa_{sim_name}_zs{z_s:.3f}_ccl.npz", ell=ell, cl_kappa=cls_kappa_th)
    quit()
    
# dndz params
if want_rsd:
    #Z_label = 'Z_RSD'
    Z_label = 'Z_COSMO' # TESTING!!!!!!
else:
    Z_label = 'Z_COSMO'


# gather intel on the wl maps
file_name_ls = sorted(glob.glob(cat_dir + sub_dir_l + sim_name + (f"/z*/{dir_name}/") + fn_l))
z_lenss = []
for i in range(len(file_name_ls)):
    z_lens = asdf.open(file_name_ls[i])['header']['CatalogRedshift']
    z_lenss.append(z_lens)
z_lenss = np.sort(np.array(z_lenss))
i_lens = np.argmin(np.abs(z_lenss-z_l))
z_neighs = []
for i in range(1, n_neighs):
    if i_lens - i >= 0:
        z_neighs.append(z_lenss[i_lens-i])
    z_neighs.append(z_lenss[i_lens+i])
z_neighs = np.sort(np.array(z_neighs))
print("z neighs = ", z_neighs)

def get_dNdz(file_name, z_edges, Delta_z=None, load_neighbors=False):
    """
    function for estimating the N(z) of that particular catalog
    """
    Z = asdf.open(file_name)['data'][Z_label]
    if load_neighbors:
        for i, z_neigh in enumerate(z_neighs):
            redshift_l = f"/z{z_neigh:.3f}/{dir_name}/"
            file_name_l = cat_dir + sub_dir_l + sim_name + redshift_l + fn_l
            Z_this = asdf.open(file_name_l)['data'][Z_label]
            Z = np.hstack((Z, Z_this)).T
    print("Z min, max, mean, length = ", Z.min(), Z.max(), np.mean(Z), len(Z))
    if (Delta_z is not None):

        # version 3
        z_edges = np.linspace(0.1, 2.5, 1001)
        z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
        n_tot = len(Z)
        n_per = n_tot/len(z_cent)
        dNdz, _ = np.histogram(Z, bins=z_edges)
        inds = np.digitize(Z, bins=z_edges) - 1
        w_bin = dNdz/n_per
        #w_bin /= np.max(w_bin)
        fac = gaussian(Z, z_l, Delta_z/2.)
        #fac = 1.
        fac /= w_bin[inds]
        #fac /= np.max(fac) # not ideal because edges can throw things off
        fac /= np.max(fac[(Z > z_l-Delta_z/2.) & (Z < z_l+Delta_z/2.)])
        down = np.random.rand(n_tot) < fac
        Z = Z[down]
        print("kept fraction = ", np.sum(down)/len(down), np.sum(down))
    dNdz, _ = np.histogram(Z, bins=z_edges)
    dNdz = dNdz.astype(float)
    return dNdz
z_edges = np.linspace(0.1, 2.5, 1001)
z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
if want_shear: # there are galaxies (otherwise probs CMB)
    dNdz_s = get_dNdz(file_name_s, z_edges)
dNdz_l = get_dNdz(file_name_l, z_edges, Delta_z=Delta_z, load_neighbors=load_neighbors)
plot_dndz = False
if plot_dndz:
    plt.plot(z_cent, dNdz_l)
    plt.xlim([z_l-0.5, z_l+0.5])
    plt.savefig("dndz.png")
    quit()
# decide on bias form
bz_l = bias_ampl/ccl.background.growth_factor(cosmo, 1./(1+z_cent))
print("bias lens = ", bias_ampl/ccl.background.growth_factor(cosmo, 1./(1+z_l)))
print("comoving distance lens", ccl.background.comoving_radial_distance(cosmo, 1./(1+z_l)))
print("comoving distance source", ccl.background.comoving_radial_distance(cosmo, 1./(1+z_s)))
print("k at which 0.2 Mpc^-1", 0.2*ccl.background.comoving_radial_distance(cosmo, 1./(1+z_l))) # k = ell/chi
      
"""
for z in [0.1, 0.3, 0.5, 0.8, 1.1, 1.7]:
    print("z = ", z)
    print("bias (0.95) = ", 0.95/ccl.background.growth_factor(cosmo, 1./(1+z)))
    print("growth = ", ccl.background.growth_factor(cosmo, 1./(1+z)))
    print("--------------")
"""
# create another weak lensing tracer with the redshift distribution of your lenses
# pass that to ccl angular with number counts tracer with same dndz (what about bias)
# get C_ell magnification term kappa,g, which I can then multiply by mag coeff 

# alpha = 5. * s / 2.; no magnification means s = 0.4
# delta_m = alpha delta_mu (magnification) = 5s delta_k; delta_p = - delta_mu (deflection) = -2 delta_k; delta_all = delta_m + delta_p; delta_mu = 2 delta_kappa
s = 0.2 # = 2 alpha / 5 = dlog10N(<m, z)/dm slope of background (source) number counts

# corr params
theta = np.geomspace(0.1, 400, 100) # in arcmin, but ccl uses degrees
theta /= 60. # degrees

# tracers
if want_shear:
    weak = ccl.tracers.WeakLensingTracer(cosmo, dndz=(z_cent, dNdz_s), has_shear=True, ia_bias=None, z_min=z_min, z_max=z_max)#, use_A_ia=True)
    cmbl_l = ccl.CMBLensingTracer(cosmo, z_source=z_l, z_min=z_min, z_max=z_max)

# for C_ell kappa g and gg
cmbl_s = ccl.CMBLensingTracer(cosmo, z_source=z_s, z_min=z_min, z_max=z_max)
number = ccl.tracers.NumberCountsTracer(cosmo, dndz=(z_cent, dNdz_l), has_rsd=False, bias=(z_cent, bz_l), mag_bias=None)#, use_A_ia=True) # rsd #TESTING!!!!!!

# compute cross power spectra
if want_shear: # only makes sense to do shear_s if not CMB
    cls_shear_th = ccl.angular_cl(cosmo, weak, weak, ell)/factor**2 # shear_s shear_s (usual xipm)
    cls_gal_shear_th = ccl.angular_cl(cosmo, number, weak, ell)/factor # shear_s gal (usual gammat)
    cls_kappa_shear_th = ccl.angular_cl(cosmo, cmbl_l, weak, ell)/factor**2 # shear_s kappa_l (mag bias)
    cls_kappa_l_th = ccl.angular_cl(cosmo, cmbl_l, cmbl_l, ell)/factor**2 # kappa_l kappa_l (mag bias)
    cls_kappa_l_gal_th = ccl.angular_cl(cosmo, cmbl_l, number, ell)/factor  # kappa_l g (mag bias)

# pk cb
if False: #z_s < 1.1: # should be more exact?????
    # read in CLASS power spectra
    z_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.1]) # 99 is last
    a_arr = 1./(1+z_arr)
    class_dir = os.path.expanduser("~/repos/AbacusSummit/Cosmologies/abacus_cosm000/")
    ks, Pk = np.loadtxt(class_dir+'abacus_cosm000.z%d_pk_cb.dat'%(1),unpack=True)
    Pk_a_s = np.zeros((len(a_arr),len(ks)))
    for i in range(len(a_arr)):
        print(i)
        Pk_a_s[i,:] = np.loadtxt(class_dir+'abacus_cosm000.z%d_pk_cb.dat'%(i+1))[:,1]
    # change the order cause that's what CCL prefers
    i_sort = np.argsort(a_arr)
    a_arr = a_arr[i_sort]
    Pk_a_s = Pk_a_s[i_sort,:]
    lpk_arr = np.log(Pk_a_s/h**3)
    #a_arr = np.array([1./(1+z) for z in np.linspace(0, 1.1, 1000)[::-1]]) # doesn't make a difference
    #lpk_arr = np.log(np.array([ccl.nonlin_matter_power(cosmo, ks*h, a) for a in a_arr])) # same as default
    pk_tmp = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(ks*h), pk_arr=lpk_arr, is_logp=True)
    pk_tmp = pk_tmp.apply_halofit(cosmo)#, pk_linear=pk_tmp) # very important (runs only on perlmutter pyccl version)
    #factor = 1. # TESTING!!!!!
else:
    pk_tmp = None    

# save gg, kk, kg
cls_gal_th = ccl.angular_cl(cosmo, number, number, ell) # gg (usual)
cls_kappa_s_gal_th = ccl.angular_cl(cosmo, cmbl_s, number, ell, p_of_k_a=pk_tmp)/factor # kappa_s g (usual)
cls_kappa_th = ccl.angular_cl(cosmo, cmbl_s, cmbl_s, ell, p_of_k_a=pk_tmp)/factor**2 # kappa_s kappa_s (usual)
np.savez(f"data/kappa_{sim_name}_zs{z_s:.3f}_ccl.npz", ell=ell, cl_kappa=cls_kappa_th)
np.savez(f"data/kappa_gal_{sim_name}_{tracer}_zl{z_l:.3f}_zs{z_s:.3f}_ccl.npz", ell=ell, cl_kappa_gal=cls_kappa_s_gal_th)
np.savez(f"data/gal_{sim_name}_{tracer}_zl{z_l:.3f}_ccl.npz", ell=ell, cl_gal=cls_gal_th)


# assessing the mag bias contribution 2 (5s/2 -1) <kappa gamma>
if want_shear:
    # galaxy-shear
    cls_ng_mag_p = 2.*(2.5*s - 1.) * cls_kappa_shear_th
    cls_ng_mag = 2.*(2.5*s) * cls_kappa_shear_th
    cls_ng_p = -2. * cls_kappa_shear_th
    # auto galaxy
    cls_nn_mag_p = 2. * 2.*(2.5*s) * cls_kappa_l_gal_th + (2.*(2.5*s))**2 * cls_kappa_l_th
# matches perfectly with mag_bias from pyccl on the number of lensing vs. cmbl_l for kappa in <kappa shear> and also number, number both with pyccl mag bias
#mag_bias = (z_cent, np.ones_like(z_cent)*s)
#number = ccl.tracers.NumberCountsTracer(cosmo, dndz=(z_cent, dNdz_l), has_rsd=False, bias=(z_cent, bz_l), mag_bias=mag_bias)#, use_A_ia=True) # rsd
#cls_gal_th = ccl.angular_cl(cosmo, number, number, ell)
#cls_gal_shear_th = ccl.angular_cl(cosmo, number, weak, ell) # i think
#print("nn modification mag bias pyccl = ", cls_gal_th[:10])
if want_shear:
    print("gal shear mag bias = ", (cls_gal_shear_th + cls_ng_mag_p)[:10])
    print("nn mag bias = ", (cls_nn_mag_p+cls_gal_th)[:10])

if want_shear:
    # shear auto
    xip = ccl.correlation(cosmo, ell, cls_shear_th, theta, type='GG+') # number, lensing
    xim = ccl.correlation(cosmo, ell, cls_shear_th, theta, type='GG-') # number, lensing
    np.savez(f"data/GG_{sim_name}_{tracer}{rsd_str}_zs{z_s:.3f}_ccl.npz", theta=theta, xip=xip, xim=xim)
    # shear cross
    gammat = ccl.correlation(cosmo, ell, cls_gal_shear_th, theta, type='NG') # number, lensing
    gammat_mag = ccl.correlation(cosmo, ell, cls_gal_shear_th+cls_ng_mag, theta, type='NG') # number, lensing
    gammat_p = ccl.correlation(cosmo, ell, cls_gal_shear_th+cls_ng_p, theta, type='NG') # number, lensing
    gammat_mag_p = ccl.correlation(cosmo, ell, cls_gal_shear_th+cls_ng_mag_p, theta, type='NG') # number, lensing
    np.savez(f"data/NG_{sim_name}_{tracer}{rsd_str}_zl{z_l:.3f}_zs{z_s:.3f}_ccl.npz", theta=theta, gammat=gammat, gammat_mag_p=gammat_mag_p, gammat_mag=gammat_mag, gammat_p=gammat_p)
    # galaxy auto (technically doesn't need to be here, but because of load neighbors for now)
    nn = ccl.correlation(cosmo, ell, cls_gal_th, theta, type='NN') # number, number
    nn_mag_p = ccl.correlation(cosmo, ell, cls_gal_th + cls_nn_mag_p, theta, type='NN') # number, number
    np.savez(f"data/NN_{sim_name}_{tracer}{rsd_str}_zl{z_l:.3f}_ccl.npz", theta=theta, nn=nn, nn_mag_p=nn_mag_p)

want_plot = False
if want_plot:
    theta *= 60. # back to arcmin
    plt.plot(theta, xip, color='blue')
    plt.plot(theta, -xip, color='blue', ls=':')
    plt.plot(theta[xip>0], xip[xip>0], color='blue', lw=0.1, ls='')
    plt.plot(theta[xip<0], -xip[xip<0], color='blue', lw=0.1, ls='')
    lp = plt.errorbar(-theta, xip, color='blue')

    plt.plot(theta, xim, color='green')
    plt.plot(theta, -xim, color='green', ls=':')
    plt.plot(theta[xim>0], xim[xim>0], color='green', lw=0.1, ls='')
    plt.plot(theta[xim<0], -xim[xim<0], color='green', lw=0.1, ls='')
    lm = plt.errorbar(-theta, xim, color='green')

    plt.xscale('log')
    plt.yscale('log')#, nonpositive='clip')
    plt.xlabel(r'$\theta$ (arcmin)')

    plt.legend([lp, lm], [r'$\xi_+(\theta)$', r'$\xi_-(\theta)$'])
    plt.xlim( [1,200] )
    plt.ylabel(r'$\xi_{+,-}$')
    plt.savefig("figs/xipm_ccl.png")
    plt.show()

    plt.figure()
    plt.plot(theta, gammat*theta, color='blue')
    plt.savefig("figs/gammat_ccl.png")

# convert to z (this part only helps in the cross-correlation)
#z = 1./ccl.background.scale_factor_of_chi(cosmo, chi_of_z(z)/cosmo_dic['h']) - 1.
