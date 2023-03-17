import gc, os
from pathlib import Path

import numpy as np
import asdf
import random
from scipy.interpolate import interp1d

from halotools.mock_observables import mean_delta_sigma
from abacusnbody.metadata import get_meta
from abacusnbody.hod.tpcf_corrfunc import calc_wp_fast
from astropy import constants as c
from astropy import units as u
from astropy.io import ascii
#from Corrfunc.theory import xirppi

from bitpacked import unpack_rvint

# factor multiplying sigma_crit
_sigma_crit_factor = (c.c**2 / (4 * np.pi * c.G)).to(u.Msun / u.Mpc).value # Msun/Mpc

def get_sigma_crit(z_l, z_s, d_l, d_s, comoving=True):
    """ function for sigma_crit; d_s and d_l comoving distance in cMpc """
    dist_term = ((d_s / (1 + z_s)) / (d_l / (1 + z_l))) / ((d_s - d_l) / (1 + z_s)) # Mpc^-1
    # d_s/(d_l * (d_s - d_l)) * (1+z_l)
    if comoving:
        dist_term /= (1.0 + z_l)**2 # cMpc^-1 / (1+z_l)
    return _sigma_crit_factor * dist_term # Msun/cMpc^2

def load_halo_field_down(sim_dir, z, n_chunks, N_parts, f_down, type_AB='A'):
    # estimate total number of particles and preselect indices
    N_all = 0
    N_offset = np.zeros(n_chunks, dtype=int)
    N_file = np.zeros(n_chunks, dtype=int)
    for i_chunk in range(n_chunks):
        print(i_chunk, n_chunks)

        # halo and field particles
        fn_halo = sim_dir+f'/halos/z{z:.3f}/halo_rv_{type_AB}/halo_rv_{type_AB}_{i_chunk:03d}.asdf'
        fn_field = sim_dir+f'/halos/z{z:.3f}/field_rv_{type_AB}/field_rv_{type_AB}_{i_chunk:03d}.asdf'
        N_this = asdf.open(fn_halo)['data']['rvint'].shape[0]+asdf.open(fn_field)['data']['rvint'].shape[0]
        N_offset[i_chunk] = N_all
        N_file[i_chunk] = N_this
        N_all += N_this
    gc.collect()
    print("offsets", N_offset)
    print("per file", N_file)
    print("all the particles in the halo and field files", N_all)

    # global indices to keep
    #inds_keep = np.arange(N_all)
    #np.random.shuffle(inds_keep)
    #inds_keep = inds_keep[::f_down]
    inds_keep = random.sample(range(N_all), N_all//f_down)
    N_keep = len(inds_keep)
    print("N_keep", N_keep)
    pos_down = np.zeros((N_keep, 3), dtype=np.float32)
    vel_down = np.zeros((N_keep, 3), dtype=np.float32)
    
    # load the matter particles
    count = 0
    for i_chunk in range(n_chunks):
        print(i_chunk, n_chunks)
        # indices to keep in this chunk
        inds_keep_this = inds_keep - N_offset[i_chunk]
        inds_keep_this = inds_keep_this[(inds_keep_this >= 0) & (inds_keep_this < N_file[i_chunk])]
        
        # halo and field particles
        fn_halo = sim_dir+f'/halos/z{z:.3f}/halo_rv_{type_AB}/halo_rv_{type_AB}_{i_chunk:03d}.asdf'
        fn_field = sim_dir+f'/halos/z{z:.3f}/field_rv_{type_AB}/field_rv_{type_AB}_{i_chunk:03d}.asdf'

        # write out the halo (L0+L1) matter particles
        halo_data = (asdf.open(fn_halo)['data'])['rvint']
        pos_halo, vel_halo = unpack_rvint(halo_data, Lbox, float_dtype=np.float32, velout=None)
        print("pos_halo = ", pos_halo[:5])

        # write out the field matter particles
        field_data = (asdf.open(fn_field)['data'])['rvint']
        pos_field, vel_field = unpack_rvint(field_data, Lbox, float_dtype=np.float32, velout=None)
        print("pos_field = ", pos_field[:5])

        # stack halo and field particles
        pos_both = np.vstack((pos_halo, pos_field))
        vel_both = np.vstack((vel_halo, vel_field))

        # additional randomization cause am freaking out
        inds = np.arange(pos_both.shape[0])
        np.random.shuffle(inds)
        pos_both = pos_both[inds]
        vel_both = vel_both[inds]
        #inds = inds[::f_down]

        # write out pos and vel
        pos_down[count:count+len(inds_keep_this)] = pos_both[inds_keep_this]
        vel_down[count:count+len(inds_keep_this)] = vel_both[inds_keep_this]
        count += len(inds_keep_this)
        del halo_data, pos_halo, vel_halo, field_data, pos_field, vel_field
        gc.collect()
    print("these two must be the same", count, pos_down.shape[0])
    pos_down = pos_down[:count]
    vel_down = vel_down[:count]
    return pos_down, vel_down

# specify parameters
sim_name = "AbacusSummit_base_c000_ph002"
Lbox = 2000. # cMpc/h
period = np.array([Lbox, Lbox, Lbox])
N_parts = 6912**3
pcle_mass = get_meta(sim_name, redshift=0.1)['ParticleMassHMsun'] # 2.11e9 # Msun/h
n_thread = 16
inv_velz2kms = 1./(get_meta(sim_name, redshift=0.1)['VelZSpace_to_kms']/Lbox)
Om_m = get_meta(sim_name, redshift=0.1)['Omega_M'] #0.315192
H0 = get_meta(sim_name, redshift=0.1)['H0']
h = H0/100. # 0.6736
scratch_dir = "/global/cscratch1/sd/boryanah/abacus_lensing/"

# galaxy and pcle parameters
want_rsd = True
rsd_str = "_rsd" if want_rsd else ""
#tracer = "ELG"; z_l = 0.8; z_s = 1.4
tracer = "LRG"; z_l = 0.5; z_s = 1.025 
fn_gals = f"{scratch_dir}/mocks_box_output/{sim_name}/z{z_l:.3f}/galaxies{rsd_str}/{tracer}s.dat"
sim_dir = f"/global/project/projectdirs/desi/cosmosim/Abacus/{sim_name}/" #f"{scratch_dir}/{sim_name}/"
n_chunks = 34

# location of headers
header_dir = f"/global/homes/b/boryanah//repos/abacus_lc_cat/data_headers/{sim_name}/"

# all snapshots and redshifts that have light cones; early to recent redshifts
zs_all = np.load(header_dir+"redshifts.npy")

# ordered from small to large; small step number to large step number
steps_all = np.load(header_dir+"steps.npy")

# comoving distances in Mpc/h; far shells to close shells
chis_all = np.load(header_dir+"coord_dist.npy")
chis_all /= h # Mpc

# get functions relating chi and z
z_min = 0.1
chi_min = get_meta(sim_name, redshift=z_min)['CoordinateDistanceHMpc']/h
z_edges = np.append(zs_all, np.array([z_min]))
chi_edges = np.append(chis_all, np.array([chi_min]))
chi_of_z = interp1d(z_edges, chi_edges)
z_of_chi = interp1d(chi_edges, z_edges)

# transverse comoving distance
d_l = chi_of_z(z_l) # Mpc
d_s = chi_of_z(z_s) # Mpc
print("d_l, d_s", d_l, d_s)

# mean matter density in units of h^2Msun/Mpc^3
rho_matter_mean = N_parts*pcle_mass/Lbox**3 # Msun/h/(cMpc/h)^3 = h^2Msun/cMpc^3

# load galaxies
f = ascii.read(fn_gals)
pos_gals = np.vstack((f['x'], f['y'], f['z'])).T
N_gal = pos_gals.shape[0] # 6813602
print("number of galaxies", N_gal)
print("min max galaxies", pos_gals.min(), pos_gals.max())

# load particles
f_down = 1000 # 30 as many as galaxies
parts_fn = f"{sim_dir}/pos_vel_parts_z{z_l:.3f}_{f_down:d}.npz"
if os.path.exists(parts_fn):
    data = np.load(parts_fn)
    pos_parts = data['pos_parts']
    vel_parts = data['vel_parts']
else:
    pos_parts, vel_parts = load_halo_field_down(sim_dir, z_l, n_chunks, N_parts, f_down)
    np.savez(parts_fn, pos_parts=pos_parts, vel_parts=vel_parts)
N_part = pos_parts.shape[0]
print("downsampled particles more than galaxies by", N_part/N_gal)

# specify R bins
rp_bins = np.logspace(-1.5, 2, 31) # cMpc/h
rp_mids = .5*(rp_bins[1:]+rp_bins[1:])

# convert rp_mids to theta_mids at z_l
th_mids = rp_mids/(d_l*h)*180./np.pi*60. # arcmin

# compute Sigma_crit
Sigma_crit = get_sigma_crit(z_l, z_s, d_l, d_s, comoving=True)/h # hMsun/cMpc^2

# add rsd to particles
if want_rsd:
    pos_parts[:, 2] += vel_parts[:, 2]*inv_velz2kms
    del vel_parts; gc.collect()

# wrap around box
pos_gals %= Lbox
pos_parts %= Lbox

# further downsampling by a factor of X
print("additional downsampling")
X = 2
#inds = np.arange(pos_parts.shape[0])
#np.random.shuffle(inds)
#inds = inds[::X]
inds = random.sample(range(pos_parts.shape[0]), pos_parts.shape[0]//X)
pos_parts = pos_parts[inds]

# true downsampling
N_part = pos_parts.shape[0]
f_down = N_parts/N_part
print("true downsampling factor", f_down)

# adjust effective particle mass
pcle_mass *= f_down

# compute from the box
Delta_Sigma = mean_delta_sigma(pos_gals, pos_parts, pcle_mass, rp_bins, period, num_threads=n_thread) # hMsun/cMpc^2
# pretty much the same amplitude but offset -- need to do derivative
"""
pimax = 500 # 1000 complains
wp = calc_wp_fast(pos_gals[:, 0], pos_gals[:, 1], pos_gals[:, 2], rp_bins, pimax,
                  Lbox, n_thread, num_cells = 30, x2=pos_parts[:, 0], y2=pos_parts[:, 1], z2=pos_parts[:, 2]) # cMpc/h
Delta_Sigma = rho_matter_mean * wp # h^2 Msun/cMpc^3 cMpc/h = hMsun/cMpc^2
np.save(f"data/Delta_Sigma{rsd_str}_{tracer}_zl{z_l:.3f}_zs{z_s:.3f}_{sim_name}.npy", Delta_Sigma)
data = np.load(f"data/gamma_t{rsd_str}_{tracer}_zl{z_l:.3f}_zs{z_s:.3f}_{sim_name}.npz")
DS = data['Delta_Sigma']
print(Delta_Sigma/DS)
quit()
"""
# calculate ggl
gamma_t = Delta_Sigma/Sigma_crit

# save measurement
np.savez(f"data/gamma_t{rsd_str}_{tracer}_zl{z_l:.3f}_zs{z_s:.3f}_{sim_name}.npz", gamma_t=gamma_t, rp_mids=rp_mids, th_mids=th_mids, Sigma_crit=Sigma_crit, Delta_Sigma=Delta_Sigma, f_down=f_down)

