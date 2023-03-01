import sys
import time
import glob
import gc

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import treecorr
import asdf

"""
default settings are: ph002, z_l =0.5, z_s = 1.025; with RSD though I think this only matters for the comparison to theory; we vary the tracers
python get_ggl.py LRG; python get_ggl.py ELG
# QUESTION: see daniel; also apply Z cuts? use neighbors; # might wanna change Z_COSMO
"""
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# file name
default_sim_name = "AbacusSummit_base_c000_ph002"
if len(sys.argv) > 2:
    sim_name = sys.argv[2]
else:
    sim_name = default_sim_name
tracer = sys.argv[1] # "ELG", "LRG"
if tracer == "ELG":
    z_l = 0.8
    z_s = 1.4
elif tracer == "LRG":
    z_l = 0.5
    z_s = 1.025
want_rsd = True
rsd_str = "_rsd" if want_rsd else ""
load_neighbors = True
n_neighs = 6
if z_l == 0.5:
    Delta_z = 0.15
elif z_l == 0.8:
    Delta_z = 0.3
elif z_l == 1.025:
    Delta_z = 0.3 

def get_mask_ang(mask, RA, DEC, nest, lonlat=True):
    """
    function that returns a boolean mask given RA and DEC
    """
    ipix = hp.ang2pix(nside, theta=RA, phi=DEC, nest=nest, lonlat=lonlat) # RA, DEC degrees (math convention)
    return mask[ipix] == 1.
    
def load_lens(file_name_l):
    data = asdf.open(file_name_l)['data']
    RA_l = data['RA']
    DEC_l = data['DEC']
    Z_l = data['Z_COSMO']
    RA_lens_l = data['RA_lens'.upper()]
    DEC_lens_l = data['DEC_lens'.upper()]
    k_l = data['kappa'.upper()]
    RAND_RA_l = data['RAND_RA']
    RAND_DEC_l = data['RAND_DEC']
    RAND_Z_l = data['RAND_Z']
    return RA_l, DEC_l, Z_l, RA_lens_l, DEC_lens_l, k_l, RAND_RA_l, RAND_DEC_l, RAND_Z_l

for sum in range(1, 9):
    # need to repeat for each octant
    want_octant = False #f"octonly_all{sum:d} #f"octant_all{sum:d}" # "octant_all", "octonlysmooth", "octonly", "octant", False
    if not want_octant:
        if sum != 1: continue
    else:
        print(sum)
    
    # directories
    #cat_dir = "/global/cscratch1/sd/boryanah/AbacusHOD_scratch/mocks_lc_output/"; dir_name = "lensing"; sub_dir_s = sub_dir_l = "" # gqc only available at z = 0.8
    cat_dir = "/global/cfs/cdirs/desi/cosmosim/AbacusLensing/mocks/"; dir_name = ""; sub_dir_s = "halos/"; sub_dir_l = "DESI/" # z-evolved HODs
    redshift_s = f"/z{z_s:.3f}/{dir_name}/"
    redshift_l = f"/z{z_l:.3f}/{dir_name}/"
    fn_s = "catalog_halos.asdf"
    #fn_l = f"catalog_xi2d_{tracer.lower()}_main_z{z_l:.1f}_velbias_B_s_test_hod1.asdf" # gqc only available at z = 0.8
    fn_l = f"catalog_DESI_{tracer}.asdf" # z-evolved HODs
    file_name_s = cat_dir + sub_dir_s + sim_name + redshift_s + fn_s
    file_name_l = cat_dir + sub_dir_l + sim_name + redshift_l + fn_l

    # gather intel on the wl maps
    file_name_ls = sorted(glob.glob(cat_dir + sub_dir_l + sim_name + (f"/z*/{dir_name}/") + fn_l))
    z_lenss = []
    for i in range(len(file_name_ls)):
        z_lens = asdf.open(file_name_ls[i])['header']['CatalogRedshift']
        z_lenss.append(z_lens)
    z_lenss = np.sort(np.array(z_lenss))
    print("redshift lenses = ", z_lenss)
    i_lens = np.argmin(np.abs(z_lenss-z_l))
    z_neighs = []
    for i in range(1, n_neighs):
        if i_lens - i >= 0:
            z_neighs.append(z_lenss[i_lens-i])
        z_neighs.append(z_lenss[i_lens+i])
    z_neighs = np.sort(np.array(z_neighs))
    print("z neighs = ", z_neighs)
    
    # corr func parameters
    min_sep = 0.1
    max_sep = 400.
    nbins = 100
    bin_size = np.log(max_sep/min_sep) / nbins
    s = 0.2

    # setup what statistics you want
    gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
    ng = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin') #brute=True (slows down)
    kg = treecorr.KGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin') #brute=True (slows down)

    # load source catalog (new format is capitalized)
    data = asdf.open(file_name_s)['data']
    RA_s = data['RA']
    DEC_s = data['DEC']
    RA_lens_s = data['RA_lens'.upper()]
    DEC_lens_s = data['DEC_lens'.upper()]
    k_s = data['kappa'.upper()]
    g1_s = data['gamma1'.upper()]
    g2_s = data['gamma2'.upper()]
    del data

    # load mask
    z_max = np.max([z_l, z_s])
    #lens_save_dir = f"/global/cscratch1/sd/boryanah/light_cones/{sim_name}/"
    lens_save_dir = f"/global/project/projectdirs/desi/cosmosim/AbacusLensing/{sim_name}/"

    # read in file names to determine all the available z's
    mask_fns = sorted(glob.glob(lens_save_dir+f"mask_0*.asdf"))
    z_srcs = []
    for i in range(len(mask_fns)):
        z_srcs.append(asdf.open(mask_fns[i])['header']['SourceRedshift'])
    z_srcs = np.sort(np.array(z_srcs))
    print("redshift sources = ", z_srcs)

    # load the mask of the further away catalog
    #mask_fn = mask_fns[np.argmin(np.abs(z_srcs - z_max))] # the problem is that there are in betweens
    mask_fn = mask_fns[np.argmax(z_srcs - z_max >= 0.)]
    mask = asdf.open(mask_fn)['data']['mask']
    nside = asdf.open(mask_fn)['header']['HEALPix_nside']
    order = asdf.open(mask_fn)['header']['HEALPix_order']
    nest = True if order == 'NESTED' else False

    # mask sources
    choice = get_mask_ang(mask, RA_s, DEC_s, nest) # technically new for lensed but it oche (TODO)
    RA_s, DEC_s, RA_lens_s, DEC_lens_s, k_s, g1_s, g2_s = RA_s[choice], DEC_s[choice], RA_lens_s[choice], DEC_lens_s[choice], k_s[choice], g1_s[choice], g2_s[choice]
    del choice; gc.collect()

    # construct source catalog
    cat_s = treecorr.Catalog(ra=RA_s, dec=DEC_s, ra_units='deg', dec_units='deg', k=k_s, g1=g1_s, g2=g2_s) # source
    cat_lens_s = treecorr.Catalog(ra=RA_lens_s, dec=DEC_lens_s, ra_units='deg', dec_units='deg', k=k_s, g1=g1_s, g2=g2_s) # source

    # compute shear-shear
    t1 = time.time()
    gg.process(cat_s)  # Takes approx 1 minute / million objects
    if want_octant:
        np.savez(f"data/GG_{sim_name}_{tracer}{rsd_str}_{want_octant}_zs{z_s:.3f}.npz", xip=gg.xip, xim=gg.xim, r=np.exp(gg.meanlogr), err=np.sqrt(gg.varxip))
    else:
        np.savez(f"data/GG_{sim_name}_{tracer}{rsd_str}_zs{z_s:.3f}.npz", xip=gg.xip, xim=gg.xim, r=np.exp(gg.meanlogr), err=np.sqrt(gg.varxip))
    print('Time for calculating gg correlation = ', time.time()-t1)
    if sim_name != default_sim_name:
        print("we only do GG for anything that's not the default simulation")
        break
    
    # load lens catalog
    RA_l, DEC_l, Z_l, RA_lens_l, DEC_lens_l, k_l, RAND_RA_l, RAND_DEC_l, RAND_Z_l = load_lens(file_name_l)
    w_l = 1. - 2. * (2.5*s - 1.) * k_l
    if load_neighbors:
        for i, z_neigh in enumerate(z_neighs):
            redshift_l = f"/z{z_neigh:.3f}/{dir_name}/"
            file_name_l = cat_dir + sub_dir_l + sim_name + redshift_l + fn_l
            RA_l_this, DEC_l_this, Z_l_this, RA_lens_l_this, DEC_lens_l_this, k_l_this, RAND_RA_l_this, RAND_DEC_l_this, RAND_Z_l_this = load_lens(file_name_l)
            w_l_this = 1. - 2. * (2.5*s - 1.) * k_l_this
            RA_l = np.hstack((RA_l, RA_l_this)).T
            DEC_l = np.hstack((DEC_l, DEC_l_this)).T
            Z_l = np.hstack((Z_l, Z_l_this)).T
            RA_lens_l = np.hstack((RA_lens_l, RA_lens_l_this)).T
            DEC_lens_l = np.hstack((DEC_lens_l, DEC_lens_l_this)).T
            k_l = np.hstack((k_l, k_l_this)).T
            RAND_RA_l = np.hstack((RAND_RA_l, RAND_RA_l_this)).T
            RAND_DEC_l = np.hstack((RAND_DEC_l, RAND_DEC_l_this)).T
            RAND_Z_l = np.hstack((RAND_Z_l, RAND_Z_l_this)).T
            w_l = np.hstack((w_l, w_l_this)).T

        # apply cuts
        z_edges = np.linspace(0.1, 2.5, 1001)
        z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
        n_tot = len(Z_l)
        n_per = n_tot/len(z_cent)
        dNdz, _ = np.histogram(Z_l, bins=z_edges)
        inds = np.digitize(Z_l, bins=z_edges) - 1
        w_bin = dNdz/n_per
        fac = gaussian(Z_l, z_l, Delta_z/2.)
        #fac = 1. # heaviside
        fac /= w_bin[inds] # downweight many galaxies in bin
        #fac /= np.max(fac) # normaliza so that probabilities don't exceed 1
        fac /= np.max(fac[(Z_l > z_l-Delta_z/2.) & (Z_l < z_l+Delta_z/2.)])
        down = np.random.rand(n_tot) < fac
        RA_l = RA_l[down]
        DEC_l = DEC_l[down]
        Z_l = Z_l[down]
        RA_lens_l = RA_lens_l[down]
        DEC_lens_l = DEC_lens_l[down]
        k_l = k_l[down]
        w_l = w_l[down]
        n_tot = len(RAND_Z_l)
        n_per = n_tot/len(z_cent)
        dNdz, _ = np.histogram(RAND_Z_l, bins=z_edges)
        inds = np.digitize(RAND_Z_l, bins=z_edges) - 1
        w_bin = dNdz/n_per
        fac = gaussian(RAND_Z_l, z_l, Delta_z/2.)
        #fac = 1. # heaviside
        fac /= w_bin[inds] # downweight many galaxies in bin
        #fac /= np.max(fac) # normaliza so that probabilities don't exceed 1
        fac /= np.max(fac[(RAND_Z_l > z_l-Delta_z/2.) & (RAND_Z_l < z_l+Delta_z/2.)])
        down = np.random.rand(n_tot) < fac
        RAND_RA_l = RAND_RA_l[down]
        RAND_DEC_l = RAND_DEC_l[down]
        RAND_Z_l = RAND_Z_l[down]
        print("kept fraction (20-30%) = ", np.sum(down)/len(down))
        del Z_l, RAND_Z_l; gc.collect()
        print("RA DEC min, max", RA_l.min(), RA_l.max(), DEC_l.min(), DEC_l.max())
    print("number of nans before masking = ", np.sum(np.isnan(RA_l)), np.sum(np.isnan(RA_lens_l)))
        
    # mask lenses and randoms
    choice = get_mask_ang(mask, RA_l, DEC_l, nest) # technically new for lensed but it oche
    RA_l, DEC_l, RA_lens_l, DEC_lens_l, k_l, w_l = RA_l[choice], DEC_l[choice], RA_lens_l[choice], DEC_lens_l[choice], k_l[choice], w_l[choice]
    choice = get_mask_ang(mask, RAND_RA_l, RAND_DEC_l, nest)
    RAND_RA_l, RAND_DEC_l = RAND_RA_l[choice], RAND_DEC_l[choice]
    print("number of nans after masking = ", np.sum(np.isnan(RA_l)), np.sum(np.isnan(RA_lens_l)))
    del mask; gc.collect()
    
    # construct lens catalog
    cat_l = treecorr.Catalog(ra=RA_l, dec=DEC_l, ra_units='deg', dec_units='deg', k=k_l) # lens
    cat_mag_l = treecorr.Catalog(ra=RA_l, dec=DEC_l, ra_units='deg', dec_units='deg', k=k_l, w=w_l) # lens
    cat_p_l = treecorr.Catalog(ra=RA_lens_l, dec=DEC_lens_l, ra_units='deg', dec_units='deg', k=k_l) # lens
    cat_mag_p_l = treecorr.Catalog(ra=RA_lens_l, dec=DEC_lens_l, ra_units='deg', dec_units='deg', k=k_l, w=w_l) # lens
    cat_r_l = treecorr.Catalog(ra=RAND_RA_l, dec=RAND_DEC_l, ra_units='deg', dec_units='deg') # rand
    
    # once I was cool and made this work with asdf but then promptly realized I need to mask shit -- could I pass a mask and do that internally
    #cat_s = treecorr.Catalog(file_name_s, ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg', g1_col='gamma1', g2_col='gamma2', k_col='kappa')
    #cat_l = treecorr.Catalog(file_name_l, ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg')#, g1_col='gamma1', g2_col='gamma2', k_col='kappa') # lens
    #cat_l = treecorr.Catalog(file_name_l, ra_col='RA_lens', dec_col='DEC_lens', ra_units='deg', dec_units='deg', k_col='kappa') # lens
    #cat_l = treecorr.Catalog(file_name_l, ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg', k_col='kappa') # lens

    # prepare stuff needed for the lens-lens correlation function
    rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
    dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
    dd_mag_p = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
    dr_mag_p = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
    r = np.exp(gg.meanlogr)

    # compute lens-lens correlation function
    t1 = time.time()
    rr.process(cat_r_l)
    dd.process(cat_l)
    dr.process(cat_l, cat_r_l)
    xi, varxi = dd.calculateXi(rr, dr)
    dd_mag_p.process(cat_mag_p_l)
    dr_mag_p.process(cat_mag_p_l, cat_r_l)
    xi_mag_p, varxi_mag_p = dd_mag_p.calculateXi(rr, dr_mag_p)
    if want_octant:
        np.savez(f"data/NN_{sim_name}_{tracer}{rsd_str}_{want_octant}_zl{z_l:.3f}.npz", r=np.exp(dd.meanlogr), xi=xi, err=np.sqrt(varxi), xi_mag_p=xi_mag_p, err_mag_p=np.sqrt(varxi_mag_p))
    else:
        np.savez(f"data/NN_{sim_name}_{tracer}{rsd_str}_zl{z_l:.3f}.npz", r=np.exp(dd.meanlogr), xi=xi, err=np.sqrt(varxi), xi_mag_p=xi_mag_p, err_mag_p=np.sqrt(varxi_mag_p))
    print('Time for calculating nn correlation = ', time.time()-t1)

    # cross between convergence and shear (needed for mag bias)
    t1 = time.time()
    kg.process(cat_l, cat_s)
    """
    if want_octant:
        np.savez(f"data/KG_{sim_name}_{tracer}{rsd_str}_{want_octant}_zl{z_l:.3f}_zs{z_s:.3f}.npz", xi=kg.xi, xi_im=kg.xi_im, r=np.exp(kg.meanlogr))
    else:
        np.savez(f"data/KG_{sim_name}_{tracer}{rsd_str}_zl{z_l:.3f}_zs{z_s:.3f}.npz", xi=kg.xi, xi_im=kg.xi_im, r=np.exp(kg.meanlogr))
    print('Time for calculatikg kg correlation = ', time.time()-t1)
    """
    
    # compute galaxy-shear
    t1 = time.time()
    # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (is that the thing where daniel was like you need to pass the lensed source catalog???)
    ng.process(cat_l, cat_lens_s)
    #ng.process(cat_l, cat_s)
    xi = ng.xi.copy()
    # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (is that the thing where daniel was like you need to pass the lensed source catalog???)
    ng.process(cat_mag_l, cat_lens_s)
    #ng.process(cat_mag_l, cat_s)
    xi_mag = ng.xi.copy()
    # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (is that the thing where daniel was like you need to pass the lensed source catalog???)
    ng.process(cat_p_l, cat_lens_s)
    #ng.process(cat_p_l, cat_s) # og
    xi_p = ng.xi.copy()
    # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ng.process(cat_mag_p_l, cat_lens_s)
    #ng.process(cat_mag_p_l, cat_s) # og
    xi_mag_p = ng.xi.copy()

    # if giving: RA, DEC and no weights  (fake for lensed positions and for magnification)
    mag_p = 2.*(2.5*s - 1.) * kg.xi # mag and p
    mag = 2.*(2.5*s) * kg.xi # mag
    p = -2. * kg.xi # p
    no_mag_p = 0. # no mag and p

    # if giving: RA_lens, DEC_lens and no weights  (fake for magnification and not lensed positions)
    #mag_p = 2.*(2.5*s - 0.) * kg.xi # mag and p
    #mag = 2.*(2.5*s + 1.) * kg.xi # mag
    #p = 0. * kg.xi # p
    #no_mag_p = 2. * kg.xi # no mag and p

    # correlation functions in that case (ps = pseudo or fake )
    xi_ps = xi+no_mag_p
    xi_ps_mag = xi+mag
    #xi_ps_p = xi+p # fake moved positions
    xi_ps_p = xi_p
    #xi_ps_mag_p = xi+p+mag # fake moved positions
    xi_ps_mag_p = xi_p+mag

    # save differently if part of the octant series
    if want_octant:
        np.savez(f"data/NG_{sim_name}_{tracer}{rsd_str}_{want_octant}_zl{z_l:.3f}_zs{z_s:.3f}.npz", r=np.exp(ng.meanlogr), xi=xi, xi_mag=xi_mag, xi_p=xi_p, xi_mag_p=xi_mag_p, xi_ps_mag=xi_ps_mag, xi_ps_p=xi_ps_p, xi_ps_mag_p=xi_ps_mag_p, xi_ps=xi_ps)
    else:
        np.savez(f"data/NG_{sim_name}_{tracer}{rsd_str}_zl{z_l:.3f}_zs{z_s:.3f}.npz", r=np.exp(ng.meanlogr), xi=xi, xi_mag=xi_mag, xi_p=xi_p, xi_mag_p=xi_mag_p, xi_ps_mag=xi_ps_mag, xi_ps_p=xi_ps_p, xi_ps_mag_p=xi_ps_mag_p, xi_ps=xi_ps)
    print('Time for calculating ng correlation = ', time.time()-t1)


    want_plot = False
    if want_plot:
        plt.plot(r, xip, color='blue')
        plt.plot(r, -xip, color='blue', ls=':')
        plt.errorbar(r[xip>0], xip[xip>0], yerr=sig[xip>0], color='blue', lw=0.1, ls='')
        plt.errorbar(r[xip<0], -xip[xip<0], yerr=sig[xip<0], color='blue', lw=0.1, ls='')
        lp = plt.errorbar(-r, xip, yerr=sig, color='blue')

        plt.plot(r, xim, color='green')
        plt.plot(r, -xim, color='green', ls=':')
        plt.errorbar(r[xim>0], xim[xim>0], yerr=sig[xim>0], color='green', lw=0.1, ls='')
        plt.errorbar(r[xim<0], -xim[xim<0], yerr=sig[xim<0], color='green', lw=0.1, ls='')
        lm = plt.errorbar(-r, xim, yerr=sig, color='green')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\theta$ (arcmin)')

        plt.legend([lp, lm], [r'$\xi_+(\theta)$', r'$\xi_-(\theta)$'])
        plt.xlim([1, 200])
        plt.ylabel(r'$\xi_{+,-}$')
        plt.savefig("figs/xipm.png")
        plt.close()
        #plt.show()

        plt.plot(r, ng.xi*r, color='green')
        plt.plot(r, ng.xi_im*r, color='green', ls=':')
        plt.savefig("figs/gammat.png")
        plt.close()

        plt.plot(r, kg.xi*r, color='green')
        plt.plot(r, kg.xi_im*r, color='green', ls=':')
        plt.savefig("figs/kappag.png")
        plt.close()
