import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time

# Paths and stuff
import os
import sys

sys.path.append('/home/shreyas/pySICOPOLIS/src')
from pySICOPOLIS import *

if __name__ == "__main__":

    ds_grl40_bm5_paleo17a_CT4_BH0_m21ka_pkp_init = xr.open_dataset('/home/shreyas/update_to_develop_sicopolis/sicopolis_spinups/sico_out/grl40_bm5_paleo17a_CT4_BH0_8point5CS_spinup_0ka/grl40_bm5_paleo17a_CT4_BH0_8point5CS_spinup_0ka0005.nc')
    surfvel_data_40kms = xr.open_dataset("/scratch2/shreyas/GrIS_paleo_data/surfvel_data_40kms.nc")
    vx_uncert_data = surfvel_data_40kms["vx_uncert_80"].data
    vy_uncert_data = surfvel_data_40kms["vy_uncert_80"].data

    sicopolis_dir = '/home/shreyas/update_to_develop_sicopolis/sicopolis_REVD_3point5_80_new_pickup_dp'
    simulation = 'grl40_bm5_paleo17a_nudged_CT4_BH0_SVC_m21ka_pkp'
    dict_sico_out_folder_prefixes = {"nodiff": "N",
                                     "tlm": "FORWARD",
                                     "adj": "ADJOINT",
                                     "tlm_action": "FORWARDHESSACTION",
                                     "adj_action": "ADJHESSACTION"}
    dict_ad_exec_cmds_suffixes = {"nodiff": "nodiff",
                                  "tlm": "forward",
                                  "adj": "adjoint",
                                  "tlm_action": "forwardhessaction",
                                  "adj_action": "adjointhessaction"}
    dict_ad_log_file_suffixes = {"nodiff": "nodiff",
                                 "tlm": "tlm",
                                 "adj": "adj",
                                 "tlm_action": "tlm_hessaction",
                                 "adj_action": "adj_hessaction"}
    dict_ad_nc_suffixes = {"nodiff": "nodiff",
                           "tlm": "tlm",
                           "adj": "adj",
                           "tlm_action": "tlm_hessaction",
                           "adj_action": "adj_hessaction"}
    
    KCMAX = 80
    exp_sigma_level = dataCleaner.exp_sigma_level(zeta = np.arange(0,1+1./KCMAX,1./KCMAX),
                                                  exponent = 2.0)
    KRMAX = 40
    zeta_r = np.arange(0.,1. + 1.0/KRMAX, 1.0/KRMAX)
    xModel40       = np.arange(-72.,97.,4.0)*10
    yModel40       = np.arange(-345.,-56.,4.0)*10
    time_ad = np.arange(22, dtype=float)
    IMAX = xModel40.shape[0]-1
    JMAX = yModel40.shape[0]-1
    NTDAMAX = time_ad.shape[0]-1
    
    log_c_slide_init = np.log10(0.85)*np.ones((JMAX+1, IMAX+1))
    data = ds_grl40_bm5_paleo17a_CT4_BH0_m21ka_pkp_init['q_geo'].data
    log_q_geo = np.where(data > 0, np.log10(data), -10)
    log_p_weert = np.log10(3.0)
    log_q_weert = np.log10(2.0)
    log_enh_fact_da_dummy2d_scalar = np.log10(3.0)
    log_enh_intg_da_dummy2d_scalar = np.log10(1.0)
    log_n_glen_da_dummy2d_scalar = np.log10(3.0)
    
    # Ensure the sequence in these arrays is the same as defined in ad_specs.h
    dict_og_params_fields_vals = {"xx_c_slide_init": log_c_slide_init,
                                  "xx_q_geo": log_q_geo,
                                  "xx_p_weert": log_p_weert,
                                  "xx_q_weert": log_q_weert,
                                  "xx_enh_fact_da_dummy2d_scalar": log_enh_fact_da_dummy2d_scalar,
                                  "xx_enh_intg_da_dummy2d_scalar": log_enh_intg_da_dummy2d_scalar,
                                  "xx_n_glen_da_dummy2d_scalar": log_n_glen_da_dummy2d_scalar}
    dict_prior_params_fields_vals = dict_og_params_fields_vals.copy()
    dict_params_fields_num_dims = {"xx_c_slide_init": "2D",
                                   "xx_q_geo": "2D",
                                   "xx_p_weert": "2D",
                                   "xx_q_weert": "2D",
                                   "xx_enh_fact_da_dummy2d_scalar": "2D",
                                   "xx_enh_intg_da_dummy2d_scalar": "2D",
                                   "xx_n_glen_da_dummy2d_scalar": "2D"}
    dict_params_coords = {"time_ad": time_ad,
                          "zeta_c": exp_sigma_level,
                          "zeta_r": zeta_r,
                          "y": yModel40,
                          "x": xModel40}
    dict_params_attrs_type = {"xx_c_slide_init": "nodiff",
                              "xx_q_geo": "nodiff",
                              "xx_p_weert": "nodiff",
                              "xx_q_weert": "nodiff",
                              "xx_enh_fact_da_dummy2d_scalar": "nodiff",
                              "xx_enh_intg_da_dummy2d_scalar": "nodiff",
                              "xx_n_glen_da_dummy2d_scalar": "nodiff"}
    dict_params_fields_or_scalars = {"xx_c_slide_init": "field",
                                     "xx_q_geo": "field",
                                     "xx_p_weert": "scalar",
                                     "xx_q_weert": "scalar",
                                     "xx_enh_fact_da_dummy2d_scalar": "scalar",
                                     "xx_enh_intg_da_dummy2d_scalar": "scalar",
                                     "xx_n_glen_da_dummy2d_scalar": "scalar"}
    
    year2sec = 3.1556925445e+07
    sec2year = 1/year2sec
    dict_masks_observables = {"vx_s_g": (vx_uncert_data*sec2year)**(-2),
                              "vy_s_g": (vy_uncert_data*sec2year)**(-2)}
    
    dict_prior_sigmas = {"xx_c_slide_init": 0.3,
                         "xx_q_geo": 0.3,
                         "xx_p_weert": 0.01,
                         "xx_q_weert": 0.01,
                         "xx_enh_fact_da_dummy2d_scalar": 0.01,
                         "xx_enh_intg_da_dummy2d_scalar": 0.01,
                         "xx_n_glen_da_dummy2d_scalar": 0.01}
    dict_prior_gammas = {"xx_c_slide_init": 1.0,
                         "xx_q_geo": 1.0,
                         "xx_p_weert": 0.0,
                         "xx_q_weert": 0.0,
                         "xx_enh_fact_da_dummy2d_scalar": 0.0,
                         "xx_enh_intg_da_dummy2d_scalar": 0.0,
                         "xx_n_glen_da_dummy2d_scalar": 0.0}
    dict_prior_deltas = {"xx_c_slide_init": 2.0e-4,
                         "xx_q_geo": 2.0e-4,
                         "xx_p_weert": 1.0,
                         "xx_q_weert": 1.0,
                         "xx_enh_fact_da_dummy2d_scalar": 1.0,
                         "xx_enh_intg_da_dummy2d_scalar": 1.0,
                         "xx_n_glen_da_dummy2d_scalar": 1.0}
    
    list_fields_to_ignore = None
    
    MAX_ITERS_SOR = 100
    OMEGA_SOR = 1.5

    ds_state = xr.open_dataset("/scratch2/shreyas/optim_SVC_3point5_80_new_pickup/inexact_gn_hessian_cg/state_GNHessCG_iter_10.nc")
    ds_prior_X = xr.open_dataset("/scratch2/shreyas/optim_SVC_3point5_80_new/prior_X.nc")

    dict_og_params_fields_vals = {"xx_c_slide_init": ds_state["xx_c_slide_init"].data.copy(),
                                  "xx_q_geo": ds_state["xx_q_geo"].data.copy(),
                                  "xx_p_weert": ds_state["xx_p_weert"].data[0].copy(),
                                  "xx_q_weert": ds_state["xx_q_weert"].data[0].copy(),
                                  "xx_enh_fact_da_dummy2d_scalar": ds_state["xx_enh_fact_da_dummy2d_scalar"].data[0].copy(),
                                  "xx_enh_intg_da_dummy2d_scalar": ds_state["xx_enh_intg_da_dummy2d_scalar"].data[0].copy(),
                                  "xx_n_glen_da_dummy2d_scalar": ds_state["xx_n_glen_da_dummy2d_scalar"].data[0].copy()}

    DA = optim.DataAssimilation(sicopolis_dir, simulation,
                                dict_sico_out_folder_prefixes, dict_ad_exec_cmds_suffixes,
                                dict_ad_log_file_suffixes, dict_ad_nc_suffixes,
                                dict_og_params_fields_vals, dict_prior_params_fields_vals, dict_params_fields_num_dims,
                                dict_params_coords, dict_params_attrs_type, dict_params_fields_or_scalars, dict_masks_observables,
                                dict_prior_sigmas, dict_prior_gammas, dict_prior_deltas,
                                MAX_ITERS_SOR, OMEGA_SOR, list_fields_to_ignore, False, None, "/scratch2/shreyas/REVD_3point5_80_new_pickup_dp", None, ds_prior_X, "0007.nc")
    print(DA.ds_subset_costs["fc"].data[0])
    Omega_misfit_dp, Y_misfit_dp, Q_misfit_dp, MQ_misfit_dp, U_misfit_dp, Lambda_misfit_dp, S_misfit_dp = DA.revd(1000, 2, mode = "misfit_prior_precond", str_pass = "double_precise", output_freq = 3)
