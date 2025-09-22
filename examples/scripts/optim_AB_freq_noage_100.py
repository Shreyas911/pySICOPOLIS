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

    ds_grl40_bm5_paleo17a_CT4_BH0_m11ka_pkp_init = xr.open_dataset('/home/shreyas/update_to_develop_sicopolis/sicopolis_spinups/sico_out/grl40_bm5_paleo17a_CT4_BH0_8point5CS_spinup_0ka/grl40_bm5_paleo17a_CT4_BH0_8point5CS_spinup_0ka0006.nc')
    
    H_data_40 = xr.open_dataset("/scratch2/shreyas/GrIS_paleo_data/bm5_data_40kms.nc")
    age_data_40 = xr.open_dataset("/scratch2/shreyas/GrIS_paleo_data/age_data_40kms.nc")
    
    H_data = H_data_40["H"].data
    H_uncert_data = H_data_40["H_uncert"].data
    V_uncert_dummy2d_data = H_data_40["V_uncert_dummy2d"].data
    zl_uncert_data = H_data_40["zl_uncert"].data

    age_c_data = age_data_40["age_c"].data
    age_c_uncert_data = age_data_40["age_c_uncert"].data
    
    mask_age_c = np.zeros(age_c_data.shape)
    for kc in range(age_c_data.shape[0]):
        for j in range(age_c_data.shape[1]):
            for i in range(age_c_data.shape[2]):
                if age_c_uncert_data[kc, j, i] > 0 and age_c_data[kc, j, i] >= 0 and age_c_data[kc, j, i] <= 134000 and H_data[j, i] >= 2000.0:
                    mask_age_c[kc, j, i] = 1.0

    sicopolis_dir = '/home/shreyas/update_to_develop_sicopolis/sicopolis_optim_AB_freq_noage_100'
    simulation = 'grl40_bm5_paleo17a_CT4_BH0_AC_BM5_ZLC_m11ka_pkp'
    
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
    time_ad = np.arange(111, dtype=float)
    IMAX = xModel40.shape[0]-1
    JMAX = yModel40.shape[0]-1
    NTDAMAX = time_ad.shape[0]-1

    ds_state_phaseI = xr.open_dataset("/scratch2/shreyas/optim_SVC_3point5_50_new_pickup_2/inexact_gn_hessian_cg/state_GNHessCG_iter_25.nc")

    log_c_slide_init = ds_state_phaseI["xx_c_slide_init"].data.copy()
    delta_tda_const = np.zeros((JMAX+1, IMAX+1), dtype=float)
    log_c_dis_da = np.log10(15659.0)
    log_q_geo = ds_state_phaseI["xx_q_geo"].data.copy()
    log_gamma_s = np.log10(0.070458)*np.ones((JMAX+1, IMAX+1))
    log_s_stat = np.log10(5.0)
    log_beta1 = np.log10(2.73)
    log_beta2 = np.log10(7.28)
    log_Pmax = np.log10(0.6)
    log_mu = np.log10(9.7155)
    log_RHO_A = np.log10(3300.0)
    log_time_lag_asth = np.log10(3000.0)
    log_flex_rig_lith = np.log10(1.e25)*np.ones((JMAX+1, IMAX+1))
    zs = np.zeros((JMAX+1, IMAX+1), dtype=float)
    zl = np.zeros((JMAX+1, IMAX+1), dtype=float)
    
    temp_c = np.zeros((KCMAX+1, JMAX+1, IMAX+1), dtype=float)
    data = ds_grl40_bm5_paleo17a_CT4_BH0_m11ka_pkp_init['omega_c'].data
    log_omega_c = np.where(data > 0, np.log10(data), -5)
    data = ds_grl40_bm5_paleo17a_CT4_BH0_m11ka_pkp_init['age_c'].data
    log_age_c = np.where(data > 0, np.log10(data), 0)
    
    temp_r = np.zeros((KRMAX+1, JMAX+1, IMAX+1), dtype=float)
    
    delta_tda = np.zeros((NTDAMAX+1, JMAX+1, IMAX+1), dtype=float)
    
    dict_og_params_fields_vals = {"xx_c_slide_init": log_c_slide_init,
                                  "xx_delta_tda_const": delta_tda_const,
                                  "xx_c_dis_da": log_c_dis_da,
                                  "xx_q_geo": log_q_geo,
                                  "xx_gamma_s": log_gamma_s,
                                  "xx_s_stat": log_s_stat,
                                  "xx_beta1": log_beta1,
                                  "xx_beta2": log_beta2,
                                  "xx_Pmax": log_Pmax,
                                  "xx_mu": log_mu,
                                  "xx_RHO_A": log_RHO_A,
                                  "xx_time_lag_asth": log_time_lag_asth,
                                  "xx_flex_rig_lith": log_flex_rig_lith,
                                  "xx_zs": zs,
                                  "xx_zl": zl,
                                  "xx_temp_c": temp_c,
                                  "xx_omega_c": log_omega_c,
                                  "xx_age_c": log_age_c,
                                  "xx_temp_r": temp_r,
                                  "xx_delta_tda": delta_tda}
    dict_prior_params_fields_vals = dict_og_params_fields_vals.copy()
    
    dict_params_fields_num_dims = {"xx_c_slide_init": "2D",
                                   "xx_delta_tda_const": "2D",
                                   "xx_c_dis_da": "2D",
                                   "xx_q_geo": "2D",
                                   "xx_gamma_s": "2D",
                                   "xx_s_stat": "2D",
                                   "xx_beta1": "2D",
                                   "xx_beta2": "2D",
                                   "xx_Pmax": "2D",
                                   "xx_mu": "2D",
                                   "xx_RHO_A": "2D",
                                   "xx_time_lag_asth": "2D",
                                   "xx_flex_rig_lith": "2D",
                                   "xx_zs": "2D",
                                   "xx_zl": "2D",
                                   "xx_temp_c": "3D",
                                   "xx_omega_c": "3D",
                                   "xx_age_c": "3D",
                                   "xx_temp_r": "3DR",
                                   "xx_delta_tda": "2DT"}
    
    dict_params_coords = {"time_ad": time_ad,
                          "zeta_c": exp_sigma_level,
                          "zeta_r": zeta_r,
                          "y": yModel40,
                          "x": xModel40}
    
    dict_params_attrs_type = {"xx_c_slide_init": "nodiff",
                              "xx_delta_tda_const": "nodiff",
                              "xx_c_dis_da": "nodiff",
                              "xx_q_geo": "nodiff",
                              "xx_gamma_s": "nodiff",
                              "xx_s_stat": "nodiff",
                              "xx_beta1": "nodiff",
                              "xx_beta2": "nodiff",
                              "xx_Pmax": "nodiff",
                              "xx_mu": "nodiff",
                              "xx_RHO_A": "nodiff",
                              "xx_time_lag_asth": "nodiff",
                              "xx_flex_rig_lith": "nodiff",
                              "xx_zs": "nodiff",
                              "xx_zl": "nodiff",
                              "xx_temp_c": "nodiff",
                              "xx_omega_c": "nodiff",
                              "xx_age_c": "nodiff",
                              "xx_temp_r": "nodiff",
                              "xx_delta_tda": "nodiff"}
    
    dict_params_fields_or_scalars = {"xx_c_slide_init": "field",
                                     "xx_delta_tda_const": "field",
                                     "xx_c_dis_da": "scalar",
                                     "xx_q_geo": "field",
                                     "xx_gamma_s": "field",
                                     "xx_s_stat": "scalar",
                                     "xx_beta1": "scalar",
                                     "xx_beta2": "scalar",
                                     "xx_Pmax": "scalar",
                                     "xx_mu": "scalar",
                                     "xx_RHO_A": "scalar",
                                     "xx_time_lag_asth": "scalar",
                                     "xx_flex_rig_lith": "field",
                                     "xx_zs": "field",
                                     "xx_zl": "field",
                                     "xx_temp_c": "field",
                                     "xx_omega_c": "field",
                                     "xx_age_c": "field",
                                     "xx_temp_r": "field",
                                     "xx_delta_tda": "field"}
    
    year2sec = 3.1556925445e+07
    dict_masks_observables = {"H": H_uncert_data**(-2),
                              "age_c": mask_age_c*(age_c_uncert_data*year2sec)**(-2),
                              "V_da_dummy2d": V_uncert_dummy2d_data**(-2)/((IMAX + 1)*(JMAX + 1))}
    
    dict_prior_sigmas = {"xx_c_slide_init": 0.3,
                         "xx_delta_tda_const": 0.3,
                         "xx_c_dis_da": 0.3,
                         "xx_q_geo": 0.3,
                         "xx_gamma_s": 0.3,
                         "xx_s_stat": 0.3,
                         "xx_beta1": 0.3,
                         "xx_beta2": 0.3,
                         "xx_Pmax": 0.3,
                         "xx_mu": 0.3,
                         "xx_RHO_A": 0.3,
                         "xx_time_lag_asth": 0.3,
                         "xx_flex_rig_lith": 0.3,
                         "xx_zs": 100.0,
                         "xx_zl": 100.0,
                         "xx_temp_c": 0.3,
                         "xx_omega_c": 1.0,
                         "xx_age_c": 1.0,
                         "xx_temp_r": 0.3,
                         "xx_delta_tda": 0.3}
    
    dict_prior_gammas = {"xx_c_slide_init": 0.0,
                         "xx_delta_tda_const": 1.0,
                         "xx_c_dis_da": 0.0,
                         "xx_q_geo": 0.0,
                         "xx_gamma_s": 1.0,
                         "xx_s_stat": 0.0,
                         "xx_beta1": 0.0,
                         "xx_beta2": 0.0,
                         "xx_Pmax": 0.0,
                         "xx_mu": 0.0,
                         "xx_RHO_A": 0.0,
                         "xx_time_lag_asth": 0.0,
                         "xx_flex_rig_lith": 1.0,
                         "xx_zs": 1.0,
                         "xx_zl": 1.0,
                         "xx_temp_c": 0.0,
                         "xx_omega_c": 0.0,
                         "xx_age_c": 0.0,
                         "xx_temp_r": 0.0,
                         "xx_delta_tda": 1.0}
    
    dict_prior_deltas = {"xx_c_slide_init": 0.0,
                         "xx_delta_tda_const": 2.e-4,
                         "xx_c_dis_da": 1.0,
                         "xx_q_geo": 0.0,
                         "xx_gamma_s": 2.e-4,
                         "xx_s_stat": 1.0,
                         "xx_beta1": 1.0,
                         "xx_beta2": 1.0,
                         "xx_Pmax": 1.0,
                         "xx_mu": 1.0,
                         "xx_RHO_A": 1.0,
                         "xx_time_lag_asth": 1.0,
                         "xx_flex_rig_lith": 2.e-4,
                         "xx_zs": 2.e-4,
                         "xx_zl": 2.e-4,
                         "xx_temp_c": 1.0,
                         "xx_omega_c": 1.0,
                         "xx_age_c": 1.0,
                         "xx_temp_r": 1.0,
                         "xx_delta_tda": 2.e-4}
    
    list_fields_to_ignore = ["xx_c_slide_init", "xx_q_geo", "xx_RHO_A", "xx_time_lag_asth", "xx_flex_rig_lith", "xx_zl", "xx_age_c", "xx_temp_r"]
    
    MAX_ITERS_SOR = 100
    OMEGA_SOR = 1.5

    DA = optim.DataAssimilation(sicopolis_dir, simulation,
                                dict_sico_out_folder_prefixes, dict_ad_exec_cmds_suffixes,
                                dict_ad_log_file_suffixes, dict_ad_nc_suffixes,
                                dict_og_params_fields_vals, dict_prior_params_fields_vals, dict_params_fields_num_dims, 
                                dict_params_coords, dict_params_attrs_type, dict_params_fields_or_scalars, dict_masks_observables,
                                dict_prior_sigmas, dict_prior_gammas, dict_prior_deltas,
                                MAX_ITERS_SOR, OMEGA_SOR, list_fields_to_ignore, False, None, "/scratch2/shreyas/optim_AB_freq_noage_100", 5000, None, "0006.nc")

    ds = DA.inexact_gn_hessian_cg(MAX_ITERS = 100, min_alpha_cg_tol = 1.e-20, init_alpha_gd = 1.e-6, min_alpha_gd_tol = 1.e-20)
