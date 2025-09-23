import numpy as np
import xarray as xr
import subprocess

from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Tuple, Union
from jaxtyping import Float

import os
import warnings

__all__ = ["DataAssimilation"]

class DataAssimilation:

    @beartype
    def __init__(self,
                 sicopolis_dir: str,
                 simulation: str,
                 dict_sico_out_folder_prefixes: Dict[str, str],
                 dict_ad_exec_cmds_suffixes: Dict[str, str],
                 dict_ad_log_file_suffixes: Dict[str, str],
                 dict_ad_nc_suffixes: Dict[str, str],
                 dict_og_params_fields_vals: Dict[str, Union[Float[np.ndarray, "dimz dimy dimx"],
                                                             Float[np.ndarray, "dimy dimx"],
                                                             float]],
                 dict_prior_params_fields_vals: Dict[str, Union[Float[np.ndarray, "dimz dimy dimx"],
                                                                Float[np.ndarray, "dimy dimx"],
                                                                float]],
                 dict_params_fields_num_dims: Dict[str, str],
                 dict_params_coords: Dict[str, Float[np.ndarray, "dim"]],
                 dict_params_attrs_type: Dict[str, str],
                 dict_params_fields_or_scalars: Dict[str, str],
                 dict_masks_observables: Dict[str, Optional[Union[Float[np.ndarray, "dimz dimy dimx"], 
                                                                  Float[np.ndarray, "dimy dimx"]]]],
                 dict_prior_sigmas: Dict[str, Optional[float]],
                 dict_prior_gammas: Dict[str, Optional[float]],
                 dict_prior_deltas: Dict[str, Optional[float]],
                 MAX_ITERS_SOR: int = 100,
                 OMEGA_SOR: float = 1.5,
                 list_fields_to_ignore: Optional[List[str]] = None,
                 bool_vs_cost: bool = False,
                 filename_vx_vy_s_g: Optional[str] = None,
                 dirpath_store_states: Optional[str] = None,
                 num_prior_samples: Optional[int] = 1000,
                 ds_prior_X: Optional[Any] = None,
                 filename_final_sim_output: Optional[str] = None) -> None:

        super().__init__()

        if not os.path.isdir(sicopolis_dir):
            raise ValueError(f"DataAssimilation: {sicopolis_dir} doesn't seem to be an existing directory.")

        self.sicopolis_dir = sicopolis_dir
        self.src_dir = sicopolis_dir + "/src"
        self.ad_io_dir = sicopolis_dir + "/src/subroutines/tapenade/ad_io"
        self.sico_out_dir = sicopolis_dir + "/sico_out"

        self.ad_keys = ["nodiff", "tlm", "adj", "tlm_action", "adj_action"]
        if (
            not self.has_exact_keys(dict_sico_out_folder_prefixes, self.ad_keys) or
            not self.has_exact_keys(dict_ad_exec_cmds_suffixes, self.ad_keys) or
            not self.has_exact_keys(dict_ad_log_file_suffixes, self.ad_keys) or
            not self.has_exact_keys(dict_ad_nc_suffixes, self.ad_keys)
           ):
            raise ValueError("DataAssimilation: Inconsistent ad_keys.")

        self.dict_sico_out_folders = {key: self.sico_out_dir + "/" + prefix + "_" + simulation for key, prefix in dict_sico_out_folder_prefixes.items()}
        self.dict_ad_exec_cmds = {key: "./driver" + suffix for key, suffix in dict_ad_exec_cmds_suffixes.items()}
        self.dict_ad_log_files = {key: self.src_dir + "/log_" + simulation + "_" + suffix + ".txt" for key, suffix in dict_ad_log_file_suffixes.items()}
        self.dict_ad_inp_nc_files = {key: self.ad_io_dir + "/ad_input_" + suffix + ".nc" for key, suffix in dict_ad_nc_suffixes.items()}
        self.dict_ad_out_nc_files = {key: self.ad_io_dir + "/ad_output_" + suffix + ".nc" for key, suffix in dict_ad_nc_suffixes.items()}

        if dict_og_params_fields_vals.keys() != dict_prior_params_fields_vals.keys() != dict_params_fields_num_dims.keys() != dict_params_coords.keys() != dict_params_attrs_type.keys() != dict_params_fields_or_scalars.keys():
            raise ValueError("DataAssimilation: Inconsistent keys for OG state.")
 
        self.dict_og_params_fields_vals = dict_og_params_fields_vals
        self.dict_prior_params_fields_vals = dict_prior_params_fields_vals
        self.dict_params_fields_num_dims = dict_params_fields_num_dims
        self.dict_params_coords = dict_params_coords
        self.dict_params_attrs_type = dict_params_attrs_type
        self.dict_params_fields_or_scalars = dict_params_fields_or_scalars

        self.dict_tlm_action_fields_num_dims = self.create_dict_tlm_action(dict_params_fields_num_dims)
        self.dict_tlm_action_coords = self.create_dict_tlm_action(dict_params_coords)
        self.dict_tlm_action_attrs_type = self.create_dict_tlm_action(dict_params_attrs_type, "tlm")
        self.dict_tlm_action_fields_or_scalars = self.create_dict_tlm_action(dict_params_fields_or_scalars)

        self.NTDAMAX = dict_params_coords["time_ad"].shape[0] - 1
        self.KCMAX   = dict_params_coords["zeta_c"].shape[0] - 1
        self.KRMAX   = dict_params_coords["zeta_r"].shape[0] - 1
        self.JMAX    = dict_params_coords["y"].shape[0] - 1
        self.IMAX    = dict_params_coords["x"].shape[0] - 1
        self.delta_x = dict_params_coords["x"][1] - dict_params_coords["x"][0]
        self.delta_y = dict_params_coords["y"][1] - dict_params_coords["y"][0]

        self.dict_masks_observables = dict_masks_observables

        if dict_prior_params_fields_vals.keys() != dict_prior_sigmas.keys() != dict_prior_gammas.keys() != dict_prior_deltas.keys():
            raise ValueError("DataAssimilation: Inconsistent keys for prior dicts.")

        if not all(isinstance(value, float) for value in dict_prior_sigmas.values()):
            raise ValueError("DataAssimilation: All sigma values should be floats.")

        self.dict_prior_sigmas = dict_prior_sigmas
        self.dict_prior_gammas = dict_prior_gammas
        self.dict_prior_deltas = dict_prior_deltas

        if not (1.0 <= OMEGA_SOR <= 2.0):
            raise ValueError("DataAssimilation: Relaxation factor for SOR solver should be between 1 and 2.")

        self.MAX_ITERS_SOR = MAX_ITERS_SOR
        self.OMEGA_SOR = OMEGA_SOR

        self.list_fields_to_ignore = list_fields_to_ignore

        if bool_vs_cost and filename_vx_vy_s_g is None:
            raise ValueError("DataAssimilation: File to read terminal surface velocity field values not specified.")
        
        if not bool_vs_cost and filename_vx_vy_s_g is not None:
            raise ValueError("DataAssimilation: File to read terminal surface velocity field values specified even when not needed.")

        self.bool_vs_cost = bool_vs_cost
        self.filename_vx_vy_s_g = filename_vx_vy_s_g
        self.dirpath_store_states = dirpath_store_states

        self.ds_subset_params = self.create_ad_nodiff_or_adj_input_nc(dict_og_params_fields_vals, dict_params_fields_num_dims,
                                                  dict_params_coords, dict_params_attrs_type, dict_params_fields_or_scalars,
                                                  "nodiff", None)
        _ = self.create_ad_nodiff_or_adj_input_nc(dict_og_params_fields_vals, dict_params_fields_num_dims,
                                                  dict_params_coords, dict_params_attrs_type, dict_params_fields_or_scalars,
                                                  "adj", None)

        for var in self.dict_params_fields_or_scalars:

            if self.dict_params_fields_or_scalars and self.dict_params_fields_or_scalars[var] == "scalar":

                if var in self.ds_subset_params:
                    self.ds_subset_params[var] = xr.DataArray([self.ds_subset_params[var].data.flat[0]],
                                                          dims=["scalar"], attrs=self.ds_subset_params[var].attrs)
                else:
                    raise ValueError(f"DataAssimilation: {var} not present in self.ds_subset_params.")

        _ = self.create_ad_nodiff_or_adj_input_nc(dict_prior_params_fields_vals, dict_params_fields_num_dims,
                                                  dict_params_coords, dict_params_attrs_type, dict_params_fields_or_scalars,
                                                  None, "ad_input_nodiff_prior.nc")

        self.ds_prior_fields = self.open_xr_ds(self.ad_io_dir + "/ad_input_nodiff_prior.nc")

        # Ensure the sequence in these arrays is the same as defined in ad_specs.h

        if any(value == "2D" for value in self.dict_params_fields_num_dims.values()):
            self.dict_prior_sigmas_2d = {key: self.dict_prior_sigmas[key] for key in self.dict_prior_sigmas if self.dict_params_fields_num_dims[key] == "2D"}
            self.ds_prior_fields["genarr2d_sigma_arr"] = xr.DataArray(list(self.dict_prior_sigmas_2d.values()), dims=["genarr2d"], attrs={"type": "hyperparameter_prior"})
            self.dict_prior_gammas_2d = {key: self.dict_prior_gammas[key] for key in self.dict_prior_gammas if self.dict_params_fields_num_dims[key] == "2D"}
            self.ds_prior_fields["genarr2d_gamma_arr"] = xr.DataArray(list(self.dict_prior_gammas_2d.values()), dims=["genarr2d"], attrs={"type": "hyperparameter_prior"})
            self.dict_prior_deltas_2d = {key: self.dict_prior_deltas[key] for key in self.dict_prior_deltas if self.dict_params_fields_num_dims[key] == "2D"}
            self.ds_prior_fields["genarr2d_delta_arr"] = xr.DataArray(list(self.dict_prior_deltas_2d.values()), dims=["genarr2d"], attrs={"type": "hyperparameter_prior"})

        if any(value == "3D" for value in self.dict_params_fields_num_dims.values()):
            self.dict_prior_sigmas_3d = {key: self.dict_prior_sigmas[key] for key in self.dict_prior_sigmas if self.dict_params_fields_num_dims[key] == "3D"}
            self.ds_prior_fields["genarr3d_sigma_arr"] = xr.DataArray(list(self.dict_prior_sigmas_3d.values()), dims=["genarr3d"], attrs={"type": "hyperparameter_prior"})
            self.dict_prior_gammas_3d = {key: self.dict_prior_gammas[key] for key in self.dict_prior_gammas if self.dict_params_fields_num_dims[key] == "3D"}
            self.ds_prior_fields["genarr3d_gamma_arr"] = xr.DataArray(list(self.dict_prior_gammas_3d.values()), dims=["genarr3d"], attrs={"type": "hyperparameter_prior"})
            self.dict_prior_deltas_3d = {key: self.dict_prior_deltas[key] for key in self.dict_prior_deltas if self.dict_params_fields_num_dims[key] == "3D"}
            self.ds_prior_fields["genarr3d_delta_arr"] = xr.DataArray(list(self.dict_prior_deltas_3d.values()), dims=["genarr3d"], attrs={"type": "hyperparameter_prior"})

        if any(value == "3DR" for value in self.dict_params_fields_num_dims.values()):
            self.dict_prior_sigmas_3dr = {key: self.dict_prior_sigmas[key] for key in self.dict_prior_sigmas if self.dict_params_fields_num_dims[key] == "3DR"}
            self.ds_prior_fields["genarr3dr_sigma_arr"] = xr.DataArray(list(self.dict_prior_sigmas_3dr.values()), dims=["genarr3dr"], attrs={"type": "hyperparameter_prior"})
            self.dict_prior_gammas_3dr = {key: self.dict_prior_gammas[key] for key in self.dict_prior_gammas if self.dict_params_fields_num_dims[key] == "3DR"}
            self.ds_prior_fields["genarr3dr_gamma_arr"] = xr.DataArray(list(self.dict_prior_gammas_3dr.values()), dims=["genarr3dr"], attrs={"type": "hyperparameter_prior"})
            self.dict_prior_deltas_3dr = {key: self.dict_prior_deltas[key] for key in self.dict_prior_deltas if self.dict_params_fields_num_dims[key] == "3DR"}
            self.ds_prior_fields["genarr3dr_delta_arr"] = xr.DataArray(list(self.dict_prior_deltas_3dr.values()), dims=["genarr3dr"], attrs={"type": "hyperparameter_prior"})

        if any(value == "2DT" for value in self.dict_params_fields_num_dims.values()):
            self.dict_prior_sigmas_2dt = {key: self.dict_prior_sigmas[key] for key in self.dict_prior_sigmas if self.dict_params_fields_num_dims[key] == "2DT"}
            self.ds_prior_fields["gentim2d_sigma_arr"] = xr.DataArray(list(self.dict_prior_sigmas_2dt.values()), dims=["gentim2d"], attrs={"type": "hyperparameter_prior"})
            self.dict_prior_gammas_2dt = {key: self.dict_prior_gammas[key] for key in self.dict_prior_gammas if self.dict_params_fields_num_dims[key] == "2DT"}
            self.ds_prior_fields["gentim2d_gamma_arr"] = xr.DataArray(list(self.dict_prior_gammas_2dt.values()), dims=["gentim2d"], attrs={"type": "hyperparameter_prior"})
            self.dict_prior_deltas_2dt = {key: self.dict_prior_deltas[key] for key in self.dict_prior_deltas if self.dict_params_fields_num_dims[key] == "2DT"}
            self.ds_prior_fields["gentim2d_delta_arr"] = xr.DataArray(list(self.dict_prior_deltas_2dt.values()), dims=["gentim2d"], attrs={"type": "hyperparameter_prior"})

        # Some weird permission denied error if this file is not removed first.
        self.remove_dir(self.ad_io_dir + "/ad_input_nodiff_prior.nc")
        self.ds_prior_fields.to_netcdf(self.ad_io_dir + "/ad_input_nodiff_prior.nc")

        if self.dirpath_store_states is not None:

            if not os.path.isdir(self.dirpath_store_states):
                self.make_dir(self.dirpath_store_states)

            self.remove_dir(self.dirpath_store_states + "/prior_fields.nc")
            self.ds_prior_fields.to_netcdf(self.dirpath_store_states + "/prior_fields.nc")

        if ds_prior_X is not None:

            self.ds_prior_X = ds_prior_X

        elif num_prior_samples is not None:

            if num_prior_samples > 0:
                self.num_prior_samples = num_prior_samples
                self.ds_prior_C_mean, self.ds_prior_C_std = self.pointwise_marginals("prior_C", self.num_prior_samples)
                self.ds_prior_X = self.ds_prior_C_std**(-1)

            else:
                raise ValueError("DataAssimilation: Number of prior samples has to be a positive integer when prior_X is None.")

        else:

            raise ValueError("DataAssimilation: Both ds_prior_X and num_prior_samples should not be defined.")

        dict_tlm_action_only_fields_vals = {}
        for var in self.ds_prior_X:

            if not self.list_fields_to_ignore or (self.list_fields_to_ignore and var[:-1] not in self.list_fields_to_ignore):
                if self.dict_tlm_action_fields_or_scalars[var] == "scalar" and self.dict_tlm_action_fields_num_dims[var] == "2D":
                    dict_tlm_action_only_fields_vals[var] = self.ds_prior_X[var].data.flat[0].copy() * ((self.IMAX+1)*(self.JMAX+1))**0.5
                elif self.dict_tlm_action_fields_or_scalars[var] == "scalar" and self.dict_tlm_action_fields_num_dims[var] == "3D":
                    dict_tlm_action_only_fields_vals[var] = self.ds_prior_X[var].data.flat[0].copy() * ((self.IMAX+1)*(self.JMAX+1)*(self.KCMAX+1))**0.5
                elif self.dict_tlm_action_fields_or_scalars[var] == "scalar" and self.dict_tlm_action_fields_num_dims[var] == "3DR":
                    dict_tlm_action_only_fields_vals[var] = self.ds_prior_X[var].data.flat[0].copy() * ((self.IMAX+1)*(self.JMAX+1)*(self.KRMAX+1))**0.5
                elif self.dict_tlm_action_fields_or_scalars[var] == "scalar" and self.dict_tlm_action_fields_num_dims[var] == "2DT":
                    dict_tlm_action_only_fields_vals[var] = self.ds_prior_X[var].data.flat[0].copy() * ((self.IMAX+1)*(self.JMAX+1)*(self.NTDAMAX+1))**0.5
                else:
                    dict_tlm_action_only_fields_vals[var] = self.ds_prior_X[var].data.copy()

        # Ignored fields are given value 0, manually rectified a few lines below.
        _ =  self.create_ad_tlm_action_input_nc(dict_tlm_action_only_fields_vals)

        self.copy_dir(self.dict_ad_inp_nc_files["tlm_action"],
                      self.ad_io_dir + "/ad_input_nodiff_prior_X.nc")

        self.ds_prior_X_fields = self.open_xr_ds(self.ad_io_dir + "/ad_input_nodiff_prior_X.nc")

        if self.dirpath_store_states is not None:

            if not os.path.isdir(self.dirpath_store_states):
                self.make_dir(self.dirpath_store_states)

            self.remove_dir(self.dirpath_store_states + "/prior_X.nc")
            self.ds_prior_X.to_netcdf(self.dirpath_store_states + "/prior_X.nc")

            self.remove_dir(self.dirpath_store_states + "/prior_X_fields.nc")
            self.ds_prior_X_fields.to_netcdf(self.dirpath_store_states + "/prior_X_fields.nc")

        # Manually ensure that ignored fields don't have 0 in the X matrix, since it divides in the F90 code. Assigning a dummy value 1.0, should not matter.
        if self.list_fields_to_ignore:
            for field in self.list_fields_to_ignore:
                self.ds_prior_X_fields[field + "d"] = self.ds_prior_X_fields[field].copy()
                self.ds_prior_X_fields[field + "d"].data = self.ds_prior_X_fields[field].data*0.0 + 1.0
                self.ds_prior_X_fields[field + "d"].attrs["type"] = "tlm"

        # Some weird permission denied error if this file is not removed first.
        self.remove_dir(self.ad_io_dir + "/ad_input_nodiff_prior_X.nc")
        self.ds_prior_X_fields.to_netcdf(self.ad_io_dir + "/ad_input_nodiff_prior_X.nc")

        self.ds_subset_costs = self.eval_costs()

        if filename_vx_vy_s_g is not None and filename_final_sim_output is not None:
            raise ValueError("DataAssimilation: filename_final_sim_output should not be defined when filename_vx_vy_s_g is not None.")
        elif filename_vx_vy_s_g is not None:
            self.filename_final_sim_output = filename_vx_vy_s_g
        elif filename_final_sim_output is not None:
            self.filename_final_sim_output = filename_final_sim_output
        elif filename_final_sim_output is None:
            raise ValueError("DataAssimilation: filename_final_sim_output should be defined when filename_vx_vy_s_g is None.")

    @beartype
    def create_ad_nodiff_or_adj_input_nc(self,
                                         dict_fields_vals: Dict[str, Union[Float[np.ndarray, "dimz dimy dimx"],
                                                                           Float[np.ndarray, "dimy dimx"],
                                                                           float]],
                                         dict_fields_num_dims: Dict[str, str],
                                         dict_coords: Dict[str, Float[np.ndarray, "dim"]],
                                         dict_attrs_type: Dict[str, str],
                                         dict_fields_or_scalars: Dict[str, str],
                                         ad_key: Optional[str] = None,
                                         filename: Optional[str] = None) -> Any:
        
        if dict_fields_vals.keys() != dict_fields_num_dims.keys() != dict_coords.keys() != dict_attrs_type.keys() != dict_fields_or_scalars:
            raise ValueError("create_ad_nodiff_or_adj_input_nc: Inconsistent keys.")
        
        if ad_key is None and filename is None:
            raise ValueError("create_ad_nodiff_or_adj_input_nc: Both ad_key and filename cannot be None.")
        if ad_key is not None and filename is not None:
            raise ValueError("create_ad_nodiff_or_adj_input_nc: Both ad_key and filename cannot be defined. filename is defined when the path is not related to ad_keys, eg for prior file.")

        ds_inp_fields = xr.Dataset()

        for field in dict_fields_vals:
            
            field_val = dict_fields_vals[field]
            
            if isinstance(field_val, float) and dict_fields_or_scalars[field] == "scalar":
                
                if dict_fields_num_dims[field] == "2D":
                    
                    da_field = xr.DataArray(
                        data=field_val*np.ones((self.JMAX+1, self.IMAX+1), dtype=np.float64),
                        dims=["y", "x"],
                        coords={"y": dict_coords["y"].copy(),
                                "x": dict_coords["x"].copy()
                                },
                        name=field
                    )

                elif dict_fields_num_dims[field] == "3D":
                    
                    da_field = xr.DataArray(
                        data=field_val*np.ones((self.KCMAX+1, self.JMAX+1, self.IMAX+1), dtype=np.float64),
                        dims=["zeta_c", "y", "x"],
                        coords={"zeta_c": dict_coords["zeta_c"].copy(),
                                "y": dict_coords["y"].copy(),
                                "x": dict_coords["x"].copy()
                                },
                        name=field
                    )

                elif dict_fields_num_dims[field] == "3DR":

                    da_field = xr.DataArray(
                        data=field_val*np.ones((self.KRMAX+1, self.JMAX+1, self.IMAX+1), dtype=np.float64),
                        dims=["zeta_r", "y", "x"],
                        coords={"zeta_r": dict_coords["zeta_r"].copy(),
                                "y": dict_coords["y"].copy(),
                                "x": dict_coords["x"].copy()
                                },
                        name=field
                    )

                elif dict_fields_num_dims[field] == "2DT":
                    
                    da_field = xr.DataArray(
                        data=field_val*np.ones((self.NTDAMAX+1, self.JMAX+1, self.IMAX+1), dtype=np.float64),
                        dims=["time_ad", "y", "x"],
                        coords={"time_ad": dict_coords["time_ad"].copy(),
                                "y": dict_coords["y"].copy(),
                                "x": dict_coords["x"].copy()
                                },
                        name=field
                    )

                else:
                    raise ValueError(f"create_ad_nodiff_or_adj_input_nc: Issue with {field}; Only 2D or 3D or 3DR or 2DT fields accepted.")
                
            elif isinstance(field_val, np.ndarray) and not isinstance(field_val, (str, bytes)) and dict_fields_or_scalars[field] == "field":
                
                if dict_fields_num_dims[field] == "2D":
                    
                    if len(field_val.shape) != 2:
                        raise ValueError(f"create_ad_nodiff_or_adj_input_nc: Issue with {field}; field_val.shape != num_dims it is supposed to have.")
                    
                    da_field = xr.DataArray(
                        data=field_val.copy(),
                        dims=["y", "x"],
                        coords={"y": dict_coords["y"].copy(),
                                "x": dict_coords["x"].copy()
                                },
                        name=field
                    )
                
                elif dict_fields_num_dims[field] == "3D":
                    
                    if len(field_val.shape) != 3:
                        raise ValueError(f"create_ad_nodiff_or_adj_input_nc: Issue with {field}; field_val.shape != num_dims it is supposed to have.")
                    
                    da_field = xr.DataArray(
                        data=field_val.copy(),
                        dims=["zeta_c", "y", "x"],
                        coords={"zeta_c": dict_coords["zeta_c"].copy(),
                                "y": dict_coords["y"].copy(),
                                "x": dict_coords["x"].copy()
                                },
                        name=field
                    )

                elif dict_fields_num_dims[field] == "3DR":

                    if len(field_val.shape) != 3:
                        raise ValueError(f"create_ad_nodiff_or_adj_input_nc: Issue with {field}; field_val.shape != num_dims it is supposed to have.")

                    da_field = xr.DataArray(
                        data=field_val.copy(),
                        dims=["zeta_r", "y", "x"],
                        coords={"zeta_r": dict_coords["zeta_r"].copy(),
                                "y": dict_coords["y"].copy(),
                                "x": dict_coords["x"].copy()
                                },
                        name=field
                    )

                elif dict_fields_num_dims[field] == "2DT":
                    
                    if len(field_val.shape) != 3:
                        raise ValueError(f"create_ad_nodiff_or_adj_input_nc: Issue with {field}; field_val.shape != num_dims it is supposed to have.")
                    
                    da_field = xr.DataArray(
                        data=field_val.copy(),
                        dims=["time_ad", "y", "x"],
                        coords={"time_ad": dict_coords["time_ad"].copy(),
                                "y": dict_coords["y"].copy(),
                                "x": dict_coords["x"].copy()
                                },
                        name=field
                    )

                else:
                    raise ValueError(f"create_ad_nodiff_or_adj_input_nc: Issue with {field}; Only 2D or 3D or 3DR or 2DT fields accepted.")
                
            else:
                raise TypeError(f"create_ad_nodiff_or_adj_input_nc: Issue with {field}; The type doesn't seem to be either a scalar or a numpy array.")
            
            ds_inp_fields[field] = da_field
            if field in dict_attrs_type:
                ds_inp_fields[field].attrs["type"] = dict_attrs_type[field]

        if ad_key is not None:
            # Some weird permission denied error if this file is not removed first.
            self.remove_dir(self.dict_ad_inp_nc_files[ad_key])
            ds_inp_fields.to_netcdf(self.dict_ad_inp_nc_files[ad_key])
        else:
            # Some weird permission denied error if this file is not removed first.
            self.remove_dir(self.ad_io_dir + "/" + filename)
            ds_inp_fields.to_netcdf(self.ad_io_dir + "/" + filename)

        # Returns ds_inp_fields where even scalars are expressed as fields
        return ds_inp_fields

    @beartype
    def run_exec(self, ad_key: str) -> Any:

        if not os.path.isfile(self.dict_ad_inp_nc_files[ad_key]):
            raise ValueError(f"run_exec: AD input file {self.dict_ad_inp_nc_files[ad_key]} is missing.")

        self.remove_dir(self.dict_sico_out_folders[ad_key])
        self.remove_dir(self.dict_ad_out_nc_files[ad_key])

        with open(self.dict_ad_log_files[ad_key], "w") as log:
            process = subprocess.run(
                        f"time {self.dict_ad_exec_cmds[ad_key]}",
                        cwd=self.src_dir,
                        shell=True,
                        stdout=log,
                        stderr=subprocess.STDOUT)

        ds_out_fields = self.open_xr_ds(self.dict_ad_out_nc_files[ad_key])

        # Returns ds_out_fields where even scalars are expressed as fields since we are reading simulation output
        return ds_out_fields

    @beartype
    def get_vx_vy_s(self, sico_out_nc_file: str) -> Tuple[Float[np.ndarray, "dimy dimx"], Float[np.ndarray, "dimy dimx"]]:

        _ = self.run_exec(ad_key = "nodiff")

        path_sico_out_nc = self.dict_sico_out_folders["nodiff"] + "/" + sico_out_nc_file
        if not os.path.isfile(path_sico_out_nc):
            raise ValueError(f"get_vx_vy_s: AD input file {self.dict_sico_out_folders['nodiff']}/{sico_out_nc_file} is missing.")

        ds_out_fields = self.open_xr_ds(path_sico_out_nc, False)

        if "vx_s_g" not in ds_out_fields or "vy_s_g" not in ds_out_fields:
            raise ValueError("get_vx_vy_s: One or both of vx_s_g or vy_s_g missing.")
    
        vx_s_g = ds_out_fields["vx_s_g"].data
        vy_s_g = ds_out_fields["vy_s_g"].data
    
        return vx_s_g, vy_s_g

    @beartype
    def eval_costs(self) -> Any:

        ds_out_fields_nodiff = self.run_exec(ad_key = "nodiff")

        return self.subset_of_ds(ds_out_fields_nodiff, attr_key = "type", attr_value = "cost")

    @beartype
    def eval_params(self) -> Any:

        ds_out_fields_nodiff = self.open_xr_ds(self.dict_ad_out_nc_files["nodiff"])

        ds_subset_params = self.subset_of_ds(ds_out_fields_nodiff, attr_key = "type", attr_value = "nodiff")

        for var in self.dict_params_fields_or_scalars:
    
            if self.dict_params_fields_or_scalars and self.dict_params_fields_or_scalars[var] == "scalar":
        
                if var in ds_subset_params:
                    ds_subset_params[var] = xr.DataArray([ds_subset_params[var].data.flat[0]], 
                                                          dims=["scalar"], attrs=ds_subset_params[var].attrs)
                else:
                    raise ValueError(f"eval_params: {var} not present in ds_subset_params.")

        return ds_subset_params

    @beartype
    def write_params(self, ds_subset_params: Any) -> None:

        dict_params_fields_vals = {}
        for var in ds_subset_params:
            if "type" not in ds_subset_params[var].attrs:
                raise ValueError(f"write_params: Attribute 'type' is missing for variable {var} in ds_subset_params.")
            elif ds_subset_params[var].attrs["type"] != "nodiff":
                raise ValueError(f"write_params: Type of {var} is not what is expected i.e. 'nodiff'.")
            elif self.dict_params_fields_or_scalars and self.dict_params_fields_or_scalars[var] == "scalar":
                dict_params_fields_vals[var] = ds_subset_params[var].data.flat[0].copy()
            else:
                dict_params_fields_vals[var] = ds_subset_params[var].data.copy()

        _ = self.create_ad_nodiff_or_adj_input_nc(dict_fields_vals = dict_params_fields_vals,
                                                  dict_fields_num_dims = self.dict_params_fields_num_dims,
                                                  dict_coords = self.dict_params_coords,
                                                  dict_attrs_type = self.dict_params_attrs_type,
                                                  dict_fields_or_scalars = self.dict_params_fields_or_scalars,
                                                  ad_key = "nodiff", filename = None)
 
        return None

    @beartype
    def eval_gradient(self) -> Any:

        ds_fields_gradient = self.run_exec(ad_key = "adj")
        ds_subset_gradient = self.subset_of_ds(ds_fields_gradient, attr_key = "type", attr_value = "adj")
        if self.list_fields_to_ignore is not None:
            ds_subset_gradient = ds_subset_gradient.drop_vars(field + "b" for field in self.list_fields_to_ignore)

        for varb in ds_subset_gradient:
    
            if self.dict_params_fields_or_scalars and self.dict_params_fields_or_scalars[varb[:-1]] == "scalar":
                fieldb_sum = np.sum(ds_subset_gradient[varb].data)
                ds_subset_gradient[varb] = xr.DataArray([fieldb_sum], dims=["scalar"], attrs=ds_subset_gradient[varb].attrs)

        return ds_subset_gradient

    @beartype
    def line_search(self,
                    ds_subset_gradient: Any, 
                    ds_subset_descent_dir: Any,
                    init_alpha: float = 1.0,
                    min_alpha_tol: float = 1.e-10,
                    c1: float = 1.e-4) -> Tuple[float, Any]:
    
        alpha = init_alpha
        ds_subset_params_orig = self.ds_subset_params.copy()
        ds_subset_costs_orig = self.ds_subset_costs.copy()

        while True:
            
            try:                
                ds_subset_params_new = self.linear_sum([ds_subset_params_orig, ds_subset_descent_dir], 
                                                       [1.0, alpha], ["nodiff", "adj"])
                self.write_params(ds_subset_params_new)
        
                ds_subset_costs_new = self.eval_costs()
                pTg = self.l2_inner_product([ds_subset_descent_dir, ds_subset_gradient], ["adj", "adj"])
                ratio = (ds_subset_costs_new["fc"].data[0] - ds_subset_costs_orig["fc"].data[0])/(alpha*pTg)

            except:
                print("Too big step size probably crashed the simulation.")

                ## Can be used for diagnosing issues
                # self.write_params(ds_subset_params_new)
                # break

                self.write_params(ds_subset_params_orig)
                ratio = 0.0

            if alpha <= min_alpha_tol:
                print(f"Minimum tolerable step size alpha reached.")
                print(f"Step size alpha = {alpha}")
                return alpha, ds_subset_costs_new.copy()

            if ratio >= c1:
                print(f"Step size alpha = {alpha}")
                return alpha, ds_subset_costs_new.copy()
    
            alpha = alpha/2.0

    @beartype
    def gradient_descent(self, 
                         MAX_ITERS: int, 
                         MIN_GRAD_NORM_TOL: Optional[float] = None, 
                         init_alpha: float = 1.0, 
                         min_alpha_tol: float = 1.e-10,
                         c1: float = 1.e-4) -> Any:

        if self.dirpath_store_states is not None:

            if not os.path.isdir(self.dirpath_store_states):
                self.make_dir(self.dirpath_store_states)
            if os.path.isdir(self.dirpath_store_states + "/" + "gradient_descent"):
                self.remove_dir(self.dirpath_store_states + "/" + "gradient_descent")

            self.make_dir(self.dirpath_store_states + "/" + "gradient_descent")

            log_file = self.dirpath_store_states + "/gradient_descent/" + "gradient_descent.log"
            with open(log_file, "a") as f:
                f.write(f"Iteration 0: Cost = {self.ds_subset_costs['fc'].data[0]:.6f}, Misfit Cost = {self.ds_subset_costs['fc_data'].data[0]:.6f}, Regularization Cost = {self.ds_subset_costs['fc_reg'].data[0]:.6f}\n")

            self.ds_subset_params.to_netcdf(self.dirpath_store_states + "/gradient_descent/" + f"state_GD_iter_0.nc")
            self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dirpath_store_states + "/gradient_descent/" + "state_GD_iter_0_fields.nc")

            path_sico_out_nc = self.dict_sico_out_folders["nodiff"] + "/" + self.filename_final_sim_output
            if not os.path.isfile(path_sico_out_nc):
                raise ValueError(f"gradient_descent: Final simulation output file {path_sico_out_nc} is missing.")
            self.copy_dir(path_sico_out_nc, self.dirpath_store_states + "/gradient_descent/" + f"final_sim_output_GD_iter_0_fields.nc")

        print("---------------------------------------------------------------------------------------------------------------")
        print(f"iter 0, fc = {self.ds_subset_costs['fc'].data[0]}, fc_data = {self.ds_subset_costs['fc_data'].data[0]}, fc_reg = {self.ds_subset_costs['fc_reg'].data[0]}")
        print("---------------------------------------------------------------------------------------------------------------")

        for i in range(MAX_ITERS):

            ds_subset_params = self.ds_subset_params.copy()
            ds_subset_gradient = self.eval_gradient()
            norm_gradient = self.l2_inner_product([ds_subset_gradient, ds_subset_gradient], ["adj", "adj"])**0.5
            
            if MIN_GRAD_NORM_TOL is not None and norm_gradient <= MIN_GRAD_NORM_TOL:
                print("Minimum for gradient norm reached.")
                break

            ds_subset_descent_dir = self.linear_sum([ds_subset_gradient, ds_subset_gradient], 
                                                    [0.0, -1.0], ["adj", "adj"])

            alpha, self.ds_subset_costs = self.line_search(ds_subset_gradient,
                                                           ds_subset_descent_dir,
                                                           init_alpha, min_alpha_tol, c1)

            ds_subset_params_new = self.linear_sum([ds_subset_params, ds_subset_gradient], 
                                                   [1.0, -alpha], ["nodiff", "adj"])
            self.write_params(ds_subset_params_new)
            self.ds_subset_params = ds_subset_params_new.copy()

            self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dict_ad_inp_nc_files["adj"])

            if self.dirpath_store_states is not None:

                with open(log_file, "a") as f:
                    f.write(f"Iteration {i+1}: Cost = {self.ds_subset_costs['fc'].data[0]:.6f}, Misfit Cost = {self.ds_subset_costs['fc_data'].data[0]:.6f}, Regularization Cost = {self.ds_subset_costs['fc_reg'].data[0]:.6f}\n")

                self.ds_subset_params.to_netcdf(self.dirpath_store_states + "/gradient_descent/" + f"state_GD_iter_{i+1}.nc")
                ds_subset_gradient.to_netcdf(self.dirpath_store_states + "/gradient_descent/" + f"gradient_GD_iter_{i}.nc")
                self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dirpath_store_states + "/gradient_descent/" + f"state_GD_iter_{i+1}_fields.nc")

                path_sico_out_nc = self.dict_sico_out_folders["nodiff"] + "/" + self.filename_final_sim_output
                if not os.path.isfile(path_sico_out_nc):
                    raise ValueError(f"gradient_descent: Final simulation output file {path_sico_out_nc} is missing.")
                self.copy_dir(path_sico_out_nc, self.dirpath_store_states + "/gradient_descent/" + f"final_sim_output_GD_iter_{i+1}_fields.nc")

            print("---------------------------------------------------------------------------------------------------------------")
            print(f"iter {i+1}, fc = {self.ds_subset_costs['fc'].data[0]}, fc_data = {self.ds_subset_costs['fc_data'].data[0]}, fc_reg = {self.ds_subset_costs['fc_reg'].data[0]}")
            print("---------------------------------------------------------------------------------------------------------------")

        return self.ds_subset_params

    @beartype
    def create_ad_tlm_action_input_nc(self,
                                      dict_tlm_action_only_fields_vals: Optional[Dict[str, Union[Float[np.ndarray, "dimz dimy dimx"],
                                                                                                 Float[np.ndarray, "dimy dimx"],
                                                                                                 float]]] = None,
                                      bool_randomize: bool = False) -> Any:

        if dict_tlm_action_only_fields_vals is not None and bool_randomize:
            raise ValueError("create_ad_tlm_action_input_nc: Either specify the values dict for tlm action or let the code create a randomized vector.")
        elif dict_tlm_action_only_fields_vals is None and not bool_randomize:
            raise ValueError("create_ad_tlm_action_input_nc: Specify the values dict for tlm action or let the code create a randomized vector.")

        ds_subset_params = self.ds_subset_params.copy()
        dict_params_only_fields_vals = {}

        for var in ds_subset_params:

            if self.dict_params_fields_or_scalars and self.dict_params_fields_or_scalars[var] == "scalar":
                dict_params_only_fields_vals[var] = ds_subset_params[var].data[0]
            else:
                dict_params_only_fields_vals[var] = ds_subset_params[var].data.copy()

        if dict_tlm_action_only_fields_vals is not None:

            if self.list_fields_to_ignore:
                for key in self.list_fields_to_ignore:
                    if key + "d" in dict_tlm_action_only_fields_vals.keys():
                        raise ValueError(f"create_ad_tlm_action_input_nc: Don't add ignored field {key} in the dict_tlm_action_only_fields_vals.")
                    else:
                        if isinstance(dict_params_only_fields_vals[key], np.ndarray) and not isinstance(dict_params_only_fields_vals[key], (str, bytes)):
                            dict_tlm_action_only_fields_vals[key + "d"] = np.zeros(dict_params_only_fields_vals[key].shape, dtype=float)
                        elif isinstance(dict_params_only_fields_vals[key], float):
                            dict_tlm_action_only_fields_vals[key + "d"] = 0.0
                        else:
                            raise ValueError(f"create_ad_tlm_action_input_nc: Impossible condition.")

        elif bool_randomize:

            dict_tlm_action_only_fields_vals = {}
            for key, value in dict_params_only_fields_vals.items():
                if self.list_fields_to_ignore and key in self.list_fields_to_ignore and isinstance(value, np.ndarray) and not isinstance(value, (str, bytes)):
                    dict_tlm_action_only_fields_vals[key + "d"] = np.zeros(value.shape, dtype=float)
                elif self.list_fields_to_ignore and key in self.list_fields_to_ignore and isinstance(value, float):
                    dict_tlm_action_only_fields_vals[key + "d"] = 0.0
                elif isinstance(value, np.ndarray) and not isinstance(value, (str, bytes)):
                    dict_tlm_action_only_fields_vals[key + "d"] = np.random.randn(*value.shape)
                elif isinstance(value, float):
                    dict_tlm_action_only_fields_vals[key + "d"] = np.random.randn()
                else:
                    raise ValueError("create_ad_tlm_action_input_nc: Some error while creating the random vector for TLM action.")

        else:

            raise ValueError("create_ad_tlm_action_input_nc: Somehow you have entered a condition that's simply impossible.")

        ds_inp_fields_tlm_action = self.create_ad_nodiff_or_adj_input_nc(dict_params_only_fields_vals | dict_tlm_action_only_fields_vals,
                                                                         self.dict_tlm_action_fields_num_dims,
                                                                         self.dict_tlm_action_coords,
                                                                         self.dict_tlm_action_attrs_type,
                                                                         self.dict_tlm_action_fields_or_scalars,
                                                                         "tlm_action", None)

        ds_subset_tlm = self.subset_of_ds(ds_inp_fields_tlm_action, "type", "tlm")

        if self.list_fields_to_ignore:
            ds_subset_tlm = ds_subset_tlm.drop_vars(field + "d" for field in self.list_fields_to_ignore)

        for var in ds_subset_tlm:
            if self.dict_tlm_action_fields_or_scalars[var] == "scalar":
                ds_subset_tlm[var] = xr.DataArray([ds_subset_tlm[var].data.flat[0]], 
                                                   dims=["scalar"], attrs=ds_subset_tlm[var].attrs)

        return ds_subset_tlm

    @beartype
    def eval_tlm_action(self) -> Any:

        ds_out_fields_tlm_action = self.run_exec(ad_key = "tlm_action")
        ds_subset_fields_tlm_action = self.subset_of_ds(ds_out_fields_tlm_action, attr_key = "type", attr_value = "tlmhessaction")

        list_vars = [var[:-1] for var in ds_subset_fields_tlm_action]

        if set(self.dict_masks_observables.keys()) != set(list_vars):
            raise ValueError("eval_tlm_action: The observables seem to be different than expected.")

        # This is fine since none of the observables are scalars
        return ds_subset_fields_tlm_action

    @beartype
    def eval_noise_cov_inv_action(self, ds_subset_fields_tlm_action: Any) -> Any:

        ds_subset_fields_tlm_action = ds_subset_fields_tlm_action.copy()

        list_vars = [var[:-1] for var in ds_subset_fields_tlm_action]
        if set(self.dict_masks_observables.keys()) != set(list_vars):
            raise ValueError("eval_noise_cov_inv_action: The observables seem to be different than expected.")

        if not all(value is None for value in self.dict_masks_observables.values()):
            for key, value in self.dict_masks_observables.items():
                if value is not None:
                    if value.shape != ds_subset_fields_tlm_action[key + "d"].data.shape:
                        raise ValueError(f"The right shaped noise covariance mask has not been prescribed for {key}.")
                    ds_subset_fields_tlm_action[key + "d"].data = ds_subset_fields_tlm_action[key + "d"].data*value

        # This is fine since none of the observables are scalars
        return self.exch_tlm_adj_nc(ds_subset_fields_tlm_action, og_type = "tlmhessaction")

    @beartype
    def eval_adj_action(self) -> Any:

        ds_fields_adj_action = self.run_exec(ad_key = "adj_action")
        ds_subset_adj_action = self.subset_of_ds(ds_fields_adj_action, attr_key = "type", attr_value = "adj")
        if self.list_fields_to_ignore is not None:
            ds_subset_adj_action.drop_vars(field + "b" for field in self.list_fields_to_ignore)

        for varb in ds_subset_adj_action:

            if self.dict_params_fields_or_scalars and self.dict_params_fields_or_scalars[varb[:-1]] == "scalar":
                fieldb_sum = np.sum(ds_subset_adj_action[varb].data)
                ds_subset_adj_action[varb] = xr.DataArray([fieldb_sum], dims=["scalar"], attrs=ds_subset_adj_action[varb].attrs)

        return ds_subset_adj_action

    @beartype
    def eval_misfit_hessian_action(self) -> Any:

        ds_subset_tlm_action = self.eval_tlm_action()
        _ = self.eval_noise_cov_inv_action(ds_subset_tlm_action)

        ds_inp_fields_adj_action = self.open_xr_ds(self.dict_ad_inp_nc_files["adj_action"], False)

        if self.bool_vs_cost:

            vx_s_g_final, vy_s_g_final = self.get_vx_vy_s(self.filename_vx_vy_s_g)

            da_vx_s_g_final = xr.DataArray(
                        data=vx_s_g_final,
                        dims=["y", "x"],
                        coords={"y": self.dict_params_coords["y"].copy(),
                                "x": self.dict_params_coords["x"].copy()
                                },
                        name="vx_s_g_final"
                    )

            da_vy_s_g_final = xr.DataArray(
                        data=vy_s_g_final,
                        dims=["y", "x"],
                        coords={"y": self.dict_params_coords["y"].copy(),
                                "x": self.dict_params_coords["x"].copy()
                                },
                        name="vy_s_g_final"
                    )

            ds_inp_fields_adj_action["vx_s_g_final"] = da_vx_s_g_final
            ds_inp_fields_adj_action["vy_s_g_final"] = da_vy_s_g_final

        # Some weird permission denied error if this file is not removed first.
        self.remove_dir(self.dict_ad_inp_nc_files["adj_action"])
        ds_inp_fields_adj_action.to_netcdf(self.dict_ad_inp_nc_files["adj_action"])
            
        return self.eval_adj_action()

    @beartype
    def eval_sqrt_prior_C_inv_action(self) -> Any:

        ds_inp_fields_tlm = self.open_xr_ds(self.dict_ad_inp_nc_files["tlm_action"], False)

        ds_subset_fields_tlm = self.subset_of_ds(ds_inp_fields_tlm, "type", "tlm")

        for var in ds_subset_fields_tlm:

            basic_str = var[:-1]

            if self.dict_params_fields_or_scalars[basic_str] == "scalar":

                delta_x = self.delta_x
                delta_y = self.delta_y
                # NOTE: np.sqrt(delta_x * delta_y) should be changed to np.sqrt(delta_z) for 3D scalars, this is future work since this is good enough for now.
                ds_subset_fields_tlm[var].data = ds_subset_fields_tlm[var].data * self.dict_prior_deltas[basic_str] * np.sqrt(delta_x * delta_y)

            elif self.dict_params_fields_or_scalars[basic_str] == "field" and self.dict_params_fields_num_dims[basic_str] == "2D":

                gamma = self.dict_prior_gammas[basic_str]
                delta = self.dict_prior_deltas[basic_str]
                delta_x = self.delta_x
                delta_y = self.delta_y

                if gamma != 0.0:

                    field = ds_subset_fields_tlm[var].data.copy()
                    field_new = delta*field.copy()

                    IMAX = self.IMAX
                    JMAX = self.JMAX

                    field_new[0, 0] = field_new[0, 0] - gamma*((field[0, 1]-field[0, 0])/delta_x**2 + (field[1, 0]-field[0,0])/delta_y**2)
                    field_new[JMAX, 0] = field_new[JMAX, 0] - gamma*((field[JMAX, 1]-field[JMAX, 0])/delta_x**2 + (field[JMAX-1, 0]-field[JMAX, 0])/delta_y**2)
                    field_new[0, IMAX] = field_new[0, IMAX] - gamma*((field[0, IMAX-1]-field[0, IMAX])/delta_x**2 + (field[1, IMAX]-field[0, IMAX])/delta_y**2)
                    field_new[JMAX, IMAX] = field_new[JMAX, IMAX] - gamma*((field[JMAX, IMAX-1]-field[JMAX, IMAX])/delta_x**2 + (field[JMAX-1, IMAX]-field[JMAX, IMAX])/delta_y**2)

                    field_new[1:JMAX, 0] = field_new[1:JMAX, 0] - gamma*((field[0:JMAX-1, 0] - 2*field[1:JMAX, 0] + field[2:, 0])/delta_y**2 + (field[1:JMAX, 1] - field[1:JMAX, 0])/delta_x**2)
                    field_new[1:JMAX, IMAX] = field_new[1:JMAX, IMAX] - gamma*((field[0:JMAX-1, IMAX] - 2*field[1:JMAX, IMAX] + field[2:, IMAX]) / delta_y**2 + (field[1:JMAX, IMAX-1] - field[1:JMAX, IMAX]) / delta_x**2)

                    field_new[0, 1:IMAX] = field_new[0, 1:IMAX] - gamma*((field[1, 1:IMAX] - field[0, 1:IMAX])/delta_y**2 + (field[0, 0:IMAX-1] - 2*field[0, 1:IMAX] + field[0, 2:])/delta_x**2)
                    field_new[JMAX, 1:IMAX] = field_new[JMAX, 1:IMAX] - gamma*((field[JMAX-1, 1:IMAX] - field[JMAX, 1:IMAX]) / delta_y**2 + (field[JMAX, 0:IMAX-1] - 2*field[JMAX, 1:IMAX] + field[JMAX, 2:]) / delta_x**2)

                    for j in range(1, JMAX):
                        for i in range(1, IMAX):
                            field_new[j, i] = field_new[j, i] - gamma*(field[j, i-1] - 2*field[j, i] + field[j, i+1]) / delta_x**2
                            field_new[j, i] = field_new[j, i] - gamma*(field[j-1, i] - 2*field[j, i] + field[j+1, i]) / delta_y**2

                    ds_subset_fields_tlm[var].data = field_new.copy() * np.sqrt(delta_x * delta_y)

                else:

                    ds_subset_fields_tlm[var].data = delta*ds_subset_fields_tlm[var].data.copy() * np.sqrt(delta_x * delta_y)

            elif self.dict_params_fields_or_scalars[basic_str] == "field" and self.dict_params_fields_num_dims[basic_str] == "3D":

                gamma = self.dict_prior_gammas[basic_str]
                delta = self.dict_prior_deltas[basic_str]
                delta_z = 1.e6*(self.dict_params_coords["zeta_c"][1:]-self.dict_params_coords["zeta_c"][:-1])
                delta_z_numerator_field = np.zeros((delta_z.shape[0] + 1,), dtype = float)
                delta_z_numerator_field[0] = delta_z[0]
                delta_z_numerator_field[-1] = delta_z[-1]
                delta_z_numerator_field[1:-1] = (delta_z[:-1] + delta_z[1:]) / 2.0

                if gamma != 0.0:

                    field = ds_subset_fields_tlm[var].data.copy()
                    field_new = delta*field.copy()

                    KCMAX = self.KCMAX

                    field_new[0] = field_new[0] - gamma*(field[1]-field[0])/delta_z[0]**2
                    field_new[KCMAX] = field_new[KCMAX] - gamma*(field[KCMAX-1]-field[KCMAX])/delta_z[KCMAX-1]**2

                    for kc in range(1, KCMAX):
                        field_new[kc] = field_new[kc] - gamma*((field[kc+1] - field[kc])/delta_z[kc] - (field[kc] - field[kc-1])/delta_z[kc-1])*(2.0/(delta_z[kc]+delta_z[kc-1]))

                    ds_subset_fields_tlm[var].data = field_new.copy() * np.sqrt(delta_z_numerator_field[:, None, None])

                else:

                    ds_subset_fields_tlm[var].data = delta*ds_subset_fields_tlm[var].data.copy() * np.sqrt(delta_z_numerator_field[:, None, None])

            elif self.dict_params_fields_or_scalars[basic_str] == "field" and self.dict_params_fields_num_dims[basic_str] == "3DR":

                gamma = self.dict_prior_gammas[basic_str]
                delta = self.dict_prior_deltas[basic_str]
                delta_z = 1.e6*(self.dict_params_coords["zeta_r"][1:]-self.dict_params_coords["zeta_r"][:-1])
                delta_z_numerator_field = np.zeros((delta_z.shape[0] + 1,), dtype = float)
                delta_z_numerator_field[0] = delta_z[0]
                delta_z_numerator_field[-1] = delta_z[-1]
                delta_z_numerator_field[1:-1] = (delta_z[:-1] + delta_z[1:]) / 2.0

                if gamma != 0.0:

                    field = ds_subset_fields_tlm[var].data.copy()
                    field_new = delta*field.copy()

                    KRMAX = self.KRMAX

                    field_new[0] = field_new[0] - gamma*(field[1]-field[0])/delta_z[0]**2
                    field_new[KRMAX] = field_new[KRMAX] - gamma*(field[KRMAX-1]-field[KRMAX])/delta_z[KRMAX-1]**2

                    for kr in range(1, KRMAX):
                        field_new[kr] = field_new[kr] - gamma*((field[kr+1] - field[kr])/delta_z[kr] - (field[kr] - field[kr-1])/delta_z[kr-1])*(2.0/(delta_z[kr]+delta_z[kr-1]))

                    ds_subset_fields_tlm[var].data = field_new.copy() * np.sqrt(delta_z_numerator_field[:, None, None])

                else:

                    ds_subset_fields_tlm[var].data = delta*ds_subset_fields_tlm[var].data.copy() * np.sqrt(delta_z_numerator_field[:, None, None])

            elif self.dict_params_fields_or_scalars[basic_str] == "field" and self.dict_params_fields_num_dims[basic_str] == "2DT":

                gamma = self.dict_prior_gammas[basic_str]
                delta = self.dict_prior_deltas[basic_str]
                delta_x = self.delta_x
                delta_y = self.delta_y

                if gamma != 0.0:

                    field = ds_subset_fields_tlm[var].data.copy()
                    field_new = delta*field.copy()

                    IMAX = self.IMAX
                    JMAX = self.JMAX
                    NTDAMAX = self.NTDAMAX

                    field_new[:, 0, 0] = field_new[:, 0, 0] - gamma*((field[:, 0, 1]-field[:, 0, 0])/delta_x**2 + (field[:, 1, 0]-field[:, 0, 0])/delta_y**2)
                    field_new[:, JMAX, 0] = field_new[:, JMAX, 0] - gamma*((field[:, JMAX, 1]-field[:, JMAX, 0])/delta_x**2 + (field[:, JMAX-1, 0]-field[:, JMAX, 0])/delta_y**2)
                    field_new[:, 0, IMAX] = field_new[:, 0, IMAX] - gamma*((field[:, 0, IMAX-1]-field[:, 0, IMAX])/delta_x**2 + (field[:, 1, IMAX]-field[:, 0, IMAX])/delta_y**2)
                    field_new[:, JMAX, IMAX] = field_new[:, JMAX, IMAX] - gamma*((field[:, JMAX, IMAX-1]-field[:, JMAX, IMAX])/delta_x**2 + (field[:, JMAX-1, IMAX]-field[:, JMAX, IMAX])/delta_y**2)

                    field_new[:, 1:JMAX, 0] = field_new[:, 1:JMAX, 0] - gamma*((field[:, 0:JMAX-1, 0] - 2*field[:, 1:JMAX, 0] + field[:, 2:, 0])/delta_y**2 + (field[:, 1:JMAX, 1] - field[:, 1:JMAX, 0])/delta_x**2)
                    field_new[:, 1:JMAX, IMAX] = field_new[:, 1:JMAX, IMAX] - gamma*((field[:, 0:JMAX-1, IMAX] - 2*field[:, 1:JMAX, IMAX] + field[:, 2:, IMAX]) / delta_y**2 + (field[:, 1:JMAX, IMAX-1] - field[:, 1:JMAX, IMAX]) / delta_x**2)

                    field_new[:, 0, 1:IMAX] = field_new[:, 0, 1:IMAX] - gamma*((field[:, 1, 1:IMAX] - field[:, 0, 1:IMAX])/delta_y**2 + (field[:, 0, 0:IMAX-1] - 2*field[:, 0, 1:IMAX] + field[:, 0, 2:])/delta_x**2)
                    field_new[:, JMAX, 1:IMAX] = field_new[:, JMAX, 1:IMAX] - gamma*((field[:, JMAX-1, 1:IMAX] - field[:, JMAX, 1:IMAX]) / delta_y**2 + (field[:, JMAX, 0:IMAX-1] - 2*field[:, JMAX, 1:IMAX] + field[:, JMAX, 2:]) / delta_x**2)

                    for j in range(1, JMAX):
                        for i in range(1, IMAX):
                            field_new[:, j, i] = field_new[:, j, i] - gamma*(field[:, j, i-1] - 2*field[:, j, i] + field[:, j, i+1]) / delta_x**2
                            field_new[:, j, i] = field_new[:, j, i] - gamma*(field[:, j-1, i] - 2*field[:, j, i] + field[:, j+1, i]) / delta_y**2

                    ds_subset_fields_tlm[var].data = field_new.copy() * np.sqrt(delta_x * delta_y)

                else:

                    ds_subset_fields_tlm[var].data = delta*ds_subset_fields_tlm[var].data.copy() * np.sqrt(delta_x * delta_y)

            else:
                raise ValueError(f"eval_sqrt_prior_C_inv_action: Issue with {var}. Prior action only works for scalar or 2D or 3D or 3DR or 2DT fields.")

        dict_tlm_action_only_fields_vals = {}
        for var in ds_subset_fields_tlm:

            if not self.list_fields_to_ignore or (self.list_fields_to_ignore and var[:-1] not in self.list_fields_to_ignore):
                if self.dict_tlm_action_fields_or_scalars[var] == "scalar":
                    dict_tlm_action_only_fields_vals[var] = ds_subset_fields_tlm[var].data.flat[0].copy()
                else:
                    dict_tlm_action_only_fields_vals[var] = ds_subset_fields_tlm[var].data.copy()

        return self.create_ad_tlm_action_input_nc(dict_tlm_action_only_fields_vals)

    @beartype
    def eval_sqrt_prior_cov_inv_action(self) -> Any:

        ds_subset_x = self.open_xr_ds(self.dict_ad_inp_nc_files["tlm_action"], False)
        ds_subset_x = self.subset_of_ds(ds_subset_x, "type", "tlm")

        dict_tlm_action_only_fields_vals = {}
        for var in ds_subset_x:

            basic_str = var[:-1]
            if not self.list_fields_to_ignore or (self.list_fields_to_ignore and var[:-1] not in self.list_fields_to_ignore):

                if self.dict_tlm_action_fields_or_scalars[var] == "scalar":
                    dict_tlm_action_only_fields_vals[var] = ds_subset_x[var].data.flat[0].copy() / (self.dict_prior_sigmas[basic_str] * self.ds_prior_X[var].data.flat[0])
                else:
                    dict_tlm_action_only_fields_vals[var] = ds_subset_x[var].data.copy() / (self.dict_prior_sigmas[basic_str] * self.ds_prior_X[var].data)

        _ =  self.create_ad_tlm_action_input_nc(dict_tlm_action_only_fields_vals)

        return self.eval_sqrt_prior_C_inv_action()

    @beartype
    def eval_sqrt_prior_C_action(self,
                                 ad_key_adj_or_adj_action_or_tlm_action: str) -> Any:

        if ad_key_adj_or_adj_action_or_tlm_action not in ["tlm_action", "adj", "adj_action"]:
            raise ValueError("eval_sqrt_prior_C_action: Can only act on tlm or adj or adj_action quantities.")

        if ad_key_adj_or_adj_action_or_tlm_action == "tlm_action":
            ds_fields_adj_or_adj_action_or_tlm_action = self.open_xr_ds(self.dict_ad_inp_nc_files[ad_key_adj_or_adj_action_or_tlm_action], False)
            ad_subset_key = "tlm"
        else:
            # NOTE: eval_sqrt_prior_C_action called within eval_sqrt_prior_cov_action("adj") called in inexact_gn_hessian_cg is called on ds_out written to self.dict_ad_out_nc_files["adj"] and it has scalars as scalars and not fields.
            #       ds_fields_adj_or_adj_action_or_tlm_action as a name implies fields but that is not true for this call of eval_sqrt_prior_C_action. It seems to be working fine. See comments in inexact_gn_hessian_cg to understand more.
            ds_fields_adj_or_adj_action_or_tlm_action = self.open_xr_ds(self.dict_ad_out_nc_files[ad_key_adj_or_adj_action_or_tlm_action])
            ad_subset_key = "adj"

            ## Keeping this print statement here which can help you see what many (not all) of the comments starting with NOTE: are trying to explain. Print any scalar variable's shape here if xx_n_glen_da_dummy2d_scalar is not in use.
            # print(ds_fields_adj_or_adj_action_or_tlm_action["xx_n_glen_da_dummy2d_scalar"].shape)

        ds_subset_fields_params = self.subset_of_ds(ds_fields_adj_or_adj_action_or_tlm_action, "type", "nodiff")
        ds_subset_fields_adj_or_adj_action_or_tlm_action = self.subset_of_ds(ds_fields_adj_or_adj_action_or_tlm_action, "type", ad_subset_key)

        for var in ds_subset_fields_adj_or_adj_action_or_tlm_action:

            basic_str = var[:-1]

            if self.dict_params_fields_or_scalars[basic_str] == "scalar" and (not self.list_fields_to_ignore or (self.list_fields_to_ignore and basic_str not in self.list_fields_to_ignore)):

                delta_x = self.delta_x
                delta_y = self.delta_y
                # NOTE: np.sqrt(delta_x * delta_y) should be changed to np.sqrt(delta_z) for 3D scalars, this is future work since this is good enough for now.
                ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data = ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data / (self.dict_prior_deltas[basic_str] * np.sqrt(delta_x * delta_y))

            elif self.dict_params_fields_or_scalars[basic_str] == "field" and self.dict_params_fields_num_dims[basic_str] == "2D" and (not self.list_fields_to_ignore or (self.list_fields_to_ignore and basic_str not in self.list_fields_to_ignore)):

                gamma = self.dict_prior_gammas[basic_str]
                delta = self.dict_prior_deltas[basic_str]
                delta_x = self.delta_x
                delta_y = self.delta_y

                if gamma != 0.0:

                    IMAX = self.IMAX
                    JMAX = self.JMAX

                    field = ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data.copy() / np.sqrt(delta_x * delta_y)

                    result_old = np.copy(field)
                    result = np.copy(field)

                    for _ in range(self.MAX_ITERS_SOR):

                        for j in range(JMAX+1):
                            for i in range(IMAX+1):

                                if j == 0 and i == 0:
                                    diagonal = delta + gamma*(1/delta_x**2 + 1/delta_y**2)
                                    bracket = field[0, 0] + gamma*(result_old[0, 1] / delta_x**2 + result_old[1, 0] / delta_y**2)
                                elif j == JMAX and i == 0:
                                    diagonal = delta + gamma*(1/delta_x**2 + 1/delta_y**2)
                                    bracket = field[JMAX, 0] + gamma*(result_old[JMAX, 1] / delta_x**2 + result[JMAX-1, 0] / delta_y**2)
                                elif j == 0 and i == IMAX:
                                    diagonal = delta + gamma*(1/delta_x**2 + 1/delta_y**2)
                                    bracket = field[0, IMAX] + gamma*(result[0, IMAX-1] / delta_x**2 + result_old[1, IMAX] / delta_y**2)
                                elif j == JMAX and i == IMAX:
                                    diagonal = delta + gamma*(1/delta_x**2 + 1/delta_y**2)
                                    bracket = field[JMAX, IMAX] + gamma*(result[JMAX, IMAX-1] / delta_x**2 + result[JMAX-1, IMAX] / delta_y**2)
                                elif i == 0:
                                    diagonal = delta + gamma*(1/delta_x**2 + 2/delta_y**2)
                                    bracket = field[j, 0] + gamma*((result[j-1, 0] + result_old[j+1, 0]) / delta_y**2 + result_old[j, 1] / delta_x**2)
                                elif i == IMAX:
                                    diagonal = delta + gamma*(1/delta_x**2 + 2/delta_y**2)
                                    bracket = field[j, IMAX] + gamma*((result[j-1, IMAX] + result_old[j+1, IMAX]) / delta_y**2 + result[j, IMAX-1] / delta_x**2)
                                elif j == 0:
                                    diagonal = delta + gamma*(2/delta_x**2 + 1/delta_y**2)
                                    bracket = field[0, i] + gamma*(result_old[1, i] / delta_y**2 + (result[0, i-1] + result_old[0, i+1]) / delta_x**2)
                                elif j == JMAX:
                                    diagonal = delta + gamma*(1/delta_x**2 + 2/delta_y**2)
                                    bracket = field[JMAX, i] + gamma*(result[JMAX-1, i] / delta_y**2 + (result[JMAX, i-1] + result_old[JMAX, i+1]) / delta_x**2)
                                else:
                                    diagonal = delta + 2*gamma*(1/delta_x**2 + 1/delta_y**2)
                                    bracket = field[j, i] + gamma*((result[j-1, i] + result_old[j+1, i]) / delta_y**2 + (result[j, i-1] + result_old[j, i+1]) / delta_x**2)

                                result[j, i] = (1 - self.OMEGA_SOR) * result_old[j, i] + self.OMEGA_SOR / diagonal * bracket

                        result_old = result.copy()

                    ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data = result.copy()

                else:

                    ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data = ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data.copy() / (delta * np.sqrt(delta_x * delta_y))

            elif self.dict_params_fields_or_scalars[basic_str] == "field" and self.dict_params_fields_num_dims[basic_str] == "3D" and (not self.list_fields_to_ignore or (self.list_fields_to_ignore and basic_str not in self.list_fields_to_ignore)):

                gamma = self.dict_prior_gammas[basic_str]
                delta = self.dict_prior_deltas[basic_str]
                delta_z = 1.e6*(self.dict_params_coords["zeta_c"][1:]-self.dict_params_coords["zeta_c"][:-1])
                delta_z_denominator_field = np.zeros((delta_z.shape[0] + 1,), dtype = float)
                delta_z_denominator_field[0] = delta_z[0]
                delta_z_denominator_field[-1] = delta_z[-1]
                delta_z_denominator_field[1:-1] = (delta_z[:-1] + delta_z[1:]) / 2.0

                if gamma != 0.0:

                    KCMAX = self.KCMAX

                    field = ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data.copy() / np.sqrt(delta_z_denominator_field[:, None, None])

                    result_old = np.copy(field)
                    result = np.copy(field)

                    for _ in range(self.MAX_ITERS_SOR):

                        for kc in range(KCMAX+1):

                            if kc == 0:
                                diagonal = delta + gamma / delta_z[0]**2
                                bracket = field[0] + gamma * result_old[1] / delta_z[0]**2
                            elif kc == KCMAX:
                                diagonal = delta + gamma / delta_z[KCMAX-1]**2
                                bracket = field[KCMAX] + gamma * result[KCMAX-1] / delta_z[KCMAX-1]**2
                            else:
                                diagonal = delta + 2 * gamma / (delta_z[kc]*delta_z[kc-1])
                                bracket = field[kc] + gamma*(result[kc-1] / delta_z[kc-1] + result_old[kc+1] / delta_z[kc])*(2.0/(delta_z[kc]+delta_z[kc-1]))

                            result[kc] = (1 - self.OMEGA_SOR) * result_old[kc] + self.OMEGA_SOR / diagonal * bracket

                        result_old = result.copy()

                    ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data = result.copy()

                else:

                    ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data = ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data.copy() / (delta * np.sqrt(delta_z_denominator_field[:, None, None]))

            elif self.dict_params_fields_or_scalars[basic_str] == "field" and self.dict_params_fields_num_dims[basic_str] == "3DR" and (not self.list_fields_to_ignore or (self.list_fields_to_ignore and basic_str not in self.list_fields_to_ignore)):

                gamma = self.dict_prior_gammas[basic_str]
                delta = self.dict_prior_deltas[basic_str]
                delta_z = 1.e6*(self.dict_params_coords["zeta_r"][1:]-self.dict_params_coords["zeta_r"][:-1])
                delta_z_denominator_field = np.zeros((delta_z.shape[0] + 1,), dtype = float)
                delta_z_denominator_field[0] = delta_z[0]
                delta_z_denominator_field[-1] = delta_z[-1]
                delta_z_denominator_field[1:-1] = (delta_z[:-1] + delta_z[1:]) / 2.0

                if gamma != 0.0:

                    KRMAX = self.KRMAX

                    field = ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data.copy() / np.sqrt(delta_z_denominator_field[:, None, None])

                    result_old = np.copy(field)
                    result = np.copy(field)

                    for _ in range(self.MAX_ITERS_SOR):

                        for kr in range(KRMAX+1):

                            if kr == 0:
                                diagonal = delta + gamma / delta_z[0]**2
                                bracket = field[0] + gamma * result_old[1] / delta_z[0]**2
                            elif kr == KRMAX:
                                diagonal = delta + gamma / delta_z[KRMAX-1]**2
                                bracket = field[KRMAX] + gamma * result[KRMAX-1] / delta_z[KRMAX-1]**2
                            else:
                                diagonal = delta + 2 * gamma / (delta_z[kr]*delta_z[kr-1])
                                bracket = field[kr] + gamma*(result[kr-1] / delta_z[kr-1] + result_old[kr+1] / delta_z[kr])*(2.0/(delta_z[kr]+delta_z[kr-1]))

                            result[kr] = (1 - self.OMEGA_SOR) * result_old[kr] + self.OMEGA_SOR / diagonal * bracket

                        result_old = result.copy()

                    ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data = result.copy()

                else:

                    ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data = ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data.copy() / (delta * np.sqrt(delta_z_denominator_field[:, None, None]))

            elif self.dict_params_fields_or_scalars[basic_str] == "field" and self.dict_params_fields_num_dims[basic_str] == "2DT" and (not self.list_fields_to_ignore or (self.list_fields_to_ignore and basic_str not in self.list_fields_to_ignore)):

                gamma = self.dict_prior_gammas[basic_str]
                delta = self.dict_prior_deltas[basic_str]
                delta_x = self.delta_x
                delta_y = self.delta_y

                if gamma != 0.0:

                    IMAX = self.IMAX
                    JMAX = self.JMAX
                    NTDAMAX = self.NTDAMAX

                    field = ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data.copy() / np.sqrt(delta_x * delta_y)

                    result_old = np.copy(field)
                    result = np.copy(field)

                    for _ in range(self.MAX_ITERS_SOR):

                        for j in range(JMAX+1):
                            for i in range(IMAX+1):

                                if j == 0 and i == 0:
                                    diagonal = delta + gamma*(1/delta_x**2 + 1/delta_y**2)
                                    bracket = field[:, 0, 0] + gamma*(result_old[:, 0, 1] / delta_x**2 + result_old[:, 1, 0] / delta_y**2)
                                elif j == JMAX and i == 0:
                                    diagonal = delta + gamma*(1/delta_x**2 + 1/delta_y**2)
                                    bracket = field[:, JMAX, 0] + gamma*(result_old[:, JMAX, 1] / delta_x**2 + result[:, JMAX-1, 0] / delta_y**2)
                                elif j == 0 and i == IMAX:
                                    diagonal = delta + gamma*(1/delta_x**2 + 1/delta_y**2)
                                    bracket = field[:, 0, IMAX] + gamma*(result[:, 0, IMAX-1] / delta_x**2 + result_old[:, 1, IMAX] / delta_y**2)
                                elif j == JMAX and i == IMAX:
                                    diagonal = delta + gamma*(1/delta_x**2 + 1/delta_y**2)
                                    bracket = field[:, JMAX, IMAX] + gamma*(result[:, JMAX, IMAX-1] / delta_x**2 + result[:, JMAX-1, IMAX] / delta_y**2)
                                elif i == 0:
                                    diagonal = delta + gamma*(1/delta_x**2 + 2/delta_y**2)
                                    bracket = field[:, j, 0] + gamma*((result[:, j-1, 0] + result_old[:, j+1, 0]) / delta_y**2 + result_old[:, j, 1] / delta_x**2)
                                elif i == IMAX:
                                    diagonal = delta + gamma*(1/delta_x**2 + 2/delta_y**2)
                                    bracket = field[:, j, IMAX] + gamma*((result[:, j-1, IMAX] + result_old[:, j+1, IMAX]) / delta_y**2 + result[:, j, IMAX-1] / delta_x**2)
                                elif j == 0:
                                    diagonal = delta + gamma*(2/delta_x**2 + 1/delta_y**2)
                                    bracket = field[:, 0, i] + gamma*(result_old[:, 1, i] / delta_y**2 + (result[:, 0, i-1] + result_old[:, 0, i+1]) / delta_x**2)
                                elif j == JMAX:
                                    diagonal = delta + gamma*(1/delta_x**2 + 2/delta_y**2)
                                    bracket = field[:, JMAX, i] + gamma*(result[:, JMAX-1, i] / delta_y**2 + (result[:, JMAX, i-1] + result_old[:, JMAX, i+1]) / delta_x**2)
                                else:
                                    diagonal = delta + 2*gamma*(1/delta_x**2 + 1/delta_y**2)
                                    bracket = field[:, j, i] + gamma*((result[:, j-1, i] + result_old[:, j+1, i]) / delta_y**2 + (result[:, j, i-1] + result_old[:, j, i+1]) / delta_x**2)

                                result[:, j, i] = (1 - self.OMEGA_SOR) * result_old[:, j, i] + self.OMEGA_SOR / diagonal * bracket

                        result_old = result.copy()

                    ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data = result.copy()

                else:

                    ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data = ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data.copy() / (delta * np.sqrt(delta_x * delta_y))

        ds_fields = xr.merge([ds_subset_fields_params, ds_subset_fields_adj_or_adj_action_or_tlm_action])

        if ad_key_adj_or_adj_action_or_tlm_action == "tlm_action":
            # Some weird permission denied error if this file is not removed first.
            self.remove_dir(self.dict_ad_inp_nc_files[ad_key_adj_or_adj_action_or_tlm_action])
            ds_fields.to_netcdf(self.dict_ad_inp_nc_files[ad_key_adj_or_adj_action_or_tlm_action])
        else:
            # Some weird permission denied error if this file is not removed first.
            self.remove_dir(self.dict_ad_out_nc_files[ad_key_adj_or_adj_action_or_tlm_action])
            ds_fields.to_netcdf(self.dict_ad_out_nc_files[ad_key_adj_or_adj_action_or_tlm_action])

        if ad_key_adj_or_adj_action_or_tlm_action != "tlm_action" and self.dict_params_fields_or_scalars is not None:

            ds_subset_adj_or_adj_action = ds_subset_fields_adj_or_adj_action_or_tlm_action.copy()

            if self.list_fields_to_ignore:
                ds_subset_adj_or_adj_action = \
                ds_subset_adj_or_adj_action.drop_vars([var for var in ds_subset_adj_or_adj_action if var[:-1] in self.list_fields_to_ignore])

            for varb in ds_subset_adj_or_adj_action:

                if (not self.list_fields_to_ignore or (self.list_fields_to_ignore and varb[:-1] not in self.list_fields_to_ignore)) and self.dict_params_fields_or_scalars[varb[:-1]] == "scalar":
                        fieldb_sum = np.sum(ds_subset_adj_or_adj_action[varb].data)
                        ds_subset_adj_or_adj_action[varb] = xr.DataArray([fieldb_sum], dims=["scalar"], attrs=ds_subset_adj_or_adj_action[varb].attrs)

            return ds_subset_adj_or_adj_action

        elif ad_key_adj_or_adj_action_or_tlm_action == "tlm_action" and self.dict_params_fields_or_scalars is not None:

            dict_tlm_action_only_fields_vals = {}
            for var in ds_subset_fields_adj_or_adj_action_or_tlm_action:

                if not self.list_fields_to_ignore or (self.list_fields_to_ignore and var[:-1] not in self.list_fields_to_ignore):
                    if self.dict_tlm_action_fields_or_scalars[var] == "scalar":
                        dict_tlm_action_only_fields_vals[var] = ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data.flat[0].copy()
                    else:
                        dict_tlm_action_only_fields_vals[var] = ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data.copy()

            return self.create_ad_tlm_action_input_nc(dict_tlm_action_only_fields_vals)

    @beartype
    def eval_sqrt_prior_cov_action(self,
                                   ad_key_adj_or_tlm_action: str) -> Any:

        _ = self.eval_sqrt_prior_C_action(ad_key_adj_or_tlm_action)

        if ad_key_adj_or_tlm_action not in ["tlm_action", "adj"]:
            raise ValueError("eval_sqrt_prior_cov_action: Can only act on tlm or adj quantities.")

        if ad_key_adj_or_tlm_action == "tlm_action":
            ds_fields_adj_or_tlm_action = self.open_xr_ds(self.dict_ad_inp_nc_files[ad_key_adj_or_tlm_action], False)
            ad_subset_key = "tlm"
        else:
            # NOTE: eval_sqrt_prior_cov_action("adj") called in inexact_gn_hessian_cg is called on ds_out written to self.dict_ad_out_nc_files["adj"] and it has scalars as scalars and not fields.
            #       ds_fields_adj_or_tlm_action as a name implies fields but that is not true for this call of eval_sqrt_prior_cov_action. It seems to be working fine. See comments in inexact_gn_hessian_cg to understand more.
            ds_fields_adj_or_tlm_action = self.open_xr_ds(self.dict_ad_out_nc_files[ad_key_adj_or_tlm_action])
            ad_subset_key = "adj"

        ds_subset_fields_params = self.subset_of_ds(ds_fields_adj_or_tlm_action, "type", "nodiff")
        ds_subset_fields_adj_or_tlm_action = self.subset_of_ds(ds_fields_adj_or_tlm_action, "type", ad_subset_key)

        for var in ds_subset_fields_adj_or_tlm_action:

            basic_str = var[:-1]

            if (not self.list_fields_to_ignore or (self.list_fields_to_ignore and basic_str not in self.list_fields_to_ignore)):
                ds_subset_fields_adj_or_tlm_action[var].data = ds_subset_fields_adj_or_tlm_action[var].data * self.dict_prior_sigmas[basic_str] * self.ds_prior_X[basic_str + "d"].data

        ds_fields = xr.merge([ds_subset_fields_params, ds_subset_fields_adj_or_tlm_action])

        if ad_key_adj_or_tlm_action == "tlm_action":
            # Some weird permission denied error if this file is not removed first.
            self.remove_dir(self.dict_ad_inp_nc_files[ad_key_adj_or_tlm_action])
            ds_fields.to_netcdf(self.dict_ad_inp_nc_files[ad_key_adj_or_tlm_action])
        else:
            # Some weird permission denied error if this file is not removed first.
            self.remove_dir(self.dict_ad_out_nc_files[ad_key_adj_or_tlm_action])
            ds_fields.to_netcdf(self.dict_ad_out_nc_files[ad_key_adj_or_tlm_action])            
        
        if ad_key_adj_or_tlm_action == "adj" and self.dict_params_fields_or_scalars is not None:

            ds_subset_adj_or_adj_action = ds_subset_fields_adj_or_tlm_action.copy()

            if self.list_fields_to_ignore:
                ds_subset_adj_or_adj_action = \
                ds_subset_adj_or_adj_action.drop_vars([var for var in ds_subset_adj_or_adj_action if var[:-1] in self.list_fields_to_ignore])

            for varb in ds_subset_adj_or_adj_action:

                if (not self.list_fields_to_ignore or (self.list_fields_to_ignore and varb[:-1] not in self.list_fields_to_ignore)) and self.dict_params_fields_or_scalars[varb[:-1]] == "scalar":
                        fieldb_sum = np.sum(ds_subset_adj_or_adj_action[varb].data)
                        ds_subset_adj_or_adj_action[varb] = xr.DataArray([fieldb_sum], dims=["scalar"], attrs=ds_subset_adj_or_adj_action[varb].attrs)

            return ds_subset_adj_or_adj_action

        elif ad_key_adj_or_tlm_action == "tlm_action" and self.dict_params_fields_or_scalars is not None:

            dict_tlm_action_only_fields_vals = {}
            for var in ds_subset_fields_adj_or_tlm_action:

                if not self.list_fields_to_ignore or (self.list_fields_to_ignore and var[:-1] not in self.list_fields_to_ignore):
                    if self.dict_tlm_action_fields_or_scalars[var] == "scalar":
                        dict_tlm_action_only_fields_vals[var] = ds_subset_fields_adj_or_tlm_action[var].data.flat[0].copy()
                    else:
                        dict_tlm_action_only_fields_vals[var] = ds_subset_fields_adj_or_tlm_action[var].data.copy()

            return self.create_ad_tlm_action_input_nc(dict_tlm_action_only_fields_vals)

    @beartype
    def eval_sqrt_prior_covT_action(self,
                                    ad_key_adj_or_adj_action: str) -> Any:

        if ad_key_adj_or_adj_action not in ["adj", "adj_action"]:
            raise ValueError("eval_sqrt_prior_covT_action: Can only act on adj or adj_action quantities.")

        ds_adj_fields = self.open_xr_ds(self.dict_ad_out_nc_files[ad_key_adj_or_adj_action])

        ds_subset_params_fields = self.subset_of_ds(ds_adj_fields, "type", "nodiff")
        ds_subset_adj_fields = self.subset_of_ds(ds_adj_fields, "type", "adj")

        for varb in ds_subset_adj_fields:

            basic_str = varb[:-1]

            if not self.list_fields_to_ignore or (self.list_fields_to_ignore and basic_str not in self.list_fields_to_ignore):
                ds_subset_adj_fields[varb].data = ds_subset_adj_fields[varb].data * self.dict_prior_sigmas[basic_str] * self.ds_prior_X[basic_str + "d"].data

        ds_subset_params_fields = ds_subset_params_fields.assign_coords({dim: self.ds_subset_params[dim] for dim in self.ds_subset_params.dims})
        ds_subset_adj_fields = ds_subset_adj_fields.assign_coords({dim: self.ds_subset_params[dim] for dim in self.ds_subset_params.dims})

        ds_adj_fields = xr.merge([ds_subset_params_fields, ds_subset_adj_fields])
        # Some weird permission denied error if this file is not removed first.
        self.remove_dir(self.dict_ad_out_nc_files[ad_key_adj_or_adj_action])
        ds_adj_fields.to_netcdf(self.dict_ad_out_nc_files[ad_key_adj_or_adj_action])

        return self.eval_sqrt_prior_C_action(ad_key_adj_or_adj_action_or_tlm_action = ad_key_adj_or_adj_action)

    @beartype
    def eval_prior_preconditioned_misfit_hessian_action(self) -> Any:

        ds_subset_inp_tlm_action = self.eval_sqrt_prior_cov_action(ad_key_adj_or_tlm_action = "tlm_action")
        ds_subset_misfit_hessian_action = self.eval_misfit_hessian_action()

        return self.eval_sqrt_prior_covT_action(ad_key_adj_or_adj_action = "adj_action")

    @beartype
    def eval_prior_preconditioned_hessian_action(self) -> Any:

        ds_inp_tlm_action = self.open_xr_ds(self.dict_ad_inp_nc_files["tlm_action"], False)
        ds_subset_tlm = self.subset_of_ds(ds_inp_tlm_action, "type", "tlm")

        ds_subset_prior_precond_misfit_hess_action = self.eval_prior_preconditioned_misfit_hessian_action()

        for var in ds_subset_prior_precond_misfit_hess_action:

            basic_str = var[:-1]

            if self.dict_params_fields_or_scalars and self.dict_params_fields_or_scalars[basic_str] == "scalar":
                ds_subset_prior_precond_misfit_hess_action[var].data = ds_subset_prior_precond_misfit_hess_action[var].data + ds_subset_tlm[basic_str + "d"].data.flat[0]
            else:
                ds_subset_prior_precond_misfit_hess_action[var].data = ds_subset_prior_precond_misfit_hess_action[var].data + ds_subset_tlm[basic_str + "d"].data

        # NOTE: The final result is not written to any nc file
        return ds_subset_prior_precond_misfit_hess_action

    @beartype
    def conjugate_gradient(self,
                           ds_subset_gradient: Any,
                           tolerance_type: str = "superlinear",
                           MAX_ITERS_CG: Optional[int] = None,
                           max_prev_v_hat_diagnostics: Optional[int] = 5) -> Any:

        if max_prev_v_hat_diagnostics is not None and max_prev_v_hat_diagnostics < 1:
            raise ValueError("conjugate_gradient: max_prev_v_hat_diagnostics should be at least 1 if it is defined.")
        list_subset_v_hat_prev = [] # Won't be used if max_prev_v_hat_diagnostics is not defined
        list_str_angles_H_orthogonal_check = [] # Won't be used if max_prev_v_hat_diagnostics is not defined

        ds_subset_gradient_hat = self.eval_sqrt_prior_covT_action(ad_key_adj_or_adj_action = "adj")
        norm_gradient_hat = self.l2_inner_product([ds_subset_gradient_hat, ds_subset_gradient_hat], ["adj", "adj"])**0.5

        ds_subset_p_hat = self.linear_sum([ds_subset_gradient_hat, ds_subset_gradient_hat], 
                                          [0.0, 0.0], ["adj", "adj"])
        ds_subset_r_hat = self.linear_sum([ds_subset_gradient_hat, ds_subset_gradient_hat], 
                                          [0.0, -1.0], ["adj", "adj"])
        
        ds_subset_v_hat = self.linear_sum([ds_subset_gradient_hat, ds_subset_gradient_hat], 
                                          [0.0, -1.0], ["adj", "adj"])
        ds_subset_v_hat = self.exch_tlm_adj_nc(ds_subset_v_hat, og_type = "adj")

        iters = 0

        while True:

            iters = iters + 1
            print(f"CG iter {iters}")

            ds_subset_H_hat_v_hat = self.eval_prior_preconditioned_hessian_action()

            if max_prev_v_hat_diagnostics is not None and len(list_subset_v_hat_prev) > 0:
                v_curr = ds_subset_v_hat.copy()
                H_v_curr = ds_subset_H_hat_v_hat.copy()
                norm_H_v_curr = self.l2_inner_product([v_curr, H_v_curr], ["tlm", "adj"])**0.5

                for i in range(len(list_subset_v_hat_prev)):
                    v_prev, H_v_prev = list_subset_v_hat_prev[i]
                    norm_H_v_prev = self.l2_inner_product([v_prev, H_v_prev], ["tlm", "adj"])**0.5

                    inner_prod_1 = self.l2_inner_product([v_prev, H_v_curr], ["tlm", "adj"])
                    inner_prod_2 = self.l2_inner_product([v_curr, H_v_prev], ["tlm", "adj"])

                    cos_1 = inner_prod_1 / (norm_H_v_prev * norm_H_v_curr)
                    cos_2 = inner_prod_2 / (norm_H_v_prev * norm_H_v_curr)

                    angle_1 = np.degrees(np.arccos(np.clip(cos_1, -1.0, 1.0)))
                    angle_2 = np.degrees(np.arccos(np.clip(cos_2, -1.0, 1.0)))

                    label = f"v_old_{len(list_subset_v_hat_prev) - i}"
                    list_str_angles_H_orthogonal_check.append(f"{label}: {angle_1:.1f}/{angle_2:.1f}")

                print("H-orthogonality angles (deg):", " | ".join(list_str_angles_H_orthogonal_check))
                list_str_angles_H_orthogonal_check.clear()

            v_hatT_H_hat_v_hat = self.l2_inner_product([ds_subset_v_hat, ds_subset_H_hat_v_hat], ["tlm", "adj"])
            norm_r_hat_old = self.l2_inner_product([ds_subset_r_hat, ds_subset_r_hat], ["adj", "adj"])**0.5

            if v_hatT_H_hat_v_hat < 0:

                print("conjugate_gradient: Hessian no longer positive definite.")

                p_hatT_g_hat = self.l2_inner_product([ds_subset_p_hat, ds_subset_gradient_hat], ["adj", "adj"])
                norm_p_hat = self.l2_inner_product([ds_subset_p_hat, ds_subset_p_hat], ["adj", "adj"])**0.5
                cos = p_hatT_g_hat / (norm_p_hat * norm_gradient_hat)
                angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))

                print("Angle between p_hat and g_hat in degrees: ", angle)

                return ds_subset_p_hat

            alpha_hat = norm_r_hat_old**2 / v_hatT_H_hat_v_hat

            ds_subset_p_hat = self.linear_sum([ds_subset_p_hat, ds_subset_v_hat],
                                              [1.0, alpha_hat], ["adj", "tlm"])

            ds_subset_r_hat = self.linear_sum([ds_subset_r_hat, ds_subset_H_hat_v_hat],
                                              [1.0, -alpha_hat], ["adj", "adj"])

            norm_r_hat = self.l2_inner_product([ds_subset_r_hat, ds_subset_r_hat], ["adj", "adj"])**0.5

            if tolerance_type == "linear":
                eps_hat_TOL = 0.5*norm_gradient_hat
            elif tolerance_type == "superlinear":
                eps_hat_TOL = min(0.5, np.sqrt(norm_gradient_hat))*norm_gradient_hat
            elif tolerance_type == "quadratic":
                eps_hat_TOL = min(0.5, norm_gradient_hat)*norm_gradient_hat
            else:
                raise ValueError("conjugate_gradient: Invalid tolerance_type.")

            print(f"eps_TOL_CG: {eps_hat_TOL}, norm_r_hat: {norm_r_hat}")
            if norm_r_hat <= eps_hat_TOL:
                print("conjugate_gradient: Convergence.")

                p_hatT_g_hat = self.l2_inner_product([ds_subset_p_hat, ds_subset_gradient_hat], ["adj", "adj"])
                norm_p_hat = self.l2_inner_product([ds_subset_p_hat, ds_subset_p_hat], ["adj", "adj"])**0.5
                cos = p_hatT_g_hat / (norm_p_hat * norm_gradient_hat)
                angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))

                print("Angle between p_hat and g_hat in degrees: ", angle)

                return ds_subset_p_hat

            if MAX_ITERS_CG is not None and iters == MAX_ITERS_CG:
                print("Maximum CG iterations reached.")

                p_hatT_g_hat = self.l2_inner_product([ds_subset_p_hat, ds_subset_gradient_hat], ["adj", "adj"])
                norm_p_hat = self.l2_inner_product([ds_subset_p_hat, ds_subset_p_hat], ["adj", "adj"])**0.5
                cos = p_hatT_g_hat / (norm_p_hat * norm_gradient_hat)
                angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))

                print("Angle between p_hat and g_hat in degrees: ", angle)

                return ds_subset_p_hat

            beta_hat = norm_r_hat**2 / norm_r_hat_old**2

            if max_prev_v_hat_diagnostics is not None:

                if len(list_subset_v_hat_prev) >= max_prev_v_hat_diagnostics:
                    list_subset_v_hat_prev.pop(0)

                # Now append current directions to the end to serve as the most recent v_hat_prev, H_hat_v_hat_prev
                list_subset_v_hat_prev.append((ds_subset_v_hat.copy(), ds_subset_H_hat_v_hat.copy()))

            ds_subset_v_hat = self.linear_sum([ds_subset_v_hat, ds_subset_r_hat],
                                              [beta_hat, 1.0], ["tlm", "adj"])

            dict_tlm_action_only_fields_vals = {}
            for var in ds_subset_v_hat:

                if not self.list_fields_to_ignore or (self.list_fields_to_ignore and var[:-1] not in self.list_fields_to_ignore):
                    if self.dict_tlm_action_fields_or_scalars[var] == "scalar":
                        dict_tlm_action_only_fields_vals[var] = ds_subset_v_hat[var].data.flat[0].copy()
                    else:
                        dict_tlm_action_only_fields_vals[var] = ds_subset_v_hat[var].data.copy()

            ds_subset_v_hat = self.create_ad_tlm_action_input_nc(dict_tlm_action_only_fields_vals)

    @beartype
    def inexact_gn_hessian_cg(self,
                              MAX_ITERS: int,
                              init_alpha_cg: float = 1.0,
                              min_alpha_cg_tol: float = 1.e-10,
                              init_alpha_gd: float = 1.0,
                              min_alpha_gd_tol: float = 1.e-10,
                              cg_tolerance_type: str = "superlinear",
                              c1: float = 1.e-4,
                              MAX_ITERS_CG: Optional[int] = None,
                              max_prev_v_hat_diagnostics: Optional[int] = 5) -> Any:

        if self.dirpath_store_states is not None:

            if not os.path.isdir(self.dirpath_store_states):
                self.make_dir(self.dirpath_store_states)
            if os.path.isdir(self.dirpath_store_states + "/" + "inexact_gn_hessian_cg"):
                self.remove_dir(self.dirpath_store_states + "/" + "inexact_gn_hessian_cg")

            self.make_dir(self.dirpath_store_states + "/" + "inexact_gn_hessian_cg")

            log_file = self.dirpath_store_states + "/inexact_gn_hessian_cg/" + "inexact_gn_hessian_cg.log"
            with open(log_file, "a") as f:
                f.write(f"Iteration 0: Cost = {self.ds_subset_costs['fc'].data[0]:.6f}, Misfit Cost = {self.ds_subset_costs['fc_data'].data[0]:.6f}, Regularization Cost = {self.ds_subset_costs['fc_reg'].data[0]:.6f}\n")

            self.ds_subset_params.to_netcdf(self.dirpath_store_states + "/inexact_gn_hessian_cg/" + f"state_GNHessCG_iter_0.nc")
            self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dirpath_store_states + "/inexact_gn_hessian_cg/" + "state_GNHessCG_iter_0_fields.nc")

            path_sico_out_nc = self.dict_sico_out_folders["nodiff"] + "/" + self.filename_final_sim_output
            if not os.path.isfile(path_sico_out_nc):
                raise ValueError(f"inexact_gn_hessian_cg: Final simulation output file {path_sico_out_nc} is missing.")
            self.copy_dir(path_sico_out_nc, self.dirpath_store_states + "/inexact_gn_hessian_cg/" + f"final_sim_output_GNHessCG_iter_0_fields.nc")

        print("---------------------------------------------------------------------------------------------------------------")
        print(f"Initial fc = {self.ds_subset_costs['fc'].data[0]}, fc_data = {self.ds_subset_costs['fc_data'].data[0]}, fc_reg = {self.ds_subset_costs['fc_reg'].data[0]}")
        print("---------------------------------------------------------------------------------------------------------------")

        for i in range(MAX_ITERS):

            ds_subset_params = self.ds_subset_params.copy()
            ds_subset_gradient = self.eval_gradient()
            ds_subset_descent_dir_hat = self.conjugate_gradient(ds_subset_gradient, cg_tolerance_type, MAX_ITERS_CG, max_prev_v_hat_diagnostics)

            ds_out = xr.merge([ds_subset_params, ds_subset_descent_dir_hat])
            # Some weird permission denied error if this file is not removed first.
            self.remove_dir(self.dict_ad_out_nc_files["adj"])
            ds_out.to_netcdf(self.dict_ad_out_nc_files["adj"])
            # NOTE: ds_out has scalars as scalars and not fields and it still seems to be working fine.
            # NOTE: On further thought, this is the only way to do it right. There is no other way.
            #       This is because ds_subset_descent_dir_hat was computed using CG, which necessarily needs us to represent a scalar parameter and its gradient as only one
            #       dimension in the control space. Once you solve for the ds_subset_descent_dir_hat, you cannot convert a scalar direction back into the field.
            #       To see why, understand that for scalars, you accumulate the adjoint from the vector field i.e. varb = SUM(varb), but you cannot go in the reverse direction since
            #       the vector varb field is not going to be uniform.
            # NOTE: It is working right because all we do here is convert ds_subset_descent_dir_hat to ds_subset_descent_dir which doesn't need to go through any SICOPOLIS operations to trigger errors.
            #       We just get ds_subset_descent_dir and then find the new parameters ds_subset_params_new and write them to file and move on. So this minor issue or "bug" doesn't cause any problems.

            ds_subset_descent_dir = self.eval_sqrt_prior_cov_action("adj")

            alpha, self.ds_subset_costs = self.line_search(ds_subset_gradient,
                                                           ds_subset_descent_dir,
                                                           init_alpha_cg, min_alpha_cg_tol, c1)

            if alpha <= min_alpha_cg_tol:

                print(f"Step size alpha {alpha} is too small for any real improvement with Inexact GN-Hessian CG, switching to gradient descent for this step.")

                ds_subset_neg_gradient = self.linear_sum([ds_subset_gradient, ds_subset_gradient], 
                                                        [0.0, -1.0], ["adj", "adj"])

                alpha, self.ds_subset_costs = self.line_search(ds_subset_gradient,
                                                               ds_subset_neg_gradient,
                                                               init_alpha_gd, min_alpha_gd_tol, c1)

                ds_subset_params_new = self.linear_sum([ds_subset_params, ds_subset_neg_gradient], 
                                                       [1.0, alpha], ["nodiff", "adj"])
                
            else:

                ds_subset_params_new = self.linear_sum([ds_subset_params, ds_subset_descent_dir], 
                                                       [1.0, alpha], ["nodiff", "adj"])

            self.write_params(ds_subset_params_new)
            self.ds_subset_params = ds_subset_params_new.copy()

            self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dict_ad_inp_nc_files["adj"])

            if self.dirpath_store_states is not None:

                with open(log_file, "a") as f:
                    f.write(f"Iteration {i+1}: Cost = {self.ds_subset_costs['fc'].data[0]:.6f}, Misfit Cost = {self.ds_subset_costs['fc_data'].data[0]:.6f}, Regularization Cost = {self.ds_subset_costs['fc_reg'].data[0]:.6f}\n")

                self.ds_subset_params.to_netcdf(self.dirpath_store_states + "/inexact_gn_hessian_cg/" + f"state_GNHessCG_iter_{i+1}.nc")
                ds_subset_gradient.to_netcdf(self.dirpath_store_states + "/inexact_gn_hessian_cg/" + f"gradient_GNHessCG_iter_{i}.nc")
                self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dirpath_store_states + "/inexact_gn_hessian_cg/" + f"state_GNHessCG_iter_{i+1}_fields.nc")

                path_sico_out_nc = self.dict_sico_out_folders["nodiff"] + "/" + self.filename_final_sim_output
                if not os.path.isfile(path_sico_out_nc):
                    raise ValueError(f"inexact_gn_hessian_cg: Final simulation output file {path_sico_out_nc} is missing.")
                self.copy_dir(path_sico_out_nc, self.dirpath_store_states + "/inexact_gn_hessian_cg/" + f"final_sim_output_GNHessCG_iter_{i+1}_fields.nc")

            print("---------------------------------------------------------------------------------------------------------------")
            print(f"Outer iter {i+1}, fc = {self.ds_subset_costs['fc'].data[0]}, fc_data = {self.ds_subset_costs['fc_data'].data[0]}, fc_reg = {self.ds_subset_costs['fc_reg'].data[0]}")
            print("---------------------------------------------------------------------------------------------------------------")

        return self.ds_subset_params

    @beartype
    def revd(self,
             sampling_param_k_REVD: int, 
             oversampling_param_p_REVD: int = 10,
             mode: str = "misfit_prior_precond",
             str_pass: str = "single_approx",
             output_freq: Optional[int] = 10,
             Omega: Optional[Float[np.ndarray, "dim_m dim_l0"]] = None,
             Y: Optional[Float[np.ndarray, "dim_m dim_l0"]] = None,
             Q: Optional[Float[np.ndarray, "dim_m dim_l0"]] = None,
             MQ: Optional[Float[np.ndarray, "dim_m dim_l0"]] = None) -> Tuple[Float[np.ndarray, "dim_m dim_l"], Float[np.ndarray, "dim_m dim_l"], Float[np.ndarray, "dim_m dim_l"], Float[np.ndarray, "dim_m dim_l"], Float[np.ndarray, "dim_m dim_k"], Float[np.ndarray, "dim_k"], Float[np.ndarray, "dim_l dim_k"]]:

        # See remark 5.4 of Halko, Martinsson, and Tropp's paper to understand what might be missing from single pass
        # Basic idea for single pass was to truncate Q to get an overdetermined system to invert which will be more stable, but the Q in remark 5.4 is made from k leading singular vectors, and I don't have them
        # This is because the first k columns of Q are not necessarily the k leading singular vectors

        # Single pass or double pass, returned matrices will have Q.shape[1] + sampling_param_k_REVD columns
        # For now, since incremental REVD setuo is deactivated due to questionable rigor of truncating Q, Y, Omega, etc., returned matrices will always have sampling_param_k_REVD columns

        if mode not in ["misfit_prior_precond", "full_prior_precond"]:
            raise ValueError("revd: Can only decompose full prior-preconditioned Hessian or misfit prior-preconditioned Hessian.")
        elif mode == "full_prior_precond":
            func_hessian_action = self.eval_prior_preconditioned_hessian_action
        elif mode == "misfit_prior_precond":
            func_hessian_action = self.eval_prior_preconditioned_misfit_hessian_action

        if str_pass not in ["double_precise", "single_approx"]:
            raise ValueError("revd: str_pass can olny be double_precise or single_approx.")
        elif str_pass == "single_approx":
            warnings.warn("revd: Check the comments at the start of this function to see what might be missing from the single pass, it's essentially remark 5.4 of Halko, Martinsson, Tropp's paper.", RuntimeWarning)

        if not (Omega is None and Y is None and Q is None and MQ is None) and not (Omega is not None and Y is not None and Q is not None and MQ is not None):
            raise ValueError("revd: Omega, Y, Q, MQ must be either None or all not None.")

        if output_freq is not None and output_freq <= 0:
            raise ValueError("revd: output_freq should be greater than 0 when it is defined.")

        ds_subset_omega = self.create_ad_tlm_action_input_nc(bool_randomize = True)

        ds_subset_omega_dummy_for_MQ = ds_subset_omega.copy()
        ds_subset_omega_dummy_for_MQ = ds_subset_omega_dummy_for_MQ.rename({var: var[:-1] + "b" for var in ds_subset_omega_dummy_for_MQ})
        for var in ds_subset_omega_dummy_for_MQ.data_vars:
            ds_subset_omega_dummy_for_MQ[var].attrs["type"] = "adj"

        m, omega = self.flattened_vector(ds_subset_omega)

        if Omega is None and Y is None and Q is None and MQ is None:
            list_ds_subset_Q_cols = []
            Omega = omega.reshape(-1, 1)
            Y = np.empty((0, 0))
            Q = np.empty((0, 0))
            MQ = np.empty((0, 0))

            start_idx = 0
            l = sampling_param_k_REVD + oversampling_param_p_REVD
        else:
            raise ValueError("revd: The incremental setup no longer works since it's unclear how to correctly truncate Omega, Y, etc. to shape k in the first pass.")

            if Q.shape != MQ.shape or Omega.shape != Q.shape or Omega.shape != MQ.shape or Y.shape != Omega.shape or Y.shape != Q.shape or Y.shape != MQ.shape:
                raise ValueError("revd: Dimensions of given Omega and Y and Q and MQ do not match!")
            elif Q.shape[0] != m:
                raise ValueError("revd: Dimensions of given Omega, Y, Q, MQ and random vector ds_subset_omega do not match!")

            Omega = np.hstack([Omega, omega.reshape(-1, 1)])
            list_ds_subset_Q_cols = [self.construct_ds(q.reshape(-1,), ds_subset_omega) for q in Q.T]

            start_idx = len(list_ds_subset_Q_cols)
            l = sampling_param_k_REVD + oversampling_param_p_REVD + Q.shape[1]

        if self.dirpath_store_states is not None:

            if not os.path.isdir(self.dirpath_store_states):
                self.make_dir(self.dirpath_store_states)
            if os.path.isdir(self.dirpath_store_states + "/" + f"REVD_{str_pass}"):
                self.remove_dir(self.dirpath_store_states + "/" + f"REVD_{str_pass}")

            self.make_dir(self.dirpath_store_states + "/" + f"REVD_{str_pass}")

            if mode == "full_prior_precond":
                suffix = "full"
            if mode == "misfit_prior_precond":
                suffix = "misfit"

        while True:
            ds_subset_y = func_hessian_action()
            _, y = self.flattened_vector(ds_subset_y)

            if Y.size > 0:
                Y = np.hstack([Y, y.reshape(-1, 1)])
            else:
                Y = y.reshape(-1, 1)
            
            if Q.size > 0:
                q_tilde = y - Q @ (Q.T @ y)
                q = q_tilde / np.linalg.norm(q_tilde)
                Q = np.hstack([Q, q.reshape(-1, 1)])
            else:
                q = y / np.linalg.norm(y)
                Q = q.reshape(-1, 1)

            ds_subset_q = self.construct_ds(q.reshape(-1,), ds_subset_omega)
            list_ds_subset_Q_cols.append(ds_subset_q)

            if output_freq is not None and Q.shape[1] > oversampling_param_p_REVD and ((Q.shape[1] - oversampling_param_p_REVD) % output_freq == 0 or Q.shape[1] == l):

                if str_pass == "double_precise":

                    for ds_subset_q in list_ds_subset_Q_cols[start_idx:]:

                        dict_tlm_action_only_fields_vals = {}
                        for var in ds_subset_q:

                            if not self.list_fields_to_ignore or (self.list_fields_to_ignore and var[:-1] not in self.list_fields_to_ignore):
                                if self.dict_tlm_action_fields_or_scalars[var] == "scalar":
                                    dict_tlm_action_only_fields_vals[var] = ds_subset_q[var].data.flat[0].copy()
                                else:
                                    dict_tlm_action_only_fields_vals[var] = ds_subset_q[var].data.copy()

                        _ = self.create_ad_tlm_action_input_nc(dict_tlm_action_only_fields_vals)

                        ds_subset_Mq = func_hessian_action()

                        _, Mq = self.flattened_vector(ds_subset_Mq)
                        if MQ.size > 0:
                            MQ = np.hstack([MQ, Mq.reshape(-1, 1)])
                        else:
                            MQ = Mq.reshape(-1, 1)

                    T = Q.T @ MQ

                    start_idx = len(list_ds_subset_Q_cols)

                elif str_pass == "single_approx":

                    # MQ is not needed at all here but the function needs it to be returned and have some shape when picked up for subsequent runs (this functionality is turned off for now but still keeping this code here).
                    MQ = Q.copy()

                    # Use pinv (pseudo-inverse) for stable inversion using SVD under the hood instead of simply inv
                    T = (Q.T @ Y) @ np.linalg.pinv(Q.T @ Omega)
                    cond_number = np.linalg.cond(Q.T @ Omega)
                    print("Condition number of Q.T @ Omega: ", cond_number)
                    _, S_Tsvd, _ = np.linalg.svd(Q.T @ Omega, full_matrices = False)
                    sigma_min = np.min(S_Tsvd)
                    print("Minimum singular value of Q.T @ Omega: ", sigma_min)
                    print("Condition number of Q.T @ Omega explicitly computed: ", np.max(S_Tsvd) / sigma_min)

                    if cond_number > 1e5 or sigma_min < 1e-10:
                        print("Warning: Q.T @ Omega is poorly conditioned. Consider switching to two-pass REVD!")

                sym_err = np.linalg.norm(T - T.T) / np.linalg.norm(T)
                print("Relative symmetry error of T: ", sym_err)

                # This is more theoretically correct, but then symmetry of T is not guaranteed.
                # eig in the name indicates use of np.linalg.eig
                Lambda_eig_unsymm, _ = np.linalg.eig(T)

                # Make sure T is symmetric
                T = 0.5*(T + T.T)
                # eig in the name indicates use of np.linalg.eig
                Lambda_eig_symm, _ = np.linalg.eig(T)
                Lambda, S = np.linalg.eigh(T)

                print(f"Complex parts check (imag > {np.finfo(float).eps:.1e}): ")
                print("eig (unsymm): ", self.has_complex_parts(Lambda_eig_unsymm))
                print("eig (symm):   ", self.has_complex_parts(Lambda_eig_symm))
                print("eigh:         ", self.has_complex_parts(Lambda))

                Lambda_sorted = np.sort(np.real(Lambda))
                Lambda_eig_unsymm_sorted = np.sort(np.real(Lambda_eig_unsymm))
                Lambda_eig_symm_sorted = np.sort(np.real(Lambda_eig_symm))
                eig_diff_unsymm = np.linalg.norm(Lambda_eig_unsymm_sorted - Lambda_sorted) / np.linalg.norm(Lambda_sorted)
                eig_diff_symm   = np.linalg.norm(Lambda_eig_symm_sorted - Lambda_sorted) / np.linalg.norm(Lambda_sorted)

                print("Relative error: ")
                print("eig (unsymm) vs eigh: ", eig_diff_unsymm)
                print("eig (symmetrized) vs eigh: ", eig_diff_symm)

                Lambda_idx = np.argsort(Lambda)[::-1]
                # Get all eigenmodes except the last oversampling_param_p_REVD
                Lambda = Lambda[Lambda_idx[:(-oversampling_param_p_REVD)]]
                S = S[:, Lambda_idx[:(-oversampling_param_p_REVD)]]

                U = Q @ S

                num_eigenmodes = Q.shape[1] - oversampling_param_p_REVD
                print(f"First {num_eigenmodes} eigenvalues: {Lambda}")

                if self.dirpath_store_states is not None:

                    np.save(self.dirpath_store_states + "/" + f"REVD_{str_pass}" + "/" + f"Omega_{suffix}_{num_eigenmodes}.npy", Omega)
                    np.save(self.dirpath_store_states + "/" + f"REVD_{str_pass}" + "/" + f"Y_{suffix}_{num_eigenmodes}.npy", Y)
                    np.save(self.dirpath_store_states + "/" + f"REVD_{str_pass}" + "/" + f"Q_{suffix}_{num_eigenmodes}.npy", Q)
                    np.save(self.dirpath_store_states + "/" + f"REVD_{str_pass}" + "/" + f"MQ_{suffix}_{num_eigenmodes}.npy", MQ)
                    np.save(self.dirpath_store_states + "/" + f"REVD_{str_pass}" + "/" + f"U_{suffix}_{num_eigenmodes}.npy", U)
                    np.save(self.dirpath_store_states + "/" + f"REVD_{str_pass}" + "/" + f"Lambda_{suffix}_{num_eigenmodes}.npy", Lambda)
                    np.save(self.dirpath_store_states + "/" + f"REVD_{str_pass}" + "/" + f"S_{suffix}_{num_eigenmodes}.npy", S)

            if Q.shape[1] == l:
                break

            ds_subset_omega = self.create_ad_tlm_action_input_nc(bool_randomize = True)
            _, omega = self.flattened_vector(ds_subset_omega)
            Omega = np.hstack([Omega, omega.reshape(-1, 1)])

        return Omega, Y, Q, MQ, U, Lambda, S

    @beartype
    def forward_uq_propagation(self,
                               U_misfit: Float[np.ndarray, "dim_m dim_l"],
                               Lambda_misfit: Float[np.ndarray, "dim_l"]) -> Tuple[float, float, float]:

        assert U_misfit.shape[1] == Lambda_misfit.shape[0]

        self.copy_dir(self.src_dir + "/driveradjoint", self.src_dir + "/driveradjoint_orig")
        self.copy_dir(self.src_dir + "/driveradjointqoi", self.src_dir + "/driveradjointqoi_orig")
        self.copy_dir(self.src_dir + "/driveradjointqoi", self.src_dir + "/driveradjoint")

        ds_subset_gradient_qoi = self.eval_gradient()

        self.copy_dir(self.src_dir + "/driveradjoint_orig", self.src_dir + "/driveradjoint")

        ds_subset_CT_XT_SigmaT_gradient_qoi = self.eval_sqrt_prior_covT_action(ad_key_adj_or_adj_action = "adj")

        sigma_B_squared = self.l2_inner_product([ds_subset_CT_XT_SigmaT_gradient_qoi, ds_subset_CT_XT_SigmaT_gradient_qoi], ["adj", "adj"])
        sigma_P_squared = sigma_B_squared

        for i in range(Lambda_misfit.shape[0]):

            ds_subset_vi = self.construct_ds(U_misfit[:, i], ds_subset_CT_XT_SigmaT_gradient_qoi)
            sigma_P_squared = sigma_P_squared - Lambda_misfit[i] / (Lambda_misfit[i] + 1) * self.l2_inner_product([ds_subset_CT_XT_SigmaT_gradient_qoi, ds_subset_vi], ["adj", "adj"])**2

        delta_sigma_qoi_squared = 1 - sigma_P_squared/sigma_B_squared

        return sigma_B_squared, sigma_P_squared, delta_sigma_qoi_squared

    @beartype
    def sample_prior_C(self) -> Any:

        ds_subset_x = self.create_ad_tlm_action_input_nc(bool_randomize = True)
        ds_subset_prior_C_x = self.eval_sqrt_prior_C_action(ad_key_adj_or_adj_action_or_tlm_action = "tlm_action")

        return ds_subset_prior_C_x

    @beartype
    def sample_prior(self) -> Any:

        ds_subset_x = self.create_ad_tlm_action_input_nc(bool_randomize = True)
        ds_subset_prior_SigmaXC_x = self.eval_sqrt_prior_cov_action(ad_key_adj_or_tlm_action = "tlm_action")

        return ds_subset_prior_SigmaXC_x

    @beartype
    def sample_posterior(self,
                         U_misfit: Float[np.ndarray, "dim_m dim_l"],
                         Lambda_misfit: Float[np.ndarray, "dim_l"]) -> Any:

        assert U_misfit.shape[1] == Lambda_misfit.shape[0]

        ds_subset_x = self.create_ad_tlm_action_input_nc(bool_randomize = True)

        ds_subset_vi = self.construct_ds(U_misfit[:, 0], ds_subset_x)

        factor = self.l2_inner_product([ds_subset_vi, ds_subset_x], ["tlm", "tlm"])
        factor = factor * (1 - 1/(1 + Lambda_misfit[0])**0.5)

        ds_subset_V_S_VT = self.linear_sum([ds_subset_vi, ds_subset_vi],
                                          [factor, 0.0], ["tlm", "tlm"])

        for i in range(1, Lambda_misfit.shape[0]):

           ds_subset_vi = self.construct_ds(U_misfit[:, i], ds_subset_x)

           factor = self.l2_inner_product([ds_subset_vi, ds_subset_x], ["tlm", "tlm"])
           factor = factor * (1 - 1/(1 + Lambda_misfit[i])**0.5)

           ds_subset_V_S_VT = self.linear_sum([ds_subset_V_S_VT, ds_subset_vi],
                                             [1.0, factor], ["tlm", "tlm"])

        ds_subset_sample_bracket = self.linear_sum([ds_subset_x, ds_subset_V_S_VT],
                                           [1.0, -1.0], ["tlm", "tlm"])

        dict_tlm_action_only_fields_vals = {}
        for var in ds_subset_sample_bracket:

            if not self.list_fields_to_ignore or (self.list_fields_to_ignore and var[:-1] not in self.list_fields_to_ignore):
                if self.dict_tlm_action_fields_or_scalars[var] == "scalar":
                    dict_tlm_action_only_fields_vals[var] = ds_subset_sample_bracket[var].data.flat[0].copy()
                else:
                    dict_tlm_action_only_fields_vals[var] = ds_subset_sample_bracket[var].data.copy()

        _ = self.create_ad_tlm_action_input_nc(dict_tlm_action_only_fields_vals)
        return self.eval_sqrt_prior_cov_action(ad_key_adj_or_tlm_action = "tlm_action")

    @beartype
    def pointwise_marginals(self,
                            type_marginals: str,
                            N_samples: int = 10,
                            U_misfit: Optional[Float[np.ndarray, "dim_m dim_l"]] = None,
                            Lambda_misfit: Optional[Float[np.ndarray, "dim_l"]] = None) -> Any:

        if type_marginals not in ["prior_C", "prior", "posterior"]:
            raise ValueError("pointwise_marginals: type_marginals can only be prior_C, prior, or posterior.")
        elif type_marginals == "posterior" and (U_misfit is None or Lambda_misfit is None):
            raise ValueError("pointwise_marginals: For posterior sampling, specify U_misfit and Lambda_misfit from REVD.")

        if N_samples <= 0:

            raise ValueError("pointwise_marginals: N_samples has to be postive.")

        elif N_samples == 1:

            if type_marginals == "prior_C":
                ds_subset_mean_samples = self.sample_prior_C()
            elif type_marginals == "prior":
                ds_subset_mean_samples = self.sample_prior()
            elif type_marginals == "posterior":
                ds_subset_mean_samples = self.sample_posterior(U_misfit, Lambda_misfit)

            return ds_subset_mean_samples, xr.zeros_like(ds_subset_mean_samples)

        if type_marginals == "prior_C":
            ds_subset_sample = self.sample_prior_C()
        elif type_marginals == "prior":
            ds_subset_sample = self.sample_prior()
        elif type_marginals == "posterior":
            ds_subset_sample = self.sample_posterior(U_misfit, Lambda_misfit)

        ds_subset_mean_samples = ds_subset_sample.copy()

        ds_subset_mean_samples_squared = ds_subset_sample**2
        for var in ds_subset_sample.data_vars:
            ds_subset_mean_samples_squared[var].attrs = ds_subset_sample[var].attrs
        ds_subset_mean_samples_squared.attrs = ds_subset_sample.attrs

        ## DIAGNOSTICS
        #        list_0y = []
        #        list_1y = []
        #        list_2y = []
        #        list_3y = []
        #        list_4y = []
        #        list_5y = []
        #        list_6y = []
        #        list_7y = []
        #        list_8y = []
        #        list_9y = []
        #        list_10y = []
        #        list_0x = []
        #        list_1x = []
        #        list_2x = []
        #        list_3x = []
        #        list_4x = []
        #        list_5x = []
        #        list_6x = []
        #        list_7x = []
        #        list_8x = []
        #        list_9x = []
        #        list_10x = []
        #
        #        print_idy = 35
        #        print_idx = 20
        #        print_varname = "xx_c_slide_init"
        #
        #        list_0y.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx])
        #        list_1y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 1, print_idx])
        #        list_2y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 2, print_idx])
        #        list_3y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 3, print_idx])
        #        list_4y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 4, print_idx])
        #        list_5y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 5, print_idx])
        #        list_6y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 6, print_idx])
        #        list_7y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 7, print_idx])
        #        list_8y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 8, print_idx])
        #        list_9y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 9, print_idx])
        #        list_10y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 10, print_idx])
        #        list_0x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx])
        #        list_1x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 1])
        #        list_2x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 2])
        #        list_3x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 3])
        #        list_4x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 4])
        #        list_5x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 5])
        #        list_6x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 6])
        #        list_7x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 7])
        #        list_8x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 8])
        #        list_9x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 9])
        #        list_10x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 10])

        for i in range(N_samples-1):

            if type_marginals == "prior_C":
                ds_subset_sample = self.sample_prior_C()
            if type_marginals == "prior":
                ds_subset_sample = self.sample_prior()
            elif type_marginals == "posterior":
                ds_subset_sample = self.sample_posterior(U_misfit, Lambda_misfit)

            ## DIAGNOSTICS
            #            list_0y.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx])
            #            list_1y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 1, print_idx])
            #            list_2y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 2, print_idx])
            #            list_3y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 3, print_idx])
            #            list_4y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 4, print_idx])
            #            list_5y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 5, print_idx])
            #            list_6y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 6, print_idx])
            #            list_7y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 7, print_idx])
            #            list_8y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 8, print_idx])
            #            list_9y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 9, print_idx])
            #            list_10y.append(ds_subset_sample[print_varname + "d"].data[print_idy + 10, print_idx])
            #            list_0x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx])
            #            list_1x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 1])
            #            list_2x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 2])
            #            list_3x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 3])
            #            list_4x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 4])
            #            list_5x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 5])
            #            list_6x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 6])
            #            list_7x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 7])
            #            list_8x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 8])
            #            list_9x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 9])
            #            list_10x.append(ds_subset_sample[print_varname + "d"].data[print_idy, print_idx + 10])

            ds_subset_mean_samples = self.linear_sum([ds_subset_mean_samples, ds_subset_sample],
                                                    [1.0, 1.0], ["tlm", "tlm"])

            ds_subset_sample_squared = ds_subset_sample**2
            for var in ds_subset_sample.data_vars:
                ds_subset_sample_squared[var].attrs = ds_subset_sample[var].attrs
            ds_subset_sample_squared.attrs = ds_subset_sample.attrs

            ds_subset_mean_samples_squared = self.linear_sum([ds_subset_mean_samples_squared, ds_subset_sample_squared],
                                                            [1.0, 1.0], ["tlm", "tlm"])

        ## DIAGNOSTICS
        #        print("Pearson correlation y-direction 40 kms: ", np.corrcoef(list_0y, list_1y)[0, 1])
        #        print("Pearson correlation y-direction 80 kms: ", np.corrcoef(list_0y, list_2y)[0, 1])
        #        print("Pearson correlation y-direction 120 kms: ", np.corrcoef(list_0y, list_3y)[0, 1])
        #        print("Pearson correlation y-direction 160 kms: ", np.corrcoef(list_0y, list_4y)[0, 1])
        #        print("Pearson correlation y-direction 200 kms: ", np.corrcoef(list_0y, list_5y)[0, 1])
        #        print("Pearson correlation y-direction 240 kms: ", np.corrcoef(list_0y, list_6y)[0, 1])
        #        print("Pearson correlation y-direction 280 kms: ", np.corrcoef(list_0y, list_7y)[0, 1])
        #        print("Pearson correlation y-direction 320 kms: ", np.corrcoef(list_0y, list_8y)[0, 1])
        #        print("Pearson correlation y-direction 360 kms: ", np.corrcoef(list_0y, list_9y)[0, 1])
        #        print("Pearson correlation y-direction 400 kms: ", np.corrcoef(list_0y, list_10y)[0, 1])
        #        print("Pearson correlation x-direction 40 kms: ", np.corrcoef(list_0x, list_1x)[0, 1])
        #        print("Pearson correlation x-direction 80 kms: ", np.corrcoef(list_0x, list_2x)[0, 1])
        #        print("Pearson correlation x-direction 120 kms: ", np.corrcoef(list_0x, list_3x)[0, 1])
        #        print("Pearson correlation x-direction 160 kms: ", np.corrcoef(list_0x, list_4x)[0, 1])
        #        print("Pearson correlation x-direction 200 kms: ", np.corrcoef(list_0x, list_5x)[0, 1])
        #        print("Pearson correlation x-direction 240 kms: ", np.corrcoef(list_0x, list_6x)[0, 1])
        #        print("Pearson correlation x-direction 280 kms: ", np.corrcoef(list_0x, list_7x)[0, 1])
        #        print("Pearson correlation x-direction 320 kms: ", np.corrcoef(list_0x, list_8x)[0, 1])
        #        print("Pearson correlation x-direction 360 kms: ", np.corrcoef(list_0x, list_9x)[0, 1])
        #        print("Pearson correlation x-direction 400 kms: ", np.corrcoef(list_0x, list_10x)[0, 1])

        ds_subset_mean_samples = self.linear_sum([ds_subset_mean_samples, ds_subset_mean_samples],
                                                [1.0/N_samples, 0.0], ["tlm", "tlm"])
        ds_subset_mean_samples_squared = self.linear_sum([ds_subset_mean_samples_squared, ds_subset_mean_samples_squared],
                                                        [1.0/N_samples, 0.0], ["tlm", "tlm"])

        ds_subset_squared_mean_samples = ds_subset_mean_samples**2
        for var in ds_subset_sample.data_vars:
            ds_subset_squared_mean_samples[var].attrs = ds_subset_mean_samples[var].attrs
        ds_subset_squared_mean_samples.attrs = ds_subset_mean_samples.attrs

        ds_subset_variance_samples = self.linear_sum([ds_subset_mean_samples_squared, ds_subset_squared_mean_samples],
                                                    [1.0, -1.0], ["tlm", "tlm"])

        ds_subset_std_samples = ds_subset_variance_samples**0.5
        for var in ds_subset_variance_samples.data_vars:
            ds_subset_std_samples[var].attrs = ds_subset_variance_samples[var].attrs
        ds_subset_std_samples.attrs = ds_subset_variance_samples.attrs

        return ds_subset_mean_samples, ds_subset_std_samples

    @beartype
    def l_bfgs(self,
               MAX_ITERS: int,
               num_pairs_lbfgs: int,
               init_alpha: float = 1.0,
               min_alpha_tol: float = 1.e-10,
               c1: float = 1.e-4) -> Any:

        if self.dirpath_store_states is not None:

            if not os.path.isdir(self.dirpath_store_states):
                self.make_dir(self.dirpath_store_states)
            if os.path.isdir(self.dirpath_store_states + "/" + "l_bfgs"):
                self.remove_dir(self.dirpath_store_states + "/" + "l_bfgs")

            self.make_dir(self.dirpath_store_states + "/" + "l_bfgs")

            log_file = self.dirpath_store_states + "/l_bfgs/" + "l_bfgs.log"
            with open(log_file, "a") as f:
                f.write(f"Iteration 0: Cost = {self.ds_subset_costs['fc'].data[0]:.6f}, Misfit Cost = {self.ds_subset_costs['fc_data'].data[0]:.6f}, Regularization Cost = {self.ds_subset_costs['fc_reg'].data[0]:.6f}\n")

            self.ds_subset_params.to_netcdf(self.dict_ad_inp_nc_files["nodiff"], self.dirpath_store_states + "/l_bfgs/" + "state_LBFGS_iter_0.nc")
            self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dirpath_store_states + "/l_bfgs/" + "state_LBFGS_iter_0_fields.nc")

        print("---------------------------------------------------------------------------------------------------------------")
        print(f"Initial fc = {self.ds_subset_costs['fc'].data[0]}, fc_data = {self.ds_subset_costs['fc_data'].data[0]}, fc_reg = {self.ds_subset_costs['fc_reg'].data[0]}")
        print("---------------------------------------------------------------------------------------------------------------")

        m = num_pairs_lbfgs

        list_ds_subset_s = []
        list_ds_subset_y = []

        ds_subset_params_new = self.eval_params()
        ds_subset_gradient_new = self.eval_gradient()

        for k in range(MAX_ITERS):

            ds_subset_params_old = ds_subset_params_new.copy()
            ds_subset_gradient_old = ds_subset_gradient_new.copy()

            ds_subset_q = self.linear_sum([ds_subset_gradient_old, ds_subset_gradient_old], 
                                              [0.0, -1.0], ["adj", "adj"])

            idx_lower_limit = max(k - m, 0)
            list_rhos = []
            list_alphas = []
            
            for i in range(k - 1, idx_lower_limit - 1, -1):

                rho = 1 / self.l2_inner_product([list_ds_subset_y[i-idx_lower_limit], list_ds_subset_s[i-idx_lower_limit]], ["adj", "nodiff"])
                list_rhos = [rho] + list_rhos

                alpha = rho * self.l2_inner_product([list_ds_subset_s[i-idx_lower_limit], ds_subset_q], ["nodiff", "adj"])
                list_alphas = [alpha] + list_alphas

                ds_subset_q = self.linear_sum([ds_subset_q, list_ds_subset_y[i-idx_lower_limit]], [1.0, -alpha], ["adj", "adj"])

            if list_ds_subset_s and list_ds_subset_y:
                gamma_k = self.l2_inner_product([list_ds_subset_y[0], list_ds_subset_s[0]], ["adj", "nodiff"]) / self.l2_inner_product([list_ds_subset_y[0], list_ds_subset_y[0]], ["adj", "adj"])
            else:
                gamma_k = 1.0

            if gamma_k <= 0:
                print("l_bfgs: Invalid gamma encountered.")

            ds_subset_p = self.linear_sum([ds_subset_q, ds_subset_q], [0.0, gamma_k], ["adj", "adj"])

            for i in range(idx_lower_limit, k):

                beta = list_rhos[k-i-1] * self.l2_inner_product([list_ds_subset_y[i-idx_lower_limit], ds_subset_p], ["adj", "adj"])
                ds_subset_p = self.linear_sum([ds_subset_p, list_ds_subset_s[i-idx_lower_limit]], [1.0, list_alphas[k-i-1] - beta], ["adj", "nodiff"])

            alpha_line_search, self.ds_subset_costs = self.line_search(ds_subset_gradient_old,
                                                                       ds_subset_p,
                                                                       init_alpha, min_alpha_tol, c1)

            print("---------------------------------------------------------------------------------------------------------------")
            print(f"Iter {k+1}, fc = {self.ds_subset_costs['fc'].data[0]}, fc_data = {self.ds_subset_costs['fc_data'].data[0]}, fc_reg = {self.ds_subset_costs['fc_reg'].data[0]}")
            print("---------------------------------------------------------------------------------------------------------------")

            ds_subset_params_new = self.linear_sum([ds_subset_params_old, ds_subset_p], 
                                                   [1.0, alpha_line_search], ["nodiff", "adj"])
            self.write_params(ds_subset_params_new)
            self.ds_subset_params = ds_subset_params_new.copy()

            self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dict_ad_inp_nc_files["adj"])

            if self.dirpath_store_states is not None:

                with open(log_file, "a") as f:
                    f.write(f"Iteration {k+1}: Cost = {self.ds_subset_costs['fc'].data[0]:.6f}, Misfit Cost = {self.ds_subset_costs['fc_data'].data[0]:.6f}, Regularization Cost = {self.ds_subset_costs['fc_reg'].data[0]:.6f}\n")

                self.ds_subset_params.to_netcdf(self.dict_ad_inp_nc_files["nodiff"], self.dirpath_store_states + "/l_bfgs/" + "state_LBFGS_iter_{k+1}.nc")
                self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dirpath_store_states + "/l_bfgs/" + f"state_LBFGS_iter_{k+1}_fields.nc")

            ds_subset_gradient_new = self.eval_gradient()

            ds_subset_s_k = self.linear_sum([ds_subset_params_new, ds_subset_params_old], [1.0, -1.0], ["nodiff", "nodiff"])
            ds_subset_y_k = self.linear_sum([ds_subset_gradient_new, ds_subset_gradient_old], [1.0, -1.0], ["adj", "adj"])

            if len(list_ds_subset_s) < m and len(list_ds_subset_y) < m and len(list_ds_subset_s) == len(list_ds_subset_y):
                list_ds_subset_s.append(ds_subset_s_k)
                list_ds_subset_y.append(ds_subset_y_k)
            elif len(list_ds_subset_s) == m and len(list_ds_subset_y) == m:
                list_ds_subset_s.pop(0)
                list_ds_subset_y.pop(0)
                list_ds_subset_s.append(ds_subset_s_k)
                list_ds_subset_y.append(ds_subset_y_k)
            else:
                raise ValueError("l_bfgs: Some issue in lists that store the s and y vectors.")

        return self.ds_subset_params

    @beartype
    def open_xr_ds(self, path: str, bool_assign_coords: bool = True) -> Any:

        ds = xr.open_dataset(path)
        ds.load()

        if bool_assign_coords:
            ds = ds.assign_coords({dim: self.ds_subset_params[dim] for dim in self.ds_subset_params.dims})

        return ds

    @beartype
    def linear_sum(self, list_ds_subset: List[Any], list_alphas: List[float], list_types: List[str]) -> Any:

        self.ds_subset_compatibility_check(list_ds_subset, list_types)

        if len(list_alphas) != 2:
            raise ValueError("linear_sum: Only works for two ds_subset, alphas, and types.")

        ds_out = list_ds_subset[0].copy()
    
        for var_0 in list_ds_subset[0]:
    
            if list_types[0] != "nodiff":
                basic_str_0 = var_0[:-1]
            else:
                basic_str_0 = var_0
    
            for var_1 in list_ds_subset[1]:
    
                if list_types[1] != "nodiff":
                    basic_str_1 = var_1[:-1]
                else:
                    basic_str_1 = var_1
    
                if basic_str_0 == basic_str_1:
                    if list_ds_subset[0][var_0].data.shape != list_ds_subset[1][var_1].data.shape:
                        raise ValueError(f"linear_sum: {var_0}, {var_1} do not have the same shape in the two ds_subset.")
    
                    if not self.list_fields_to_ignore or (self.list_fields_to_ignore and basic_str_0 not in self.list_fields_to_ignore):
                        ds_out[var_0].data = list_alphas[0]*list_ds_subset[0][var_0].data.copy() + list_alphas[1]*list_ds_subset[1][var_1].data.copy()
    
        return ds_out

    @beartype
    def l2_inner_product(self, list_ds_subset: List[Any], list_types: List[str]) -> float:

        self.ds_subset_compatibility_check(list_ds_subset, list_types)

        inner_product = 0.0

        for var_0 in list_ds_subset[0]:

            if list_types[0] != "nodiff":
                basic_str_0 = var_0[:-1]
            else:
                basic_str_0 = var_0
    
            for var_1 in list_ds_subset[1]:
    
                if list_types[1] != "nodiff":
                    basic_str_1 = var_1[:-1]
                else:
                    basic_str_1 = var_1
    
                if basic_str_0 == basic_str_1:
                    if list_ds_subset[0][var_0].data.shape != list_ds_subset[1][var_1].data.shape:
                        raise ValueError(f"l2_inner_product: {var_0}, {var_1} do not have the same shape in the two ds_subset.")

                    if not self.list_fields_to_ignore or (self.list_fields_to_ignore and basic_str_0 not in self.list_fields_to_ignore):
                        inner_product = inner_product + np.sum(list_ds_subset[0][var_0].data*list_ds_subset[1][var_1].data)
    
        return inner_product

    @beartype
    def exch_tlm_adj_nc(self, ds_inp_subset_tlmhessaction_or_adj_action: Any, og_type: str) -> Any:

        if og_type == "tlmhessaction":
            new_type = "adjhessaction"
            ad_key_new = "adj_action"
        elif og_type == "adj":
            new_type = "tlm"
            ad_key_new = "tlm_action"
        else:
            raise ValueError(f"exch_tlm_adj_nc: Invalid og_type {og_type}.")

        dict_coords = self.dict_params_coords.copy()
        dict_fields_vals = {}

        for var in self.ds_subset_params:
            if self.dict_params_fields_or_scalars and self.dict_params_fields_or_scalars[var] == "scalar":
                dict_fields_vals[var] = self.ds_subset_params[var].data[0]
            else:
                dict_fields_vals[var] = self.ds_subset_params[var].data.copy()

        if og_type == "tlmhessaction":

            dict_fields_num_dims = self.dict_params_fields_num_dims.copy()
            dict_attrs_type = self.dict_params_attrs_type.copy()
            dict_fields_or_scalars = self.dict_params_fields_or_scalars.copy()
 
            for var in ds_inp_subset_tlmhessaction_or_adj_action:
                dict_fields_num_dims[var[:-1] + "b"] = str(len(ds_inp_subset_tlmhessaction_or_adj_action[var].data.shape)) + "D"
                dict_attrs_type[var[:-1] + "b"] = new_type
                dict_fields_vals[var[:-1] + "b"] = ds_inp_subset_tlmhessaction_or_adj_action[var].data.copy()
                dict_fields_or_scalars[var[:-1] + "b"] = "field"

        elif og_type == "adj":

            dict_fields_num_dims = self.dict_tlm_action_fields_num_dims.copy()
            dict_attrs_type = self.dict_tlm_action_attrs_type.copy()
            dict_fields_or_scalars = self.dict_tlm_action_fields_or_scalars.copy()

            for var in ds_inp_subset_tlmhessaction_or_adj_action:
                if dict_fields_or_scalars[var[:-1] + "d"] == "scalar":
                    dict_fields_vals[var[:-1] + "d"] = ds_inp_subset_tlmhessaction_or_adj_action[var].data.flat[0].copy()
                elif dict_fields_or_scalars[var[:-1] + "d"] == "field":
                    dict_fields_vals[var[:-1] + "d"] = ds_inp_subset_tlmhessaction_or_adj_action[var].data.copy()
                else:
                    raise ValueError(f"exch_tlm_adj_nc: var {var[:-1] + 'd'} should be either a scalar or a field.")

            if self.list_fields_to_ignore:

                for var in self.list_fields_to_ignore:
                    if dict_fields_or_scalars[var + "d"] == "scalar":
                        dict_fields_vals[var + "d"] = 0.0
                    elif dict_fields_or_scalars[var + "d"] == "field":
                        dict_fields_vals[var + "d"] = self.ds_subset_params[var].data*0.0
                    else:
                        raise ValueError(f"exch_tlm_adj_nc: var {var[:-1] + 'd'} should be either a scalar or a field.")

        else:
            raise ValueError("exch_tlm_adj_nc: This should be impossible.")

        ds_inp_fields_tlm_or_adj_action = self.create_ad_nodiff_or_adj_input_nc(dict_fields_vals = dict_fields_vals,
                                                                                dict_fields_num_dims = dict_fields_num_dims,
                                                                                dict_coords = dict_coords,
                                                                                dict_attrs_type = dict_attrs_type,
                                                                                dict_fields_or_scalars = dict_fields_or_scalars,
                                                                                ad_key = ad_key_new, filename = None)

        if og_type == "tlmhessaction":

            # This is fine since none of the observables are scalars 
            return self.subset_of_ds(ds_inp_fields_tlm_or_adj_action, "type", new_type)

        elif og_type == "adj":
    
            ds_subset_tlm = self.subset_of_ds(ds_inp_fields_tlm_or_adj_action, "type", new_type) 

            for var in ds_subset_tlm:
                if dict_fields_or_scalars is not None and dict_fields_or_scalars[var] == "scalar":
                    ds_subset_tlm[var] = xr.DataArray([ds_subset_tlm[var].data.flat[0]], 
                                                       dims=["scalar"], attrs=ds_subset_tlm[var].attrs)

            return ds_subset_tlm

    @beartype
    def ds_subset_compatibility_check(self, list_ds_subset: List[Any], list_types: List[str]) -> None:
    
        if len(list_ds_subset) != 2 or len(list_types) != 2:
            return ValueError("ds_subset_compatibility_check: Only works for two ds_subset at a time.")

        for var in list_ds_subset[0]:
            if "type" not in list_ds_subset[0][var].attrs:
                raise ValueError(f"ds_subset_compatibility_check: Attribute 'type' is missing for variable {var} in first ds_subset.")
            elif list_ds_subset[0][var].attrs["type"] != list_types[0]:
                raise ValueError(f"ds_subset_compatibility_check: Type of {var} in first ds_subset is not what is expected i.e. {list_types[0]}.")
            else:
                pass

        for var in list_ds_subset[1]:
            if "type" not in list_ds_subset[1][var].attrs:
                raise ValueError(f"ds_subset_compatibility_check: Attribute 'type' is missing for variable {var} in second ds_subset.")
            elif list_ds_subset[1][var].attrs["type"] != list_types[1]:
                raise ValueError(f"ds_subset_compatibility_check: Type of {var} in second ds_subset is not what is expected i.e. {list_types[1]}.")
            else:
                pass
                
        list_suffixes = []

        for type_var in list_types:
            if type_var == "nodiff":
                list_suffixes.append("")
            elif type_var == "adj" or type_var == "adjhessaction":
                list_suffixes.append("b")
            elif type_var == "tlm" or type_var == "tlmhessaction":
                list_suffixes.append("d")
            else:
                return ValueError(f"ds_subset_compatibility_check: {type_var} is not a valid type for this function.")
    
        for var_0 in list_ds_subset[0]:
            if list_suffixes[0] != "":
                basic_str = var_0[:-1]
                var_1 = basic_str + list_suffixes[1]
            else:
                basic_str = var_0
                var_1 = basic_str + list_suffixes[1]

            if var_1 not in list_ds_subset[1]:
                if self.list_fields_to_ignore and basic_str in self.list_fields_to_ignore:
                    pass
                else:
                    raise ValueError(f"ds_subset_compatibility_check: {var_1} not present in second ds_subset when {var_0} is present in first ds_subset.")
    
        for var_1 in list_ds_subset[1]:
            if list_suffixes[1] != "":
                basic_str = var_1[:-1]
                var_0 = basic_str + list_suffixes[0]
            else:
                basic_str = var_1
                var_0 = basic_str + list_suffixes[0]

            if var_0 not in list_ds_subset[0]:
                if self.list_fields_to_ignore and basic_str in self.list_fields_to_ignore:
                    pass
                else:
                    raise ValueError(f"ds_subset_compatibility_check: {var_0} not present in first ds_subset when {var_1} is present in second ds_subset.")
    
        return None

    @staticmethod
    @beartype
    def has_exact_keys(d: Dict[str, str], required_keys: List[str]) -> bool:
        return set(d.keys()) == set(required_keys)

    @staticmethod
    @beartype
    def create_dict_tlm_action(dict_params: Union[Dict[str, str], Dict[str, Float[np.ndarray, "dim"]]],
                               value_new: Optional[Union[str, Float[np.ndarray, "dim"]]]  = None) -> Union[Dict[str, str], Dict[str, Float[np.ndarray, "dim"]]]:

        dict_tlm_action = {}
        for key, value in dict_params.items():
            dict_tlm_action[key] = value
            if value_new is None:
                dict_tlm_action[key + "d"] = value
            else:
                dict_tlm_action[key + "d"] = value_new

        return dict_tlm_action

    @staticmethod
    @beartype
    def subset_of_ds(ds: Any, attr_key: str, attr_value: str) -> Any:

        ds = ds.copy()
        subset_vars = []

        for var in ds:
            if attr_key not in ds[var].attrs:
                raise ValueError(f"subset_of_ds: Attribute '{attr_key}' is missing for variable {var}")
            else:
                if ds[var].attrs[attr_key] == attr_value:
                    subset_vars.append(var)

        ds_subset = ds[subset_vars]

        return ds_subset

    @staticmethod
    @beartype
    def flattened_vector(ds_subset: Any) -> Tuple[int | np.integer, Float[np.ndarray, "dim_m"]]:
        m = sum(np.prod(var.shape) for var in ds_subset.data_vars.values())
        flattened_vector = np.concatenate([var.values.ravel() for var in ds_subset.data_vars.values()])
        assert m == flattened_vector.shape[0]

        return m, flattened_vector

    @staticmethod
    @beartype
    def construct_ds(flattened_vector: Float[np.ndarray, "dim"], original_ds: Any) -> Any:

        ds_reconstructed_data = {}
        start = 0
        for var_name, var_data in original_ds.data_vars.items():

            shape = var_data.shape
            size = np.prod(shape)

            reshaped_data = flattened_vector[start : start + size].reshape(shape)

            ds_reconstructed_data[var_name] = (var_data.dims, reshaped_data)

            start += size

        ds_reconstructed = xr.Dataset(ds_reconstructed_data, coords=original_ds.coords)

        for var in ds_reconstructed:
            ds_reconstructed[var].attrs["type"] = original_ds[var].attrs["type"]

        return ds_reconstructed

    @staticmethod
    @beartype
    def copy_dir(old_path: str, new_path: str) -> None:
    
        subprocess.run(
            f"cp -r {old_path} {new_path}",
            shell=True)
    
        return None

    @staticmethod
    @beartype
    def move_dir(old_path: str, new_path: str) -> None:
    
        subprocess.run(
            f"mv {old_path} {new_path}",
            shell=True)
    
        return None

    @staticmethod
    @beartype
    def make_dir(path: str) -> None:

        subprocess.run(
            f"mkdir {path}",
            shell=True)
    
        return None

    @staticmethod
    @beartype
    def remove_dir(path: str) -> None:

        subprocess.run(
            f"rm -rf {path}",
            shell=True)
    
        return None

    @staticmethod
    @beartype
    def has_complex_parts(arr: Union[np.ndarray, float, complex],
                          tol: float = np.finfo(float).eps) -> bool:
        return bool(np.any(np.abs(np.imag(arr)) > tol))

