import numpy as np
import xarray as xr
import subprocess

from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Tuple, Union
from jaxtyping import Float

import os

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
                 dict_params_fields_num_dims: Dict[str, str],
                 dict_params_coords: Dict[str, Float[np.ndarray, "dim"]],
                 dict_params_attrs_type: Dict[str, str],
                 dict_params_fields_or_scalars: Dict[str, str],
                 dict_masks_observables: Dict[str, Optional[Union[Float[np.ndarray, "dimz dimy dimx"], 
                                                                  Float[np.ndarray, "dimy dimx"]]]],
                 prior_alpha: float,
                 dict_prior_sigmas: Dict[str, Optional[float]],
                 dict_prior_gammas: Dict[str, Optional[float]],
                 dict_prior_deltas: Dict[str, Optional[float]],
                 list_fields_to_ignore: Optional[List[str]] = None,
                 bool_surfvel_cost: bool = False,
                 filename_vx_vy_s_g: Optional[str] = None,
                 dirpath_store_states: Optional[str] = None) -> None:

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

        if dict_og_params_fields_vals.keys() != dict_params_fields_num_dims.keys() != dict_params_coords.keys() != dict_params_attrs_type.keys() != dict_params_fields_or_scalars.keys():
            raise ValueError("DataAssimilation: Inconsistent keys for OG state.")
 
        self.dict_og_params_fields_vals = dict_og_params_fields_vals
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
        self.JMAX    = dict_params_coords["y"].shape[0] - 1
        self.IMAX    = dict_params_coords["x"].shape[0] - 1
        self.delta_x = dict_params_coords["x"][1] - dict_params_coords["x"][0]
        self.delta_y = dict_params_coords["y"][1] - dict_params_coords["y"][0]

        self.dict_masks_observables = dict_masks_observables

        if dict_og_params_fields_vals.keys() != dict_prior_sigmas.keys() != dict_prior_gammas.keys() != dict_prior_deltas.keys():
            raise ValueError("DataAssimilation: Inconsistent keys for prior dicts.")

        self.prior_alpha = prior_alpha
        self.dict_prior_sigmas = dict_prior_sigmas
        self.dict_prior_gammas = dict_prior_gammas
        self.dict_prior_deltas = dict_prior_deltas

        self.list_fields_to_ignore = list_fields_to_ignore

        if bool_surfvel_cost and filename_vx_vy_s_g is None:
            raise ValueError("DataAssimilation: File to read terminal surface velocity field values not specified.")
        
        if not bool_surfvel_cost and filename_vx_vy_s_g is not None:
            raise ValueError("DataAssimilation: File to read terminal surface velocity field values specified even when not needed.")

        self.bool_surfvel_cost = bool_surfvel_cost
        self.filename_vx_vy_s_g = filename_vx_vy_s_g
        self.dirpath_store_states = dirpath_store_states

        _ = self.create_ad_nodiff_or_adj_input_nc(dict_og_params_fields_vals, dict_params_fields_num_dims,
                                                  dict_params_coords, dict_params_attrs_type, dict_params_fields_or_scalars,
                                                  "nodiff")
        _ = self.create_ad_nodiff_or_adj_input_nc(dict_og_params_fields_vals, dict_params_fields_num_dims,
                                                  dict_params_coords, dict_params_attrs_type, dict_params_fields_or_scalars,
                                                  "adj")
        self.fc = self.eval_cost()
        self.ds_subset_params = self.eval_params()

    @beartype
    def create_ad_nodiff_or_adj_input_nc(self,
                                         dict_fields_vals: Dict[str, Union[Float[np.ndarray, "dimz dimy dimx"],
                                                                           Float[np.ndarray, "dimy dimx"],
                                                                           float]],
                                         dict_fields_num_dims: Dict[str, str],
                                         dict_coords: Dict[str, Float[np.ndarray, "dim"]],
                                         dict_attrs_type: Dict[str, str],
                                         dict_fields_or_scalars: Dict[str, str],
                                         ad_key: str) -> Any:
        
        if dict_fields_vals.keys() != dict_fields_num_dims.keys() != dict_coords.keys() != dict_attrs_type.keys() != dict_fields_or_scalars:
            raise ValueError("create_ad_nodiff_or_adj_input_nc: Inconsistent keys.")
        
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
                    raise ValueError(f"create_ad_nodiff_or_adj_input_nc: Issue with {field}; Only 2D or 2DT or 3D fields accepted.")
                
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
                    raise ValueError(f"create_ad_nodiff_or_adj_input_nc: Issue with {field}; Only 2D or 2DT or 3D fields accepted.")
                
            else:
                raise TypeError(f"create_ad_nodiff_or_adj_input_nc: Issue with {field}; The type doesn't seem to be either a scalar or a numpy array.")
            
            ds_inp_fields[field] = da_field
            if field in dict_attrs_type:
                ds_inp_fields[field].attrs["type"] = dict_attrs_type[field]

        # Some weird permission denied error if this file is not removed first.
        self.remove_dir(self.dict_ad_inp_nc_files[ad_key])  
        ds_inp_fields.to_netcdf(self.dict_ad_inp_nc_files[ad_key])

        # Returns ds_inp_fields where even scalars are expressed as fields
        return ds_inp_fields

    @beartype
    def run_exec(self, ad_key: str) -> Any:

        if not os.path.isfile(self.dict_ad_inp_nc_files[ad_key]):
            raise ValueError(f"run_exec: AD input file {self.dict_ad_inp_nc_files[ad_key]} is missing.")

        self.remove_dir(self.dict_sico_out_folders[ad_key])
        self.remove_dir(self.dict_ad_out_nc_files[ad_key])

        with open(self.dict_ad_log_files[ad_key], "w") as log:
            subprocess.run(
                f"time {self.dict_ad_exec_cmds[ad_key]}",
                cwd=self.src_dir,
                shell=True,
                stdout=log,
                stderr=subprocess.STDOUT)

        ds_out_fields = xr.open_dataset(self.dict_ad_out_nc_files[ad_key])

        # Returns ds_out_fields where even scalars are expressed as fields since we are reading simulation output
        return ds_out_fields

    @beartype
    def get_vx_vy_s(self, sico_out_nc_file: str) -> Tuple[Float[np.ndarray, "dimy dimx"], Float[np.ndarray, "dimy dimx"]]:

        _ = self.run_exec(ad_key = "nodiff")

        path_sico_out_nc = self.dict_sico_out_folders["nodiff"] + "/" + sico_out_nc_file
        if not os.path.isfile(path_sico_out_nc):
            raise ValueError(f"get_vx_vy_s: AD input file {self.dict_sico_out_folders['nodiff']}/{sico_out_nc_file} is missing.")

        ds_out_fields = xr.open_dataset(path_sico_out_nc)

        if "vx_s_g" not in ds_out_fields or "vy_s_g" not in ds_out_fields:
            raise ValueError("get_vx_vy_s: One or both of vx_s_g or vy_s_g missing!")
    
        vx_s_g = ds_out_fields["vx_s_g"].data
        vy_s_g = ds_out_fields["vy_s_g"].data
    
        return vx_s_g, vy_s_g

    @beartype
    def eval_cost(self) -> Union[Float[np.ndarray, "dim"], float]:

        ds_out_fields_nodiff = self.run_exec(ad_key = "nodiff")
        fc = ds_out_fields_nodiff['fc'].data[0]
    
        return fc

    @beartype
    def eval_params(self) -> Any:

        # Have to evaluate the out nc for coords value consistency when merging with other output datasets.
        ds_out_fields_nodiff = xr.open_dataset(self.dict_ad_out_nc_files["nodiff"])
        ds_subset_params = self.subset_of_ds(ds_out_fields_nodiff, attr_key = "type", attr_value = "nodiff")

        for var in self.dict_params_fields_or_scalars:
    
            if self.dict_params_fields_or_scalars and self.dict_params_fields_or_scalars[var] == "scalar":
        
                if var in ds_subset_params:
                    ds_subset_params[var] = xr.DataArray([ds_subset_params[var].data.flat[0]], 
                                                          dims=["scalar"], attrs=ds_subset_params[var].attrs)
                else:
                    raise ValueError(f"eval_params: {var} not present in ds_subset_params!")

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
                                                  ad_key = "nodiff")
 
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
                    c1: float = 1.e-4) -> Tuple[float, float]:
    
        alpha = init_alpha
        ds_subset_params_orig = self.ds_subset_params.copy()
        fc = self.fc

        while True:
            
            try:                
                ds_subset_params_new = self.linear_sum([ds_subset_params_orig, ds_subset_descent_dir], 
                                                       [1.0, alpha], ["nodiff", "adj"])
                self.write_params(ds_subset_params_new)
        
                fc_new = self.eval_cost()
                pTg = self.l2_inner_product([ds_subset_descent_dir, ds_subset_gradient], ["adj", "adj"])
                ratio = (fc_new - fc)/(alpha*pTg)

            except:
                print("Too big step size probably crashed the simulation.")
                self.write_params(ds_subset_params_orig)
                ratio = 0.0

            if alpha <= min_alpha_tol:
                print(f"Minimum tolerable step size alpha reached.")
                print(f"Step size alpha = {alpha}")
                return alpha, fc_new

            if ratio >= c1:
                print(f"Step size alpha = {alpha}")
                return alpha, fc_new
    
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
                f.write(f"Iteration 0: Cost = {self.fc:.6f}\n")

            self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dirpath_store_states + "/gradient_descent/" + "state_GD_iter_0.nc")

        print("-------------------------------------")
        print(f"iter 0, fc = {self.fc}")
        print("-------------------------------------")

        for i in range(MAX_ITERS):

            ds_subset_params = self.ds_subset_params.copy()
            ds_subset_gradient = self.eval_gradient()
            norm_gradient = self.l2_inner_product([ds_subset_gradient, ds_subset_gradient], ["adj", "adj"])**0.5
            
            if MIN_GRAD_NORM_TOL is not None and norm_gradient <= MIN_GRAD_NORM_TOL:
                print("Minimum for gradient norm reached.")
                break

            ds_subset_descent_dir = self.linear_sum([ds_subset_gradient, ds_subset_gradient], 
                                                    [0.0, -1.0], ["adj", "adj"])

            alpha, self.fc = self.line_search(ds_subset_gradient,
                                              ds_subset_descent_dir,
                                              init_alpha, min_alpha_tol, c1)

            ds_subset_params_new = self.linear_sum([ds_subset_params, ds_subset_gradient], 
                                                   [1.0, -alpha], ["nodiff", "adj"])
            self.write_params(ds_subset_params_new)
            self.ds_subset_params = ds_subset_params_new.copy()

            self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dict_ad_inp_nc_files["adj"])

            if self.dirpath_store_states is not None:

                with open(log_file, "a") as f:
                    f.write(f"Iteration {i+1}: Cost = {self.fc:.6f}\n")

                self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dirpath_store_states + "/gradient_descent/" + f"state_GD_iter_{i+1}.nc")

            print("-------------------------------------")
            print(f"iter {i+1}, fc = {self.fc}")
            print("-------------------------------------")

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
                                                                         "tlm_action")

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
        ds_inp_fields_adj_action = xr.open_dataset(self.dict_ad_inp_nc_files["adj_action"])

        if self.bool_surfvel_cost:

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
    def eval_sqrt_prior_cov_inv_action(self) -> Any:

        ds_inp_fields_tlm = xr.open_dataset(self.dict_ad_inp_nc_files["tlm_action"])
        ds_subset_fields_tlm = self.subset_of_ds(ds_inp_fields_tlm, "type", "tlm")

        for var in ds_subset_fields_tlm:

            basic_str = var[:-1]

            if self.dict_params_fields_or_scalars[basic_str] == "scalar":

                if not isinstance(self.dict_prior_sigmas[basic_str], float):
                    raise ValueError("eval_sqrt_prior_cov_inv_action: sigma for scalar field should also be scalar.")

                ds_subset_fields_tlm[var].data = ds_subset_fields_tlm[var].data / self.dict_prior_sigmas[basic_str]

            elif self.dict_params_fields_or_scalars[basic_str] == "field" and self.dict_params_fields_num_dims[basic_str] == "2D":

                gamma = self.dict_prior_gammas[basic_str]
                delta = self.dict_prior_gammas[basic_str]
                delta_x = self.delta_x
                delta_y = self.delta_y
                IMAX = self.IMAX
                JMAX = self.JMAX

                field = ds_subset_fields_tlm[var].data.copy()
                field_new = delta*field.copy()

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

                ds_subset_fields_tlm[var].data = field_new.copy()

            elif self.dict_params_fields_or_scalars[basic_str] == "field" and self.dict_params_fields_num_dims[basic_str] == "3D":

                if not isinstance(self.dict_prior_sigmas[basic_str], np.ndarray) or ds_subset_fields_tlm[var].data.shape != self.dict_prior_sigmas[basic_str].shape:
                    raise ValueError("eval_sqrt_prior_cov_inv_action: sigma for 3D field should also be a 3D np.ndarray with the correct shape.")

                ds_subset_fields_tlm[var].data = ds_subset_fields_tlm[var].data / self.dict_prior_sigmas[basic_str]

            else:
                raise ValueError("eval_sqrt_prior_cov_inv_action: Issue with var. Prior action only works for scalar, or 2D and 3D fields.")

        dict_tlm_action_only_fields_vals = {}
        for var in ds_subset_fields_tlm:

            if not self.list_fields_to_ignore or (self.list_fields_to_ignore and var[:-1] not in self.list_fields_to_ignore):
                if self.dict_tlm_action_fields_or_scalars[var] == "scalar":
                    dict_tlm_action_only_fields_vals[var] = ds_subset_fields_tlm[var].data.flat[0].copy()
                else:
                    dict_tlm_action_only_fields_vals[var] = ds_subset_fields_tlm[var].data.copy()

        return self.create_ad_tlm_action_input_nc(dict_tlm_action_only_fields_vals)

    @beartype
    def eval_sqrt_prior_cov_action(self, 
                                   ad_key_adj_or_adj_action_or_tlm_action: str, 
                                   MAX_ITERS_SOR: int = 100, 
                                   OMEGA_SOR: float = 1.5) -> Any:

        if not (1.0 <= OMEGA_SOR <= 2.0):
            raise ValueError("eval_sqrt_prior_cov_action: Relaxation factor for SOR solver should be between 1 and 2.")

        if ad_key_adj_or_adj_action_or_tlm_action not in ["tlm_action", "adj", "adj_action"]:
            raise ValueError("eval_sqrt_prior_cov_inv_action: Can only act on tlm or adj or adj_action quantities.")

        if ad_key_adj_or_adj_action_or_tlm_action == "tlm_action":
            ds_fields_adj_or_adj_action_or_tlm_action = xr.open_dataset(self.dict_ad_inp_nc_files[ad_key_adj_or_adj_action_or_tlm_action])
            ad_subset_key = "tlm"
        else:
            ds_fields_adj_or_adj_action_or_tlm_action = xr.open_dataset(self.dict_ad_out_nc_files[ad_key_adj_or_adj_action_or_tlm_action])
            ad_subset_key = "adj"

        ds_subset_fields_params = self.subset_of_ds(ds_fields_adj_or_adj_action_or_tlm_action, "type", "nodiff")
        ds_subset_fields_adj_or_adj_action_or_tlm_action = self.subset_of_ds(ds_fields_adj_or_adj_action_or_tlm_action, "type", ad_subset_key)

        for var in ds_subset_fields_adj_or_adj_action_or_tlm_action:

            basic_str = var[:-1]

            if self.dict_params_fields_or_scalars[basic_str] == "scalar" and (not self.list_fields_to_ignore or (self.list_fields_to_ignore and var not in self.list_fields_to_ignore)):

                if not isinstance(self.dict_prior_sigmas[basic_str], float):
                    raise ValueError("eval_sqrt_prior_cov_action: sigma for scalar field should also be scalar.")

                ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data = ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data * self.dict_prior_sigmas[basic_str]

            elif self.dict_params_fields_or_scalars[basic_str] == "field" and self.dict_params_fields_num_dims[basic_str] == "2D" and (not self.list_fields_to_ignore or (self.list_fields_to_ignore and var not in self.list_fields_to_ignore)):

                gamma = self.dict_prior_gammas[basic_str]
                delta = self.dict_prior_gammas[basic_str]
                delta_x = self.delta_x
                delta_y = self.delta_y
                IMAX = self.IMAX
                JMAX = self.JMAX

                field = ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data.copy()

                result_old = np.copy(field)
                result = np.copy(field)

                for _ in range(MAX_ITERS_SOR):

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

                            result[j, i] = (1 - OMEGA_SOR) * result_old[j, i] + OMEGA_SOR / diagonal * bracket

                    result_old = result.copy()

                ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data = result.copy()

            elif self.dict_params_fields_or_scalars[basic_str] == "field" and self.dict_params_fields_num_dims[basic_str] == "3D" and (not self.list_fields_to_ignore or (self.list_fields_to_ignore and var not in self.list_fields_to_ignore)):

                if not isinstance(self.dict_prior_sigmas[basic_str], np.ndarray) or ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data.shape != self.dict_prior_sigmas[basic_str].shape:
                    raise ValueError("eval_sqrt_prior_cov_action: sigma for 3D field should also be a 3D np.ndarray with the correct shape.")

                ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data = ds_subset_fields_adj_or_adj_action_or_tlm_action[var].data * self.dict_prior_sigmas[basic_str]

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
    def eval_prior_preconditioned_misfit_hessian_action(self) -> Any:

        ds_subset_inp_tlm_action = self.eval_sqrt_prior_cov_action(ad_key_adj_or_adj_action_or_tlm_action = "tlm_action")
        ds_subset_misfit_hessian_action = self.eval_misfit_hessian_action()

        return self.eval_sqrt_prior_cov_action(ad_key_adj_or_adj_action_or_tlm_action = "adj_action")

    @beartype
    def eval_prior_preconditioned_hessian_action(self) -> Any:

        ds_inp_tlm_action = xr.open_dataset(self.dict_ad_inp_nc_files["tlm_action"])
        ds_subset_tlm = self.subset_of_ds(ds_inp_tlm_action, "type", "tlm")

        ds_subset_prior_precond_misfit_hess_action = self.eval_prior_preconditioned_misfit_hessian_action()

        for var in ds_subset_prior_precond_misfit_hess_action:

            basic_str = var[:-1]

            if self.dict_params_fields_or_scalars and self.dict_params_fields_or_scalars[basic_str] == "scalar":
                ds_subset_prior_precond_misfit_hess_action[var].data = ds_subset_prior_precond_misfit_hess_action[var].data + self.prior_alpha*ds_subset_tlm[basic_str + "d"].data.flat[0]
            else:
                ds_subset_prior_precond_misfit_hess_action[var].data = ds_subset_prior_precond_misfit_hess_action[var].data + self.prior_alpha*ds_subset_tlm[basic_str + "d"].data

        # Note that the final result is not written to any nc file
        return ds_subset_prior_precond_misfit_hess_action

    @beartype
    def conjugate_gradient(self, tolerance_type = "superlinear") -> Any:

        ds_subset_gradient = self.eval_gradient()
        ds_subset_gradient_hat = self.eval_sqrt_prior_cov_action(ad_key_adj_or_adj_action_or_tlm_action = "adj")

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

            v_hatT_H_hat_v_hat = self.l2_inner_product([ds_subset_v_hat, ds_subset_H_hat_v_hat], ["tlm", "adj"])
            norm_r_hat_old = self.l2_inner_product([ds_subset_r_hat, ds_subset_r_hat], ["adj", "adj"])**0.5

            if v_hatT_H_hat_v_hat < 0:

                print("conjugate_gradient: Hessian no longer positive definite!")

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
                eps_hat_TOL = min(0.5, norm_gradient)*norm_gradient_hat
            else:
                raise ValueError("conjugate_gradient: Invalid tolerance_type.")

            if norm_r_hat <= eps_hat_TOL:
                print("conjugate_gradient: Convergence.")

                p_hatT_g_hat = self.l2_inner_product([ds_subset_p_hat, ds_subset_gradient_hat], ["adj", "adj"])
                norm_p_hat = self.l2_inner_product([ds_subset_p_hat, ds_subset_p_hat], ["adj", "adj"])**0.5
                cos = p_hatT_g_hat / (norm_p_hat * norm_gradient_hat)
                angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))

                print("Angle between p_hat and g_hat in degrees: ", angle)

                return ds_subset_p_hat

            beta_hat = norm_r_hat**2 / norm_r_hat_old**2

            ds_subset_v_hat = self.linear_sum([ds_subset_v_hat, ds_subset_r_hat],
                                              [1.0, beta_hat], ["tlm", "adj"])
            
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
                              MAX_ITERS_SOR: int = 100, 
                              OMEGA_SOR: float = 1.5) -> Any:

        if self.dirpath_store_states is not None:

            if not os.path.isdir(self.dirpath_store_states):
                self.make_dir(self.dirpath_store_states)
            if os.path.isdir(self.dirpath_store_states + "/" + "inexact_gn_hessian_cg"):
                self.remove_dir(self.dirpath_store_states + "/" + "inexact_gn_hessian_cg")

            self.make_dir(self.dirpath_store_states + "/" + "inexact_gn_hessian_cg")

            log_file = self.dirpath_store_states + "/inexact_gn_hessian_cg/" + "inexact_gn_hessian_cg.log"
            with open(log_file, "a") as f:
                f.write(f"Iteration 0: Cost = {self.fc:.6f}\n")

            self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dirpath_store_states + "/inexact_gn_hessian_cg/" + "state_GNHessCG_iter_0.nc")

        print("-------------------------------------")
        print(f"Initial fc = {self.fc}")
        print("-------------------------------------")

        for i in range(MAX_ITERS):

            ds_subset_params = self.ds_subset_params.copy()
            ds_subset_gradient = self.eval_gradient()
            ds_subset_descent_dir_hat = self.conjugate_gradient(cg_tolerance_type)

            ds_out = xr.merge([ds_subset_params, ds_subset_descent_dir_hat])
            # Some weird permission denied error if this file is not removed first.
            self.remove_dir(self.dict_ad_out_nc_files["adj"])
            ds_out.to_netcdf(self.dict_ad_out_nc_files["adj"])
            ds_subset_descent_dir = self.eval_sqrt_prior_cov_action("adj")


            alpha, self.fc = self.line_search(ds_subset_gradient,
                                              ds_subset_descent_dir,
                                              init_alpha_cg, min_alpha_cg_tol, c1)

            if alpha <= min_alpha_cg_tol:

                print(f"Step size alpha {alpha} is too small for any real improvement with Inexact GN-Hessian CG, switching to gradient descent for this step.")

                ds_subset_neg_gradient = self.linear_sum([ds_subset_gradient, ds_subset_gradient], 
                                                        [0.0, -1.0], ["adj", "adj"])

                alpha, self.fc = self.line_search(ds_subset_gradient,
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
                    f.write(f"Iteration {i+1}: Cost = {self.fc:.6f}\n")

                self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dirpath_store_states + "/inexact_gn_hessian_cg/" + f"state_GNHessCG_iter_{i+1}.nc")

            print("-------------------------------------")
            print(f"Outer iter {i+1}, fc = {self.fc}")
            print("-------------------------------------")

        return self.ds_subset_params

    @beartype
    def revd(self,
             sampling_param_k_REVD: int, 
             oversampling_param_p_REVD: int = 10,
             mode: str = "misfit_prior_precond") -> Tuple[Float[np.ndarray, "dim_m dim_l"], Float[np.ndarray, "dim_l"]]:

        if mode not in ["misfit_prior_precond", "full_prior_precond"]:
            raise ValueError("revd: Can only decompose full prior-preconditioned Hessian or misfit prior-preconditioned Hessian.")
        elif mode == "full_prior_precond":
            func_hessian_action = self.eval_prior_preconditioned_hessian_action
        elif mode == "misfit_prior_precond":
            func_hessian_action = self.eval_prior_preconditioned_misfit_hessian_action
            
        ds_omega_tlm_only = self.create_ad_tlm_action_input_nc(bool_randomize = True)
        ds_omega = xr.open_dataset(self.dict_ad_inp_nc_files["tlm_action"])
        ds_subset_params = self.subset_of_ds(ds_omega, "type", "nodiff")

        l = sampling_param_k_REVD + oversampling_param_p_REVD
        m, _ = self.flattened_vector(ds_omega_tlm_only, "tlm")
        list_ds_Q_cols = []
        Q = np.empty((0, 0))

        while True:
            ds_subset_y = func_hessian_action()
            _, y = self.flattened_vector(ds_subset_y, "adj")
            
            if Q.size > 0:
                q_tilde = y - Q @ (Q.T @ y)
                q = q_tilde / np.linalg.norm(q_tilde)
                Q = np.hstack([Q, q.reshape(-1, 1)])
            else:
                q = y / np.linalg.norm(y)
                Q = q.reshape(-1, 1)

            ds_q = self.construct_ds(q, ds_omega_tlm_only)
            list_ds_Q_cols.append(ds_q)

            if Q.shape[1] == l:
                break

            ds_omega_tlm_only = self.create_ad_tlm_action_input_nc(bool_randomize = True)
            ds_omega = xr.open_dataset(self.dict_ad_inp_nc_files["tlm_action"])
            ds_subset_params = self.subset_of_ds(ds_omega, "type", "nodiff")

        list_ds_AQ_cols = []

        for ds_q in list_ds_Q_cols:

            dict_tlm_action_only_fields_vals = {}
            for var in ds_q:

                if not self.list_fields_to_ignore or (self.list_fields_to_ignore and var[:-1] not in self.list_fields_to_ignore):
                    if self.dict_tlm_action_fields_or_scalars[var] == "scalar":
                        dict_tlm_action_only_fields_vals[var] = ds_q[var].data.flat[0].copy()
                    else:
                        dict_tlm_action_only_fields_vals[var] = ds_q[var].data.copy()

            _ = self.create_ad_tlm_action_input_nc(dict_tlm_action_only_fields_vals)
            ds_q_fields = self.subset_of_ds(xr.open_dataset(self.dict_ad_inp_nc_files["tlm_action"]), "type", "tlm")

            ds_subset_Aq = func_hessian_action()

            list_ds_AQ_cols.append(ds_subset_Aq)

        T = np.zeros((l, l), dtype = float)
        for i, ds_qi in enumerate(list_ds_Q_cols):
            for j, ds_qj in enumerate(list_ds_AQ_cols):
                T[i, j] = self.l2_inner_product([ds_qi, ds_qj], ["tlm", "adj"])

        Lambda, S = np.linalg.eig(T)
        U = Q @ S

        return U, Lambda

    @beartype
    def forward_uq_propagation(self,
                               sampling_param_k_REVD: int, 
                               oversampling_param_p_REVD: int = 10) -> Tuple[float, float, float]:

        self.copy_dir(self.src_dir + "/driveradjoint", self.src_dir + "/driveradjoint_orig")
        self.copy_dir(self.src_dir + "/driveradjointqoi", self.src_dir + "/driveradjointqoi_orig")
        self.copy_dir(self.src_dir + "/driveradjointqoi", self.src_dir + "/driveradjoint")

        ds_subset_params = self.eval_params()
        ds_subset_gradient_qoi = self.eval_gradient()

        self.copy_dir(self.src_dir + "/driveradjoint_orig", self.src_dir + "/driveradjoint")

        U_misfit, Lambda_misfit = self.revd(sampling_param_k_REVD, 
                                            oversampling_param_p_REVD,
                                            mode = "misfit_prior_precond")

        ds_subset_gradient_qoi_type_tlm = self.exch_tlm_adj_nc(ds_subset_gradient_qoi, og_type = "adj")
        ds_subset_C_gradQoI = self.eval_sqrt_prior_cov_action(ad_key_adj_or_adj_action_or_tlm_action = "tlm_action")
        sigma_B_squared = self.l2_inner_product([ds_subset_C_gradQoI, ds_subset_C_gradQoI], ["tlm", "tlm"])

        sigma_P_squared = sigma_B_squared
        l = sampling_param_k_REVD + oversampling_param_p_REVD

        for i in range(l):

            ds_vi = self.construct_ds(U_misfit[:, i], ds_subset_C_gradQoI)
            sigma_P_squared = sigma_P_squared - Lambda_misfit[i] / (Lambda_misfit[i] + 1) * self.l2_inner_product([ds_subset_C_gradQoI, ds_vi], ["tlm", "tlm"])**2

        delta_sigma_qoi_squared = 1 - sigma_P_squared/sigma_B_squared

        return sigma_B_squared, sigma_P_squared, delta_sigma_qoi_squared

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
                f.write(f"Iteration 0: Cost = {self.fc:.6f}\n")

            self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dirpath_store_states + "/l_bfgs/" + "state_LBFGS_iter_0.nc")

        print("-------------------------------------")
        print(f"Initial fc = {self.fc}")
        print("-------------------------------------")

        m = num_pairs_lbfgs

        list_ds_s = []
        list_ds_y = []

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

                rho = 1 / self.l2_inner_product([list_ds_y[i-idx_lower_limit], list_ds_s[i-idx_lower_limit]], ["adj", "nodiff"])
                list_rhos = [rho] + list_rhos

                alpha = rho * self.l2_inner_product([list_ds_s[i-idx_lower_limit], ds_subset_q], ["nodiff", "adj"])
                list_alphas = [alpha] + list_alphas

                ds_subset_q = self.linear_sum([ds_subset_q, list_ds_y[i-idx_lower_limit]], [1.0, -alpha], ["adj", "adj"])

            if list_ds_s and list_ds_y:
                gamma_k = self.l2_inner_product([list_ds_y[0], list_ds_s[0]], ["adj", "nodiff"]) / self.l2_inner_product([list_ds_y[0], list_ds_y[0]], ["adj", "adj"])
            else:
                gamma_k = 1.0

            if gamma_k <= 0:
                print("l_bfgs: Invalid gamma encountered.")

            ds_subset_p = self.linear_sum([ds_subset_q, ds_subset_q], [0.0, gamma_k], ["adj", "adj"])

            for i in range(idx_lower_limit, k):

                beta = list_rhos[k-i-1] * self.l2_inner_product([list_ds_y[i-idx_lower_limit], ds_subset_p], ["adj", "adj"])
                ds_subset_p = self.linear_sum([ds_subset_p, list_ds_s[i-idx_lower_limit]], [1.0, list_alphas[k-i-1] - beta], ["adj", "nodiff"])  

            alpha_line_search, self.fc = self.line_search(ds_subset_gradient_old,
                                                          ds_subset_p,
                                                          init_alpha, min_alpha_tol, c1)

            print("-------------------------------------")
            print(f"Iter {k+1}, fc = {self.fc}")
            print("-------------------------------------")

            ds_subset_params_new = self.linear_sum([ds_subset_params_old, ds_subset_p], 
                                                   [1.0, alpha_line_search], ["nodiff", "adj"])
            self.write_params(ds_subset_params_new)
            self.ds_subset_params = ds_subset_params_new.copy()

            self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dict_ad_inp_nc_files["adj"])

            if self.dirpath_store_states is not None:

                with open(log_file, "a") as f:
                    f.write(f"Iteration {k+1}: Cost = {self.fc:.6f}\n")

                self.copy_dir(self.dict_ad_inp_nc_files["nodiff"], self.dirpath_store_states + "/l_bfgs/" + f"state_LBFGS_iter_{k+1}.nc")

            ds_subset_gradient_new = self.eval_gradient()

            ds_s_k = self.linear_sum([ds_subset_params_new, ds_subset_params_old], [1.0, -1.0], ["nodiff", "nodiff"])
            ds_y_k = self.linear_sum([ds_subset_gradient_new, ds_subset_gradient_old], [1.0, -1.0], ["adj", "adj"])

            if len(list_ds_s) < m and len(list_ds_y) < m and len(list_ds_s) == len(list_ds_y):
                list_ds_s.append(ds_s_k)
                list_ds_y.append(ds_y_k)
            elif len(list_ds_s) == m and len(list_ds_y) == m:
                list_ds_s.pop(0)
                list_ds_y.pop(0)
                list_ds_s.append(ds_s_k)
                list_ds_y.append(ds_y_k)
            else:
                raise ValueError("l_bfgs: Some issue in lists that store the s and y vectors.")

        return self.ds_subset_params
        
    @beartype
    def linear_sum(self, list_subset_ds: List[Any], list_alphas: List[float], list_types: List[str]) -> Any:

        self.ds_subset_compatibility_check(list_subset_ds, list_types)

        if len(list_alphas) != 2:
            raise ValueError("linear_sum: Only works for two subset_ds, alphas, and types.")

        ds_out = list_subset_ds[0].copy()
    
        for var_0 in list_subset_ds[0]:
    
            if list_types[0] != "nodiff":
                basic_str_0 = var_0[:-1]
            else:
                basic_str_0 = var_0
    
            for var_1 in list_subset_ds[1]:
    
                if list_types[1] != "nodiff":
                    basic_str_1 = var_1[:-1]
                else:
                    basic_str_1 = var_1
    
                if basic_str_0 == basic_str_1:
                    if list_subset_ds[0][var_0].data.shape != list_subset_ds[1][var_1].data.shape:
                        raise ValueError(f"linear_sum: {var_0}, {var_1} do not have the same shape in the two subset_ds.")
    
                    if not self.list_fields_to_ignore or (self.list_fields_to_ignore and basic_str_0 not in self.list_fields_to_ignore):
                        ds_out[var_0].data = list_alphas[0]*list_subset_ds[0][var_0].data.copy() + list_alphas[1]*list_subset_ds[1][var_1].data.copy()
    
        return ds_out

    @beartype
    def l2_inner_product(self, list_subset_ds: List[Any], list_types: List[str]) -> float:

        self.ds_subset_compatibility_check(list_subset_ds, list_types)

        inner_product = 0.0

        for var_0 in list_subset_ds[0]:

            if list_types[0] != "nodiff":
                basic_str_0 = var_0[:-1]
            else:
                basic_str_0 = var_0
    
            for var_1 in list_subset_ds[1]:
    
                if list_types[1] != "nodiff":
                    basic_str_1 = var_1[:-1]
                else:
                    basic_str_1 = var_1
    
                if basic_str_0 == basic_str_1:
                    if list_subset_ds[0][var_0].data.shape != list_subset_ds[1][var_1].data.shape:
                        raise ValueError(f"l2_inner_product: {var_0}, {var_1} do not have the same shape in the two subset_ds.")

                    if not self.list_fields_to_ignore or (self.list_fields_to_ignore and basic_str_0 not in self.list_fields_to_ignore):
                        inner_product = inner_product + np.sum(list_subset_ds[0][var_0].data*list_subset_ds[1][var_1].data)
    
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
            raise ValueError("exch_tlm_adj_nc: Invalid og_type {og_type}.")

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
                                                                                ad_key = ad_key_new)

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
    def ds_subset_compatibility_check(self, list_subset_ds: List[Any], list_types: List[str]) -> None:
    
        if len(list_subset_ds) != 2 or len(list_types) != 2:
            return ValueError("ds_subset_compatibility_check: Only works for two subset_ds at a time")

        for var in list_subset_ds[0]:
            if "type" not in list_subset_ds[0][var].attrs:
                raise ValueError(f"ds_subset_compatibility_check: Attribute 'type' is missing for variable {var} in first subset_ds.")
            elif list_subset_ds[0][var].attrs["type"] != list_types[0]:
                raise ValueError(f"ds_subset_compatibility_check: Type of {var} in first subset_ds is not what is expected i.e. {list_types[0]}.")
            else:
                pass

        for var in list_subset_ds[1]:
            if "type" not in list_subset_ds[1][var].attrs:
                raise ValueError(f"ds_subset_compatibility_check: Attribute 'type' is missing for variable {var} in second subset_ds.")
            elif list_subset_ds[1][var].attrs["type"] != list_types[1]:
                raise ValueError(f"ds_subset_compatibility_check: Type of {var} in second subset_ds is not what is expected i.e. {list_types[1]}.")
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
    
        for var_0 in list_subset_ds[0]:
            if list_suffixes[0] != "":
                basic_str = var_0[:-1]
                var_1 = basic_str + list_suffixes[1]
            else:
                basic_str = var_0
                var_1 = basic_str + list_suffixes[1]

            if var_1 not in list_subset_ds[1]:
                if self.list_fields_to_ignore and basic_str in self.list_fields_to_ignore:
                    pass
                else:
                    raise ValueError(f"ds_subset_compatibility_check: {var_1} not present in second subset_ds when {var_0} is present in first subset_ds.")
    
        for var_1 in list_subset_ds[1]:
            if list_suffixes[1] != "":
                basic_str = var_1[:-1]
                var_0 = basic_str + list_suffixes[0]
            else:
                basic_str = var_1
                var_0 = basic_str + list_suffixes[0]

            if var_0 not in list_subset_ds[0]:
                if self.list_fields_to_ignore and basic_str in self.list_fields_to_ignore:
                    pass
                else:
                    raise ValueError(f"ds_subset_compatibility_check: {var_0} not present in first subset_ds when {var_1} is present in second subset_ds.")
    
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
    def flattened_vector(ds_subset: Any, type_vars: str) -> Tuple[int, Float[np.ndarray, "dim_m"]]:
        m = sum(np.prod(var.shape) for var in ds_subset.data_vars.values())
        flattened_vector = np.concatenate([var.values.ravel() for var in ds_subset.data_vars.values()])
        assert m == flattened_vector.shape[0]

        return m, flattened_vector

    @staticmethod
    @beartype
    def construct_ds(flattened_vector: Float[np.ndarray, "dim dim1"], original_ds: Any) -> Any:

        reconstructed_ds_data = {}
        start = 0
        for var_name, var_data in original_ds.data_vars.items():

            shape = var_data.shape
            size = np.prod(shape)

            reshaped_data = self.flattened_vector[start : start + size].reshape(shape)

            reconstructed_ds_data[var_name] = (var_data.dims, reshaped_data)

            start += size

        reconstructed_ds = xr.Dataset(reconstructed_ds_data, coords=original_ds.coords)

        for var in reconstructed_ds:
            reconstructed_ds[var].attrs["type"] = original_ds[var].attrs["type"]

        return reconstructed_ds

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

