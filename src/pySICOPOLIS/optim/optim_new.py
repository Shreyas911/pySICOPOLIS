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
                 dict_masks_observables: Dict[str, Union[Float[np.ndarray, "dimz dimy dimx"], Float[np.ndarray, "dimy dimx"]]], 
                 list_fields_to_ignore: Optional[List[str]] = None) -> None:
        
        super().__init__()

        if not os.path.isdir(sicopolis_dir):
            raise ValueError(f"DataAssimilation: {sicopolis_dir} doesn't seem to be an existing directory.")

        self.sicopolis_dir = sicopolis_dir
        self.src_dir = sicopolis_dir + "/src"
        self.ad_io_dir = sicopolis_dir + "/src/subroutines/tapenade/ad_io"
        self.sico_out_dir = sicopolis_dir + "/sico_out"

        @beartype
        def has_exact_keys(d: Dict[str, str], required_keys: List[str]) -> bool:
            return list(d.keys()) == required_keys
        self.ad_keys = ["nodiff", "tlm", "adj", "tlm_action", "adj_action"]
        if (
            not has_exact_keys(dict_sico_out_folder_prefixes, self.ad_keys) or 
            not has_exact_keys(dict_ad_exec_cmds_suffixes, self.ad_keys) or
            not has_exact_keys(dict_ad_log_file_suffixes, self.ad_keys) or
            not has_exact_keys(dict_ad_nc_suffixes, self.ad_keys)
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

        self.dict_tlm_action_fields_num_dims = create_dict_tlm_action(dict_params_fields_num_dims)
        self.dict_tlm_action_coords = create_dict_tlm_action(dict_params_coords)
        self.dict_tlm_action_attrs_type = create_dict_tlm_action(dict_params_attrs_type, "tlm")
        self.dict_tlm_action_fields_or_scalars = create_dict_tlm_action(dict_params_fields_or_scalars)

        self.NTDAMAX = dict_params_coords["time_ad"].shape[0] - 1
        self.KCMAX   = dict_params_coords["zeta_c"].shape[0] - 1
        self.JMAX    = dict_params_coords["y"].shape[0] - 1
        self.IMAX    = dict_params_coords["x"].shape[0] - 1

        self.dict_masks_observables = dict_masks_observables

        self.list_fields_to_ignore = list_fields_to_ignore

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
        
        ds = xr.Dataset()  

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
            
            ds[field] = da_field
            if field in dict_attrs_type:
                ds[field].attrs["type"] = dict_attrs_type[field]
            
        ds.to_netcdf(self.dict_ad_inp_nc_files[ad_key])

        return ds

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
    
        return xr.open_dataset(self.dict_ad_out_nc_files[ad_key])

    @beartype
    def get_vx_vy_s(self, sico_out_nc_file: str) -> Tuple[Float[np.ndarray, "dimy dimx"], Float[np.ndarray, "dimy dimx"]]:

        _ = self.run_exec(ad_key = "nodiff")

        path_sico_out_nc = self.dict_sico_out_folders["nodiff"] + "/" + sico_out_nc_file
        if not os.path.isfile(path_sico_out_nc):
            raise ValueError(f"get_vx_vy_s: AD input file {self.dict_sico_out_folders['nodiff']}/{sico_out_nc_file} is missing.")

        ds_out = xr.open_dataset(path_sico_out_nc)

        if "vx_s_g" not in ds_out or "vy_s_g" not in ds_out:
            raise ValueError("get_vx_vy_s: One or both of vx_s_g or vy_s_g missing!")
    
        vx_s_g = ds_out["vx_s_g"].data
        vy_s_g = ds_out["vy_s_g"].data
    
        return vx_s_g, vy_s_g

    @beartype
    def eval_cost(self) -> Union[Float[np.ndarray, "dim"], float]:

        ds_out_nodiff = self.run_exec(ad_key = "nodiff") 
        fc = ds_out_nodiff['fc'].data[0]
    
        return fc

    @beartype
    def eval_params(self) -> Any:

        # Have to evaluate the out nc for coords value consistency when merging with other output datasets.
        ds_params = xr.open_dataset(self.dict_ad_out_nc_files["nodiff"])
        ds_subset_params = self.subset_of_ds(ds_params, attr_key = "type", attr_value = "nodiff")

        if self.dict_params_fields_or_scalars is not None:

            for var in self.dict_params_fields_or_scalars:
    
                if self.dict_params_fields_or_scalars[var] == "scalar":
        
                    if var in ds_subset_params:
                        ds_subset_params[var] = xr.DataArray([ds_subset_params[var].data.flat[0]], 
                                                              dims=["scalar"], attrs=ds_subset_params[var].attrs)
                    else:
                        raise ValueError(f"eval_params: {var} not present in ds_subset_params!")

        return ds_subset_params

    @beartype
    def write_params(self, ds_subset_params: Any) -> Any:

        dict_params_fields_vals = {}
        for var in ds_subset_params:
            if "type" not in ds_subset_params[var].attrs:
                raise ValueError(f"write_params: Attribute 'type' is missing for variable {var} in ds_subset_params.")
            elif ds_subset_params[var].attrs["type"] != "nodiff":
                raise ValueError(f"write_params: Type of {var} is not what is expected i.e. 'nodiff'.")
            elif self.dict_params_fields_or_scalars and self.dict_params_fields_or_scalars[var] == "scalar":
                dict_params_fields_vals[var] = ds_subset_params[var].data[0].copy()
            else:
                dict_params_fields_vals[var] = ds_subset_params[var].data.copy()

        ds_inp_nodiff = self.create_ad_nodiff_or_adj_input_nc(dict_fields_vals = dict_params_fields_vals,
                                                              dict_fields_num_dims = self.dict_params_fields_num_dims,
                                                              dict_coords = self.dict_params_coords,
                                                              dict_attrs_type = self.dict_params_attrs_type,
                                                              dict_fields_or_scalars = self.dict_params_fields_or_scalars,
                                                              ad_key = "nodiff")
 
        return ds_inp_nodiff

    @beartype
    def eval_gradient(self) -> Any:

        ds_gradient = self.run_exec(ad_key = "adj")
        ds_subset_gradient = self.subset_of_ds(ds_gradient, attr_key = "type", attr_value = "adj")

        if self.dict_params_fields_or_scalars is not None:

            for var in self.dict_params_fields_or_scalars:
    
                if self.dict_params_fields_or_scalars[var] == "scalar":
                
                    varb = var + "b"
        
                    if varb in ds_subset_gradient:
                        if ds_subset_gradient[varb].attrs["type"] != "adj":
                            raise ValueError(f"eval_gradient: A supposedly adjoint variable should have attribute type adj!")
                        fieldb_sum = np.sum(ds_subset_gradient[varb].data)
                        ds_subset_gradient[varb] = xr.DataArray([fieldb_sum], dims=["scalar"], attrs=ds_subset_gradient[varb].attrs)
                    else:
                        raise ValueError(f"eval_gradient: {varb} not present in ds_subset_gradient!")

        if self.list_fields_to_ignore is not None:
            return ds_subset_gradient.drop_vars(field + "b" for field in self.list_fields_to_ignore)
        else:
            return ds_subset_gradient

    @beartype
    def line_search(self,
                    ds_subset_params: Any, 
                    ds_subset_gradient: Any, 
                    ds_subset_descent_dir: Any,
                    init_alpha: float = 1.0, 
                    c1: float = 1.e-4) -> Tuple[float, float]:
    
        alpha = init_alpha
        ds_subset_params_orig = ds_subset_params.copy()
        fc = self.eval_cost()

        while True:
            
            try:                
                ds_subset_params_new = self.linear_sum([ds_subset_params_orig, ds_subset_descent_dir], 
                                                       [1.0, alpha], ["nodiff", "adj"])
                _ = self.write_params(ds_subset_params_new)
        
                fc_new = self.eval_cost()
                pTg = self.l2_inner_product([ds_subset_descent_dir, ds_subset_gradient], ["adj", "adj"])
                ratio = (fc_new - fc)/(alpha*pTg)

            except:
                print("Too big step size probably crashed the simulation.")
                _ = self.write_params(ds_subset_params_orig)
                ratio = 0.0

            if ratio >= c1 or alpha <= 1.e-5:
                print(f"Step size alpha = {alpha}")
                return alpha, fc_new
    
            alpha = alpha/2.0

    @beartype
    def gradient_descent(self, 
                         MAX_ITERS: int, 
                         MIN_GRAD_NORM_TOL: Optional[float] = None, 
                         init_alpha: float = 1.0, 
                         c1: float = 1.e-4) -> Any:

        ds_inp = self.create_ad_nodiff_or_adj_input_nc(dict_fields_vals = self.dict_og_params_fields_vals,
                                                       dict_fields_num_dims = self.dict_params_fields_num_dims,
                                                       dict_coords = self.dict_params_coords,
                                                       dict_attrs_type = self.dict_params_attrs_type,
                                                       dict_fields_or_scalars = self.dict_params_fields_or_scalars,
                                                       ad_key = "nodiff")
 
        fc_orig = self.eval_cost()

        print("-------------------------------------")
        print(f"iter 0, fc = {fc_orig}")
        print("-------------------------------------")

        for i in range(MAX_ITERS):

            ds_subset_params = self.eval_params()
            ds_subset_gradient = self.eval_gradient()
            norm_gradient = self.l2_inner_product([ds_subset_gradient, ds_subset_gradient], ["adj", "adj"])**0.5
            
            if MIN_GRAD_NORM_TOL is not None and norm_gradient <= MIN_GRAD_NORM_TOL:
                print("Minimum for gradient norm reached.")
                break

            ds_subset_descent_dir = self.linear_sum([ds_subset_gradient, ds_subset_gradient], 
                                                    [0.0, -1.0], ["adj", "adj"])

            alpha, fc_new = self.line_search(ds_subset_params,
                                             ds_subset_gradient,
                                             ds_subset_descent_dir,
                                             init_alpha, c1)

            ds_subset_params_new = self.linear_sum([ds_subset_params, ds_subset_gradient], 
                                                   [1.0, -alpha], ["nodiff", "adj"])
            _ = self.write_params(ds_subset_params_new)

            print("-------------------------------------")
            print(f"iter {i+1}, fc = {fc_new}")
            print("-------------------------------------")

        return ds_subset_params

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
        else:

            ds_subset_params = self.eval_params()
            dict_params_only_fields_vals = {}

            for var in ds_subset_params:
                if self.dict_params_fields_or_scalars and self.dict_params_fields_or_scalars[var] == "scalar":
                    dict_params_only_fields_vals[var] = ds_subset_params[var].data[0].copy()
                else:
                    dict_params_only_fields_vals[var] = ds_subset_params[var].data.copy()

            if dict_tlm_action_only_fields_vals is not None:
                pass
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


            ds_inp_tlm_action = self.create_ad_nodiff_or_adj_input_nc(dict_params_only_fields_vals | dict_tlm_action_only_fields_vals,
                                                                      self.dict_tlm_action_fields_num_dims,
                                                                      self.dict_tlm_action_coords,
                                                                      self.dict_tlm_action_attrs_type,
                                                                      self.dict_tlm_action_fields_or_scalars,
                                                                      "tlm_action")

        if self.dict_tlm_action_fields_or_scalars is not None:
 
            for var in self.dict_tlm_action_fields_or_scalars:
    
                if self.dict_tlm_action_fields_or_scalars[var] == "scalar":
        
                    if var in ds_inp_tlm_action:
                        ds_inp_tlm_action[var] = xr.DataArray([ds_inp_tlm_action[var].data.flat[0]], 
                                                              dims=["scalar"], attrs=ds_inp_tlm_action[var].attrs)
                    else:
                        raise ValueError(f"create_ad_tlm_action_input_nc: {var} not present in ds_inp_tlm_action!")
      
        return self.subset_of_ds(ds_inp_tlm_action, "type", "tlm")

    @beartype
    def eval_tlm_action(self) -> Any:

        ds_tlm_action = self.run_exec(ad_key = "tlm_action")
        ds_subset_tlm_action = self.subset_of_ds(ds_tlm_action, attr_key = "type", attr_value = "tlmhessaction")

        list_vars = [var[:-1] for var in ds_subset_tlm_action]
        if list(self.dict_masks_observables.keys()) != list_vars:
            raise ValueError("eval_tlm_action: The observables seem to be different than expected.")

        return ds_subset_tlm_action

    @beartype
    def eval_noise_cov_inv_action(self, ds_subset_tlm_action: Any) -> Any:

        ds_subset_tlm_action = ds_subset_tlm_action.copy()

        list_vars = [var[:-1] for var in ds_subset_tlm_action]
        if list(self.dict_masks_observables.keys()) != list_vars:
            raise ValueError("eval_noise_cov_inv_action: The observables seem to be different than expected.")
       
        if not all(value is None for value in self.dict_masks_observables.values()):
            for key, value in self.dict_masks_observables.items():
                if value is not None:
                    if value.shape != ds_subset_tlm_action[key + "d"].data.shape:
                        raise ValueError(f"The right shaped noise covariance mask has not been prescribed for {key}.")
                    ds_subset_tlm_action[key + "d"].data = ds_subset_tlm_action[key + "d"].data*value

        return self.exch_tlm_adj_nc(ds_subset_tlm_action, og_type = "tlmhessaction")

    @beartype
    def eval_adj_action(self) -> Any:

        ds_adj_action = self.run_exec(ad_key = "adj_action")
        ds_subset_adj_action = self.subset_of_ds(ds_adj_action, attr_key = "type", attr_value = "adj")

        if self.dict_params_fields_or_scalars is not None:

            for var in self.dict_params_fields_or_scalars:
    
                if self.dict_params_fields_or_scalars[var] == "scalar":
                
                    varb = var + "b"
        
                    if varb in ds_subset_adj_action:
                        if ds_subset_adj_action[varb].attrs["type"] != "adj":
                            raise ValueError(f"eval_adj_action: A supposedly adjoint variable should have attribute type adj!")
                        fieldb_sum = np.sum(ds_subset_adj_action[varb].data)
                        ds_subset_adj_action[varb] = xr.DataArray([fieldb_sum], dims=["scalar"], attrs=ds_subset_adj_action[varb].attrs)
                    else:
                        raise ValueError(f"eval_adj_action: {varb} not present in ds_subset_adj_action!")

        if self.list_fields_to_ignore is not None:
            return ds_subset_adj_action.drop_vars(field + "b" for field in self.list_fields_to_ignore)
        else:
            return ds_subset_adj_action

    @beartype
    def eval_misfit_hessian_action(self) -> Any:
        ds_subset_tlm_action = self.eval_tlm_action()
        ds_inp_subset_adj_action = self.eval_noise_cov_inv_action(ds_subset_tlm_action)
        return self.eval_adj_action()

    @beartype
    def conjugate_gradient(self, tolerance_type = "superlinear") -> Any:

        ds_subset_params = self.eval_params()

        ds_subset_gradient = self.eval_gradient()
        norm_gradient = self.l2_inner_product([ds_subset_gradient, ds_subset_gradient], ["adj", "adj"])**0.5

        ds_subset_p = self.linear_sum([ds_subset_gradient, ds_subset_gradient], 
                                      [0.0, 0.0], ["adj", "adj"])
        ds_subset_r = self.linear_sum([ds_subset_gradient, ds_subset_gradient], 
                                      [0.0, -1.0], ["adj", "adj"])
        
        ds_subset_v = self.linear_sum([ds_subset_gradient, ds_subset_gradient], 
                                      [0.0, -1.0], ["adj", "adj"])
        ds_subset_v = self.exch_tlm_adj_nc(ds_subset_v, og_type = "adj")

        iters = 0

        while True:

            iters = iters + 1
            print(f"CG iter {iters}")

            ds_subset_Hv = self.eval_misfit_hessian_action()

            vTHv = self.l2_inner_product([ds_subset_v, ds_subset_Hv], ["tlm", "adj"])
            norm_r_old = self.l2_inner_product([ds_subset_r, ds_subset_r], ["adj", "adj"])**0.5

            if vTHv < 0:

                print("conjugate_gradient: Hessian no longer positive definite!")

                pTg = self.l2_inner_product([ds_subset_p, ds_subset_gradient], ["adj", "adj"])
                norm_p = self.l2_inner_product([ds_subset_p, ds_subset_p], ["adj", "adj"])**0.5
                cos = pTg / (norm_p * norm_gradient)
                angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))

                print("Angle between p and g in degrees: ", angle)

                return ds_subset_p

            alpha = norm_r_old**2 / vTHv

            ds_subset_p = self.linear_sum([ds_subset_p, ds_subset_v],
                                          [1.0, alpha], ["adj", "tlm"])

            ds_subset_r = self.linear_sum([ds_subset_p, ds_subset_Hv],
                                          [1.0, -alpha], ["adj", "adj"])

            norm_r = self.l2_inner_product([ds_subset_r, ds_subset_r], ["adj", "adj"])**0.5

            if tolerance_type == "linear":
                eps_TOL = 0.5*norm_gradient
            elif tolerance_type == "superlinear":
                eps_TOL = min(0.5, np.sqrt(norm_gradient))*norm_gradient
            elif tolerance_type == "quadratic":
                eps_TOL = min(0.5, norm_gradient)*norm_gradient
            else:
                raise ValueError("conjugate_gradient: Invalid tolerance_type.")

            if norm_r <= eps_TOL:
                print("conjugate_gradient: Convergence.")

                pTg = self.l2_inner_product([ds_subset_p, ds_subset_gradient], ["adj", "adj"])
                norm_p = self.l2_inner_product([ds_subset_p, ds_subset_p], ["adj", "adj"])**0.5
                cos = pTg / (norm_p * norm_gradient)
                angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))

                print("Angle between p and g in degrees: ", angle)

                return ds_subset_p

            beta = norm_r**2 / norm_r_old**2

            ds_subset_v = self.linear_sum([ds_subset_v, ds_subset_r],
                                          [1.0, beta], ["tlm", "adj"])
            
            dict_tlm_action_only_fields_vals = {}
            for var in ds_subset_v:
                if self.dict_tlm_action_fields_or_scalars[var] == "scalar":
                    dict_tlm_action_only_fields_vals[var] = ds_subset_v[var].data[0].copy()
                else:
                    dict_tlm_action_only_fields_vals[var] = ds_subset_v[var].data.copy()

            ds_subset_v = self.create_ad_tlm_action_input_nc(dict_tlm_action_only_fields_vals)

    @beartype
    def inexact_gn_hessian_cg(self,
                              MAX_ITERS: int,
                              init_alpha: float = 1.0,
                              init_alpha_gd: float = 1.0,
                              cg_tolerance_type: str = "superlinear",
                              c1: float = 1.e-4) -> Any:

        ds_inp = self.create_ad_nodiff_or_adj_input_nc(dict_fields_vals = self.dict_og_params_fields_vals,
                                                       dict_fields_num_dims = self.dict_params_fields_num_dims,
                                                       dict_coords = self.dict_params_coords,
                                                       dict_attrs_type = self.dict_params_attrs_type,
                                                       dict_fields_or_scalars = self.dict_params_fields_or_scalars,
                                                       ad_key = "nodiff")
 
        fc_orig = self.eval_cost()

        print("-------------------------------------")
        print(f"Initial fc = {fc_orig}")
        print("-------------------------------------")

        for i in range(MAX_ITERS):

            ds_subset_params = self.eval_params()
            ds_subset_gradient = self.eval_gradient()
            ds_subset_descent_dir = self.conjugate_gradient(cg_tolerance_type)

            alpha, fc_new = self.line_search(ds_subset_params,
                                             ds_subset_gradient,
                                             ds_subset_descent_dir,
                                             init_alpha, c1)

            if alpha <= 1.e-5:

                print("Step size is too small for any real improvement, switching to gradient descent for this step.")

                ds_subset_neg_gradient = self.linear_sum([ds_subset_gradient, ds_subset_gradient], 
                                                        [0.0, -1.0], ["adj", "adj"])

                alpha, fc_new = self.line_search(ds_subset_params,
                                                 ds_subset_gradient,
                                                 ds_subset_neg_gradient,
                                                 init_alpha_gd, c1)

                ds_subset_params_new = self.linear_sum([ds_subset_params, ds_subset_neg_gradient], 
                                                       [1.0, alpha], ["nodiff", "adj"])
                
            else:

                ds_subset_params_new = self.linear_sum([ds_subset_params, ds_subset_descent_dir], 
                                                       [1.0, alpha], ["nodiff", "adj"])

            _ = self.write_params(ds_subset_params_new)

            print("-------------------------------------")
            print(f"Outer iter {i+1}, fc = {fc_new}")
            print("-------------------------------------")

        return self.eval_params()

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
    def exch_tlm_adj_nc(self, ds_inp_subset_tlm_or_adj_action: Any, og_type: str) -> Any:

        if og_type == "tlmhessaction":
            new_type = "adjhessaction"
            ad_key_new = "adj_action"
        elif og_type == "adj":
            new_type = "tlm"
            ad_key_new = "tlm_action"
        else:
            raise ValueError("exch_tlm_adj_nc: Invalid og_type {og_type}.")

        ds_inp_subset_params = self.eval_params()

        dict_coords = self.dict_params_coords.copy()
        dict_fields_vals = {}

        for var in ds_inp_subset_params:
            if "type" not in ds_inp_subset_params[var].attrs:
                raise ValueError(f"write_params: Attribute 'type' is missing for variable {var} in ds_inp_subset_params.")
            elif ds_inp_subset_params[var].attrs["type"] != "nodiff":
                raise ValueError(f"write_params: Type of {var} is not what is expected i.e. 'nodiff'.")
            elif self.dict_params_fields_or_scalars and self.dict_params_fields_or_scalars[var] == "scalar":
                dict_fields_vals[var] = ds_inp_subset_params[var].data[0].copy()
            else:
                dict_fields_vals[var] = ds_inp_subset_params[var].data.copy()

        if og_type == "tlmhessaction":

            dict_fields_num_dims = self.dict_params_fields_num_dims.copy()
            dict_attrs_type = self.dict_params_attrs_type.copy()
            dict_fields_or_scalars = self.dict_params_fields_or_scalars.copy()
 
            for var in ds_inp_subset_tlm_or_adj_action:    
                dict_fields_num_dims[var[:-1] + "b"] = str(len(ds_inp_subset_tlm_or_adj_action[var].data.shape)) + "D"
                dict_attrs_type[var[:-1] + "b"] = new_type
                dict_fields_vals[var[:-1] + "b"] = ds_inp_subset_tlm_or_adj_action[var].data.copy()
                dict_fields_or_scalars[var[:-1] + "b"] = "field"

        elif og_type == "adj":

            dict_fields_num_dims = self.dict_tlm_action_fields_num_dims.copy()
            dict_attrs_type = self.dict_tlm_action_attrs_type.copy()
            dict_fields_or_scalars = self.dict_tlm_action_fields_or_scalars.copy()

            for var in ds_inp_subset_tlm_or_adj_action:
                if dict_fields_or_scalars[var[:-1] + "d"] == "scalar":
                    dict_fields_vals[var[:-1] + "d"] = ds_inp_subset_tlm_or_adj_action[var].data[0].copy()
                elif dict_fields_or_scalars[var[:-1] + "d"] == "field":
                    dict_fields_vals[var[:-1] + "d"] = ds_inp_subset_tlm_or_adj_action[var].data.copy()
                else:
                    raise ValueError(f"exch_tlm_adj_nc: var {var[:-1] + 'd'} should be either a scalar or a field.")

            if self.list_fields_to_ignore:

                for var in self.list_fields_to_ignore:
                    if dict_fields_or_scalars[var + "d"] == "scalar":
                        dict_fields_vals[var + "d"] = 0.0
                    elif dict_fields_or_scalars[var + "d"] == "field":
                        dict_fields_vals[var + "d"] = ds_inp_subset_params[var].data.copy()*0.0
                    else:
                        raise ValueError(f"exch_tlm_adj_nc: var {var[:-1] + 'd'} should be either a scalar or a field.")

        else:
            raise ValueError("exch_tlm_adj_nc: This should be impossible.")

        ds_inp_tlm_or_adj_action = self.create_ad_nodiff_or_adj_input_nc(dict_fields_vals = dict_fields_vals,
                                                                         dict_fields_num_dims = dict_fields_num_dims,
                                                                         dict_coords = dict_coords,
                                                                         dict_attrs_type = dict_attrs_type,
                                                                         dict_fields_or_scalars = dict_fields_or_scalars,
                                                                         ad_key = ad_key_new)

        ds_inp_tlm_or_adj_action = self.subset_of_ds(ds_inp_tlm_or_adj_action, "type", new_type)

        if og_type == "adj":


            if dict_fields_or_scalars is not None:
    
                for var in ds_inp_tlm_or_adj_action:
        
                    if dict_fields_or_scalars[var] == "scalar":
                        ds_inp_tlm_or_adj_action[var] = xr.DataArray([ds_inp_tlm_or_adj_action[var].data.flat[0]], 
                                                                      dims=["scalar"], attrs=ds_inp_tlm_or_adj_action[var].attrs)

        return ds_inp_tlm_or_adj_action

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
    def remove_dir(path: str) -> None:

        subprocess.run(
            f"rm -rf {path}",
            shell=True)
    
        return None

