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
        self.dict_tlm_action_attrs_type = create_dict_tlm_action(dict_params_attrs_type, "tlm_action")
        self.dict_tlm_action_fields_or_scalars = create_dict_tlm_action(dict_params_fields_or_scalars)

        self.NTDAMAX = dict_params_coords["time_ad"].shape[0] - 1
        self.KCMAX   = dict_params_coords["zeta_c"].shape[0] - 1
        self.JMAX    = dict_params_coords["y"].shape[0] - 1
        self.IMAX    = dict_params_coords["x"].shape[0] - 1

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
        ds_gradient = self.subset_of_ds(ds_gradient, attr_key = "type", attr_value = "adj")

        if self.dict_params_fields_or_scalars is not None:

            for var in self.dict_params_fields_or_scalars:
    
                if self.dict_params_fields_or_scalars[var] == "scalar":
                
                    varb = var + "b"
        
                    if varb in ds_gradient:
                        if ds_gradient[varb].attrs["type"] != "adj":
                            raise ValueError(f"eval_gradient: A supposedly adjoint variable should have attribute type adj!")
                        fieldb_sum = np.sum(ds_gradient[varb].data)
                        ds_gradient[varb] = xr.DataArray([fieldb_sum], dims=["scalar"], attrs=ds_gradient[varb].attrs)
                    else:
                        raise ValueError(f"eval_gradient: {varb} not present in ds_gradient!")

        if self.list_fields_to_ignore is not None:
            return ds_gradient.drop_vars(field + "b" for field in self.list_fields_to_ignore)
        else:
            return ds_gradient

    @beartype
    def line_search(self,
                    ds_subset_params: Any, 
                    ds_subset_gradient: Any, 
                    ds_subset_descent_dir: Any,
                    init_alpha: float = 1.0, 
                    c1: float = 1.e-4) -> Tuple[float, float]:
    
        alpha = init_alpha
        
        while True:
            
            ds_subset_params_old = ds_subset_params.copy()
            fc = self.eval_cost()

            try:
                ds_subset_params_new = self.linear_sum([ds_subset_params, ds_subset_descent_dir], 
                                                       [1.0, alpha], ["nodiff", "adj"])
                _ = self.write_params(ds_subset_params_new)
        
                fc_new = self.eval_cost()
        
                pTg = self.l2_inner_product([ds_subset_descent_dir, ds_subset_gradient], ["adj", "adj"])
                ratio = (fc_new - fc)/(alpha*pTg)

            except:
                print("Too big step size probably crashed the simulation.")
                _ = self.write_params(ds_subset_params_old)
                ratio = 0.0

            if ratio >= c1:
                print(f"ratio = {ratio}, alpha = {alpha}")
                return alpha, fc_new
    
            alpha = alpha/2.0

    @beartype
    def gradient_descent(self, 
                         MAX_ITERS: int, 
                         MIN_GRAD_NORM_TOL: Optional[float] = None, 
                         init_alpha: float = 1.0, 
                         c1: float = 1.e-4) -> None:

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
                    if key not in self.list_fields_to_ignore:
                        dict_tlm_action_only_fields_vals[key + "d"] = np.random.randn(*value.shape)
                    else:
                        dict_tlm_action_only_fields_vals[key + "d"] = np.zeros(value.shape, dtype=float)
            else:
                raise ValueError("create_ad_tlm_action_input_nc: Somehow you have entered a condition that's simply impossible.")
                
            ds_inp_tlm_action = self.create_ad_nodiff_or_adj_input_nc(dict_params_only_fields_vals | dict_tlm_action_only_fields_vals,
                                                                      self.dict_tlm_action_fields_num_dims,
                                                                      self.dict_tlm_action_coords,
                                                                      self.dict_tlm_action_attrs_type,
                                                                      self.dict_tlm_action_fields_or_scalars,
                                                                      "tlm_action")

        return ds_inp_tlm_action
            
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
    
                    if self.list_fields_to_ignore and basic_str_0 not in self.list_fields_to_ignore:
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
                        raise ValueError(f"linear_sum: {var_0}, {var_1} do not have the same shape in the two subset_ds.")
                    if self.list_fields_to_ignore and basic_str_0 not in self.list_fields_to_ignore:
                        inner_product = inner_product + np.sum(list_subset_ds[0][var_0].data*list_subset_ds[1][var_1].data)
    
        return inner_product
 
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
                    raise ValueError(f"ds_subset_compatibility_check: {var_0} not present in second subset_ds when {var_1} is present in first subset_ds.")
    
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

