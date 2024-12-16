import numpy as np
import xarray as xr
import subprocess

__all__ = ["create_ad_input_nc", "run_exec", 
           "copy_dir", "move_dir", "remove_dir", 
           "eval_cost"]

def create_ad_input_nc(dict_fields_vals, 
                       dict_fields_num_dims,
                       dict_dimensions,
                       dict_attrs_type,
                       write_path = None):

    NTDAMAX = dict_dimensions["time_ad"].shape[0] - 1
    KCMAX = dict_dimensions["zeta_c"].shape[0] - 1
    JMAX = dict_dimensions["y"].shape[0] - 1
    IMAX = dict_dimensions["x"].shape[0] - 1

    ds = xr.Dataset()

    if dict_fields_vals.keys() != dict_fields_num_dims.keys():
        raise ValueError("Some fields are not defined as keys in either the values or the num_dims dictionary.")

    for field in dict_fields_vals:

        field_val = dict_fields_vals[field]

        if isinstance(field_val, (int, float)):

            if dict_fields_num_dims[field] == "2D":

                if len(field_val.shape) != 2:
                    raise ValueError(f"Issue with {field}: field_val.shape != num_dims it is supposed to have.")

                da_field = xr.DataArray(
                    data=field_val*np.ones((JMAX+1, IMAX+1), dtype=np.float64),
                    dims=["y", "x"],
                    coords={"y": dict_dimensions["y"].copy(),
                            "x": dict_dimensions["x"].copy()
                            },
                    name=field
                )
            elif dict_fields_num_dims[field] == "3D":

                if len(field_val.shape) != 3:
                    raise ValueError(f"Issue with {field}: field_val.shape != num_dims it is supposed to have.")

                da_field = xr.DataArray(
                    data=field_val*np.ones((KCMAX+1, JMAX+1, IMAX+1), dtype=np.float64),
                    dims=["zeta_c", "y", "x"],
                    coords={"zeta_c": dict_dimensions["zeta_c"].copy(),
                            "y": dict_dimensions["y"].copy(),
                            "x": dict_dimensions["x"].copy()
                            },
                    name=field
                )
            elif dict_fields_num_dims[field] == "2DT":

                if len(field_val.shape) != 3:
                    raise ValueError(f"Issue with {field}: field_val.shape != num_dims it is supposed to have.")

                da_field = xr.DataArray(
                    data=field_val*np.ones((NTDAMAX+1, JMAX+1, IMAX+1), dtype=np.float64),
                    dims=["time_ad", "y", "x"],
                    coords={"time_ad": dict_dimensions["time_ad"].copy(),
                            "y": dict_dimensions["y"].copy(),
                            "x": dict_dimensions["x"].copy()
                            },
                    name=field
                )
            else:
                raise ValueError(f"Issue with {field}: Only 2D or 2DT or 3D fields accepted.")
        
        elif isinstance(field_val, np.ndarray) and not isinstance(field_val, (str, bytes)):

            if dict_fields_num_dims[field] == "2D":

                if len(field_val.shape) != 2:
                    raise ValueError(f"Issue with {field}: field_val.shape != num_dims it is supposed to have.")

                da_field = xr.DataArray(
                    data=field_val.copy(),
                    dims=["y", "x"],
                    coords={"y": dict_dimensions["y"].copy(),
                            "x": dict_dimensions["x"].copy()
                            },
                    name=field
                )

            elif dict_fields_num_dims[field] == "3D":

                if len(field_val.shape) != 3:
                    raise ValueError(f"Issue with {field}: field_val.shape != num_dims it is supposed to have.")

                da_field = xr.DataArray(
                    data=field_val.copy(),
                    dims=["zeta_c", "y", "x"],
                    coords={"zeta_c": dict_dimensions["zeta_c"].copy(),
                            "y": dict_dimensions["y"].copy(),
                            "x": dict_dimensions["x"].copy()
                            },
                    name=field
                )
            elif dict_fields_num_dims[field] == "2DT":

                if len(field_val.shape) != 3:
                    raise ValueError(f"Issue with {field}: field_val.shape != num_dims it is supposed to have.")

                da_field = xr.DataArray(
                    data=field_val.copy(),
                    dims=["time_ad", "y", "x"],
                    coords={"time_ad": dict_dimensions["time_ad"].copy(),
                            "y": dict_dimensions["y"].copy(),
                            "x": dict_dimensions["x"].copy()
                            },
                    name=field
                )
            else:
                raise ValueError(f"Issue with {field}: Only 2D or 2DT or 3D fields accepted.")

        else:
            raise TypeError(f"Issue with {field}: The type doesn't seem to be either a scalar or a numpy array.")

        ds[field] = da_field
        if field in dict_attrs_type:
            ds[field].attrs["type"] = dict_attrs_type[field]

    if write_path is not None:
        ds.to_netcdf(write_path)

    return ds

def run_exec(sicopolis_dir,
             log_file,
             sico_out_folder,
             ad_output_nc,
             exec_cmd):

    src_dir = sicopolis_dir + "/src"
    ad_io_dir = sicopolis_dir + "/src/subroutines/tapenade/ad_io"
    sico_out_dir = sicopolis_dir + "/sico_out"

    subprocess.run(
        f"rm -r {sico_out_dir}/{sico_out_folder}",
        cwd=src_dir,
        shell=True)

    subprocess.run(
        f"rm {ad_io_dir}/{ad_output_nc}",
        cwd=src_dir,
        shell=True)

    with open(f"{src_dir}/{log_file}", "w") as log:
        subprocess.run(
            f"time {exec_cmd}",
            cwd=src_dir,
            shell=True,
            stdout=log,
            stderr=subprocess.STDOUT)

    return None

def copy_dir(old_path, new_path):

    subprocess.run(
        f"cp -r {old_path} {new_path}",
        shell=True)

    return None

def move_dir(old_path, new_path):

    subprocess.run(
        f"mv {old_path} {new_path}",
        shell=True)

    return None

def remove_dir(path):

    subprocess.run(
        f"rm -rf {path}",
        shell=True)

    return None

def get_vx_vy_s(sicopolis_dir,
                log_file,
                sico_out_folder,
                sico_out_nc_file,
                ad_output_nc = "ad_output_nodiff.nc",
                exec_cmd = "./drivernodiff"):

    run_exec(sicopolis_dir,
             log_file,
             sico_out_folder,
             ad_output_nc,
             exec_cmd)

    file_nc = sicopolis_dir + "/sico_out/" + sico_out_folder + "/" + sico_out_nc_file

    if "vx_s_g" not in file_nc or "vy_s_g" not in nc:
        raise ValueError("get_vx_vy_s: One or both of vx_s_g or vy_s_g missing!")

    vx_s_g = ds_out["vx_s_g"].data
    vy_s_g = ds_out["vy_s_g"].data

    return vx_s_g, vy_s_g


def eval_cost(sicopolis_dir,
              log_file,
              sico_out_folder,
              ad_output_nc = "ad_output_nodiff.nc",
              exec_cmd = "./drivernodiff"):

    run_exec(sicopolis_dir,
             log_file,
             sico_out_folder,
             ad_output_nc,
             exec_cmd)

    ad_io_dir = sicopolis_dir + "/src/subroutines/tapenade/ad_io"
    ds_out_nodiff = xr.open_dataset(ad_io_dir + '/ad_output_nodiff.nc')
    fc = ds_out_nodiff['fc'].data

    return fc

def eval_gradient(sicopolis_dir,
                  log_file,
                  sico_out_folder,
                  ad_output_nc = "ad_output_adj.nc",
                  exec_cmd = "./driveradjoint",
                  fields_to_ignore = None,
                  dict_fields_or_scalars = None):

    run_exec(sicopolis_dir,
             log_file,
             sico_out_folder,
             ad_output_nc,
             exec_cmd)

    ad_io_dir = sicopolis_dir + "/src/subroutines/tapenade/ad_io"
    ds_gradient = xr.open_dataset(ad_io_dir + '/ad_output_adj.nc')

    if dict_fields_or_scalars is not None:
        for var in dict_fields_or_scalars:

            if dict_fields_or_scalars[var] == "scalar":
            
                varb = var + "b"
    
                if varb in ds_gradient:
                    if ds_gradient[varb].attrs["type"] != "adj":
                        raise ValueError(f"eval_gradient: A supposedly adjoint variable should have attribute type adj!")
                    fieldb_sum = np.sum(ds_gradient[varb].data)
                    ds_gradient[varb] = xr.DataArray([fieldb_sum], dims=["scalar"], attrs=ds_gradient[varb].attrs)
                else:
                    raise ValueError(f"eval_gradient: {var} not present in ds_gradient!")

    if fields_to_ignore is not None:
        return ds_gradient.drop_vars(field + "b" for field in fields_to_ignore)
    else:
        return ds_gradient

def subset_of_ds(ds, attr_key, attr_value):

    subset_vars = []
    for var in ds:
        if attr_key not in ds[var].attrs:
            raise ValueError(f"ds_subset: Attribute '{attr_key}' is missing for variable {var}")
        else:
            if ds[var].attrs[attr_key] == attr_value:
                subset_vars.append(var)

    ds_subset = ds[subset_vars]

    return ds_subset

def ds_compatibility_prep(list_ds, list_types,
                          fields_to_ignore = None):

    if len(list_ds) != 2 or len(list_types) != 2:
        return ValueError("ds_compatibility_prep: Only works for two ds at a time")

    list_suffixes = []
    for type_var in list_types:
        if type_var == "nodiff":
            list_suffixes.append("")
        elif type_var == "adj":
            list_suffixes.append("b")
        elif type_var == "tlm":
            list_suffixes.append("d")
        else:
            return ValueError(f"ds_compatibility_prep: {type_var} is not a valid type for this function for now.")

    ds_subset_0 = subset_of_ds(list_ds[0], "type", list_types[0]).copy()
    ds_subset_1 = subset_of_ds(list_ds[1], "type", list_types[1]).copy()

    for var_0 in ds_subset_0:
        if list_suffixes[0] != "":
            basic_str = var_0[:-1]
            var_1 = basic_str + list_suffixes[1]
        else:
            basic_str = var_0
            var_1 = basic_str + list_suffixes[1]
        if var_1 not in ds_subset_1:
            if fields_to_ignore and basic_str in fields_to_ignore:
                pass
            else:
                raise ValueError(f"ds_compatibility_prep: {var_1} not present in second ds when {var_0} is present in first ds.")

    for var_1 in ds_subset_1:
        if list_suffixes[1] != "":
            basic_str = var_1[:-1]
            var_0 = basic_str + list_suffixes[0]
        else:
            basic_str = var_1
            var_0 = basic_str + list_suffixes[0]
        if var_0 not in ds_subset_0:
            if fields_to_ignore and basic_str in fields_to_ignore:
                pass
            else:
                raise ValueError(f"ds_compatibility_prep: {var_0} not present in second ds when {var_1} is present in first ds.")

    return ds_subset_0, ds_subset_1

def L2_inner_product(ds_1, ds_2, type_var, fields_to_ignore = None):

    ds_subset_1, ds_subset_2 = ds_compatibility_prep([ds_1, ds_2],
                                                      [type_var, type_var],
                                                      fields_to_ignore = fields_to_ignore)

    inner_product = 0.0

    for var in ds_subset_1:

        if type_var != "nodiff":
            temp = var[:-1]
        else:
            temp = var

        if fields_to_ignore and temp not in fields_to_ignore:
            if ds_subset_1[var].data.shape != ds_subset_2[var].data.shape:
                raise ValueError("L2_inner_product: {var} does not have the same shape in both datasets.")
            inner_product = inner_product + np.sum(ds_subset_1[var].data*ds_subset_2[var].data)

    return inner_product
   
def grad_descent_step(ds_state, ds_gradient, alpha, dict_fields_or_scalars = None, fields_to_ignore = None):

    ds_s, ds_g = ds_compatibility_prep([ds_state, ds_gradient], 
                                       ["nodiff", "adj"],
                                       fields_to_ignore = fields_to_ignore)

    for var in ds_s:
        if fields_to_ignore and var not in fields_to_ignore:
            if dict_fields_or_scalars is not None and var in dict_fields_or_scalars[var] == "scalar":
                if ds_g[var + "b"].data.shape == (1,):
                    ds_s[var].data[:] = ds_s[var].data.flat[0] - alpha*ds_g[var + "b"].data
                else:
                    raise ValueError(f"grad_descent_step: {var + 'b'} should be a scalar gradient value.")
            else:
                ds_s[var].data = ds_s[var].data - alpha*ds_g[var + "b"].data

    return ds_s

def line_search(ds_state, ds_gradient, ds_descent_dir,
                eval_cost, sicopolis_dir, log_file, sico_out_folder,
                init_alpha = 1.0,
                ad_input_nc = "ad_input_nodiff.nc",
                ad_output_nc = "ad_output_nodiff.nc",
                exec_cmd = "./drivernodiff", c1 = 1.e-4, 
                fields_to_ignore = None, dict_fields_or_scalars = None):

    alpha = init_alpha

    while True:

        ds_s, ds_p = ds_compatibility_prep([ds_state, ds_descent_dir], 
                                           ["nodiff", "adj"],
                                           fields_to_ignore = fields_to_ignore)
        _, ds_g = ds_compatibility_prep([ds_state, ds_gradient], 
                                        ["nodiff", "adj"],
                                        fields_to_ignore = fields_to_ignore)

        fc = eval_cost(sicopolis_dir,
                       log_file,
                       sico_out_folder,
                       ad_output_nc,
                       exec_cmd)

        for var in ds_s:
            if fields_to_ignore and var not in fields_to_ignore:
                if dict_fields_or_scalars is not None and var in dict_fields_or_scalars[var] == "scalar":
                    if ds_p[var + "b"].data.shape == (1,):
                        ds_s[var].data[:] = ds_s[var].data.flat[0] + alpha*ds_p[var + "b"].data
                    else:
                        raise ValueError(f"line_search: {var + 'b'} should be a scalar gradient value.")
                else:
                    ds_s[var].data = ds_s[var].data + alpha*ds_p[var + "b"].data

        ad_io_dir = sicopolis_dir + "/src/subroutines/tapenade/ad_io"
        ds_s.to_netcdf(ad_io_dir + "/" + ad_input_nc)

        fc_new = eval_cost(sicopolis_dir,
                           log_file,
                           sico_out_folder,
                           ad_output_nc,
                           exec_cmd)

        pTg = L2_inner_product(ds_p, ds_g, type_var="adj", fields_to_ignore=fields_to_ignore)

        ratio = (fc_new - fc)/(alpha*pTg)

        print(ratio)
        if ratio >= c1:
            print(f"alpha = {alpha}")
            return alpha

        alpha = alpha/2.0

