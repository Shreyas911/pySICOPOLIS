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
        return ds_gradient.drop_vars(fields_to_ignore)
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

