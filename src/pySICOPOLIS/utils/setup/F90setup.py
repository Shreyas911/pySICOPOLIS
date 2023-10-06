import os
import subprocess
import warnings
import re
from pySICOPOLIS.backend.types import Dict

__all__ = ['deleteExistingDir', 'deleteContentsDir',
           'copyExistingDir', 'moveExistingDir',
           'linkExistingDir', 
           'readPragma', 'replacePragmaValuesInFile']

def deleteExistingDir(dirPath: str) -> None:

    """
    If previously existing, delete the directory at dirPath.
    """

    if os.path.isdir(dirPath):
        subprocess.run(['rm', '-rf', dirPath])
    else:
        warnings.warn(f"Directory {dirPath} does not exist, so cannot be deleted.")

    return None

def deleteContentsDir(dirPath: str) -> None:

    """
    If previously existing, delete the contents inside a directory at dirPath.
    """

    if os.path.isdir(dirPath):
        subprocess.run(['rm', '-rf', dirPath+'/*'])
    else:
        warnings.warn(f"Directory {dirPath} does not exist, so its contents cannot be deleted.")

    return None

def copyExistingDir(existingPath: str, newPath: str) -> None:

    """
    Make a copy of an existing directory.
    Parameters
    ----------
    existingPath : str
        Path of the existing directory to copy from.
    newPath : str
        Path of the new directory
    """

    if os.path.isdir(existingPath):
        subprocess.run(['cp', '-r', existingPath, newPath])
    else:
        warnings.warn(f"Directory {existingPath} does not exist, so it cannot be copied.")
    
    return None

def moveExistingDir(existingPath: str, newPath: str) -> None:

    """
    Move an existing directory.
    Parameters
    ----------
    existingPath : str
        Path of the existing directory to move.
    newPath : str
        Path of the new directory
    """

    if os.path.isdir(existingPath):
        subprocess.run(['mv', existingPath, newPath])
    else:
        warnings.warn(f"Directory {existingPath} does not exist, so it cannot be moved.")
    
    return None

def linkExistingDir(existingPath: str, newPath: str) -> None:

    """
    Soft link an existing directory.
    Parameters
    ----------
    existingPath : str
        Path of the existing directory to move.
    newPath : str
        Path of the new directory
    """

    if os.path.isdir(existingPath):
        subprocess.run(['ln', '-s', existingPath, newPath])
    else:
        warnings.warn(f"Directory {existingPath} does not exist, so it cannot be soft linked.")
    
    return None

def readPragma(path: str, fileName: str, pragma: str) -> Dict:

    """
    Read all pragma key and value pairs into a dictionary.
    Parameters
    ----------
    path : str
        Path to the "header" file.
    fileName : str
        Name of the "header" file to be read.
    pragma : str
        pragmas to be read, example '#define', '#include', 'export'
    """

    dict_vars_values = {}

    with open(path+'/'+fileName) as header_file:
        for line in header_file.readlines():
            if line.startswith(pragma):
                # Remove trailing whitespace
                line.rstrip()
                print(line)
                # Use regex to split line into pragma, key, value
                # https://docs.python.org/3/library/re.html
                variables_and_values = re.search(f'{pragma}\s+([A-Za-z]\w+)\s*=*\s*(.*)', 
                                                 line)
                if variables_and_values:
                    dict_vars_values[variables_and_values.group(1)] = variables_and_values.group(2)

    return dict_vars_values

def replacePragmaValuesInFile(path: str, 
                              fileName: str, 
                              pragma: str, 
                              newValues: Dict) -> None:

    """
    Read all pragma key and value pairs, replace some of them in-place in the file.
    Parameters
    ----------
    path : str
        Path to the "header" file.
    fileName : str
        Name of the "header" file to be read.
    pragma : str
        pragmas to be read, example '#define', '#include', 'export'
    newValues : Dict
        Replace values for some of the keys
    """

    with open(path+'/'+fileName) as header_file:
        text = header_file.readlines()
        for i in range(len(text)):
            if text[i].startswith(pragma):
                # Remove trailing whitespace
                text[i].rstrip()
                # Use regex to split line into pragma, key, value
                # https://docs.python.org/3/library/re.html
                variables_and_values = re.search(f'{pragma}\s+([A-Za-z]\w+)\s*=*\s*(.*)', 
                                                 text[i])
                if variables_and_values:
                    if variables_and_values.group(1) in newValues:
                        text[i] = re.sub(variables_and_values.group(2), 
                                         newValues[variables_and_values.group(1)], 
                                         text[i])
                        
    with open(path+'/'+fileName, 'w') as header_file:
        header_file.writelines(text)

    return None