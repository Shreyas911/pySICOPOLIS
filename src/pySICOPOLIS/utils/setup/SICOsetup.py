import subprocess
import pySICOPOLIS.utils.setup.F90setup as F90setup
from pySICOPOLIS.backend.types import Dict

__all__ = ['setup_SICOPOLIS', 'LIS_installation']#, 'setup_SICOPOLIS_AD_v2']

class setup_SICOPOLIS:

    def __init__(self,
                 clonePath: str,
                 gitRepoHTTPS: str = "https://gitlab.awi.de/sicopolis/sicopolis.git" 
                 ) -> None:
        
        """
        Initiate the automatic setup of SICOPOLIS object.
        Parameters
        ----------
        clonePath : str
            Path where to clone the git repository
        gitRepoHTTPS : str
            HTTPS for the git repository for cloning purposes.
            Default 'https://gitlab.awi.de/sicopolis/sicopolis.git'
        """

        super().__init__()
        self.gitRepoHTTPS = gitRepoHTTPS
        self.clonePath = clonePath

    def cloneSICOPOLIS(self, branch = 'ad') -> None:

        """
        Clone the git repository at the desired location.
        Parameters
        ----------
        branch : str
            Branch to clone, default 'ad'.
        """

        # In the future, include ability to clone specific commits/tags
        subprocess.run(['git', 'clone', 
                        '--branch', branch,
                        self.gitRepoHTTPS, 
                        self.clonePath])
        
    def moveToNewDirectory(self, newClonePath: str) -> None:

        """
        Move the git repository to the desired location.
        Parameters
        ----------
        newClonePath : str
            New path for the git repository.
        """

        F90setup.moveExistingDir(self.clonePath, newClonePath)
        self.clonePath = newClonePath
        
    def deleteExistingSICOPOLIS(self) -> None:

        """
        If previously existing, delete the directory at self.clonePath.
        """

        F90setup.deleteExistingDir(self.clonePath)
        
    def copyFromExistingSICOPOLIS(self, 
                                  existingPath: str) -> None:

        """
        Make a copy of an existing repository.
        Parameters
        ----------
        existingPath : str
            Path of the existing repository to copy from.
        """

        F90setup.copyExistingDir(existingPath, self.clonePath)

    def copyHeaderTemplates(self,
                            scriptName: str = 'copy_templates.sh') -> None:
        
        """
        Copy header templates from runs/headers/templates to runs/headers
        Parameters
        ----------
        scriptName : str
            Name of shell script that does the copying.
            Default 'copy_templates.sh'.
        """

        subprocess.run(['./'+scriptName], cwd = self.clonePath)
    
    def deleteInputFiles(self,
                         subDir: str = 'sico_in'):
        
        """
        Delete all the input files, maybe to link them instead.
        Parameters
        ----------
        subDir : str
            Subpath to where the input files are, default 'sico_in'
        """
        
        F90setup.deleteContentsDir(self.clonePath+'/'+subDir)

    def downloadInputFiles(self, 
                           scriptName: str = 'get_input_files.sh') -> None:
        
        """
        Download the input files.
        Parameters
        ----------
        scriptName : str
            Name of shell script that downloads the input files.
        """

        subprocess.run(['./'+scriptName], cwd = self.clonePath)

    def linkInputFiles(self, 
                       existingPath: str, 
                       subDir: str = 'sico_in') -> None:
        
        """
        Soft link the input files, generally done instead of downloading to save disk space.
        Parameters
        ----------
        existingPath : str
            Path of the existing repository to soft link from.
        subDir : str
            Subpath to where the input files are, default 'sico_in'
        """
        
        subprocess.run(['ln', '-s', existingPath+'/'+subDir+'/*', '.'], cwd = self.clonePath+'/'+subDir)

    def setSicoConfigs(self,
                       listNewValues: list[str,Dict],
                       subDir: str = 'runs',
                       scriptName: str = 'sico_configs.sh') -> None:
        
        """
        Set configuration for SICOPOLIS.
        Parameters
        ----------
        listNewValues : list[str,Dict]
            List of pragma, dictionary pairs. Each dictionary has keys and new values.
        subDir : str
            Subpath to configs shell script, default 'runs'.
        scriptName : str
            Name of the configs script, default 'sico_configs.sh'.
        """
        
        pass
        # for l in listNewValues:
        #     pragma = l[0]
        #     newValues = l[1]
        #     F90setup.replacePragmaValuesInFile(self.clonePath+'/'+subDir, 
        #                                     scriptName, 
        #                                     pragma, 
        #                                     newValues)

    def setSicoEnv(self,
                   newValues: Dict,
                   pragma: str = 'export',
                   subDir: str = 'runs',
                   fileName: str = 'sico_environment.sh'
                   ) -> None:
        
        """
        Set environment (institution name) for SICOPOLIS.
        Parameters
        ----------
        subDir : str
            Subpath to configs shell script, default 'runs'.
        scriptName : str
            Name of the environment script, default 'sico_environment.sh'.
        """       

        F90setup.replacePragmaValuesInFile(self.clonePath+'/'+subDir, 
                                           fileName, 
                                           pragma, 
                                           newValues)

 
def LIS_installation(path: str, version: str, extension: str):

    subprocess.run(['wget', f'https://www.ssisc.org/lis/dl/lis-{version}.{extension}'], 
                    cwd = path)
    
    if extension == 'zip':
        subprocess.run(['unzip', f'lis-{version}.zip'], 
                    cwd = path)
    elif extension == 'tar.gz':
        subprocess.run(['tar', '-xvzf', f'lis-{version}.tar.gz'], 
                    cwd = path)
    else:
        raise ValueError(f"Can't uncompress this LIS extension .{extension}")


    return None