#  © Copyright CERN 2024. All rights not expressly granted are reserved.  #
#                 Author: Gian.Michele.Innocenti@cern.ch                  #
# This program is free software: you can redistribute it and/or modify it #
#  under the terms of the GNU General Public License as published by the  #
# Free Software Foundation, either version 3 of the License, or (at your  #
# option) any later version. This program is distributed in the hope that #
#  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  #
#     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    #
#           See the GNU General Public License for more details.          #
#    You should have received a copy of the GNU General Public License    #
#   along with this program. if not, see <https://www.gnu.org/licenses/>. #

import os
import glob
from pathlib import Path
import shutil

from .logger import get_logger

logger = get_logger()

def list_folders(main_dir: str, filenameinput: str, maxfiles: int, select=None): # pylint: disable=too-many-branches
    """
    Return folders under main_dir which contain filenameinput

    :param maxfiles: limit to maxfiles
    :param select: iterable of substrings that must be contained in folders
    """
    if not os.path.isdir(main_dir):
        logger.error("input directory <%s> does not exist", main_dir)

    files = glob.glob(f'{main_dir}/**/{filenameinput}', recursive=True)
    listfolders = [os.path.relpath(os.path.dirname(file), main_dir) for file in files]

    if select:
        # Select only folders with a matching sub-string in their paths
        list_folders_tmp = []
        for sel_sub_string in select:
            list_folders_tmp.extend([folder for folder in listfolders if sel_sub_string in folder])
        listfolders = list_folders_tmp

    if maxfiles != -1:
        listfolders = listfolders[:maxfiles]

    return  listfolders

def create_folder_struc(maindir: str, listpath: list[str]):
    """
    Reproduce the folder structure as input
    """
    for path in listpath:
        path_elements = path.split("/")
        folder = maindir
        for element in path_elements:
            folder = os.path.join(folder, element)
            if not os.path.exists(folder):
                os.makedirs(folder)

def checkdirs(dirs: list[str]) -> list[str]:
    """
    Return list of existing directories
    """
    if isinstance(dirs, str):
        exdirs = [dirs] if Path(dirs).exists() else []
    else:
        exdirs = [d for d in dirs if Path(d).exists()]
    return exdirs

def checkmakedir(mydir: str):
    """
    Makes directory using 'mkdir'
    """
    if os.path.exists(mydir):
        logger.warning("Using existing folder %s", mydir)
        return
    logger.debug("creating folder %s", mydir)
    os.makedirs(mydir)

def checkmakedirlist(dirlist: list[str]):
    """
    Makes directories from list using 'mkdir'
    """
    for mydir in dirlist:
        checkmakedir(mydir)

def delete_dir(path: str):
    """
    Delete directory if it exists. Return True if success, False otherwise.
    """
    if not os.path.isdir(path):
        logger.warning("Directory %s does not exist", path)
        return True
    logger.warning("Deleting directory %s", path)
    try:
        shutil.rmtree(path)
    except OSError:
        logger.error("Error: Failed to delete directory %s", path)
        return False
    return True

def delete_dirlist(dirlist: list[str]):
    """
    Delete directories from list. Return True if success, False otherwise.
    """
    for path in dirlist:
        if not delete_dir(path):
            return False
    return True

def appendfiletolist(mylist: list[str], namefile: str):
    """
    Append filename to list
    """
    return [os.path.join(path, namefile) for path in mylist]

def appendmainfoldertolist(prefolder: str, mylist: list[str]):
    """
    Append base foldername to paths in list
    """
    return [os.path.join(prefolder, path) for path in mylist]

def createlist(prefolder: str, mylistfolder: list[str], namefile: str):
    """
    Appends base foldername + filename in list
    """
    listfiles = appendfiletolist(mylistfolder, namefile)
    listfiles = appendmainfoldertolist(prefolder, listfiles)
    return listfiles
