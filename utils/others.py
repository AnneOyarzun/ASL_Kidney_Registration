import os
import shutil


def split_path(p):
    a,b = os.path.split(p)
    return (split_path(a) if len(a) and len(b) else []) + [b]


def rewrite_folder(source_folder, destination_folder):
    # Delete the destination folder if it exists
    shutil.rmtree(destination_folder, ignore_errors=True)

    # Copy the contents of the source folder to the destination folder
    shutil.copytree(source_folder, destination_folder)

def delete_folder(folder_path):
    # Delete the folder and its contents
    shutil.rmtree(folder_path)
