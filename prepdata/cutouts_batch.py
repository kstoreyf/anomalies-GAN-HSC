# **************************************************
# * File Name : cutouts_batch.py
# * Creation Date : 2019-07-23
# * Created By : kstoreyf
# * Description : Get a batch of cutouts from the HSC
# *    database. based off of the script:
# https://github.com/cbottrell/SubaruHSC_TidalCNN/blob/master/batch_download.py
# **************************************************

import os
import sys
import numpy as np
import time
import math
import tarfile
import shutil
from multiprocessing import Pool

import astropy.units as u
from astropy import wcs
from astropy.io import fits
from astropy.coordinates import SkyCoord
import requests
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd

from unagi import config
from unagi import hsc
from unagi.task import hsc_tricolor, hsc_cutout


tag = 'i20.0_batch'

def main():
    
    nsample = 'all'
    #nsample = 30
    filters = ['I']
    batch_size = int(1000/len(filters))
    exclude = None
    s_ang = 10 #arcsec

    starttime = time.time()

    gen_scripts = 1
    sub_dir = f"submit_scripts/subs_{tag}" 
    if os.path.isdir(sub_dir) and os.listdir(sub_dir):
        fns = os.listdir(sub_dir)
        sub_scripts = [f"{sub_dir}/{fn}" for fn in fns if fn.endswith(".txt") and "done" not in fn]
        if sub_scripts:
            print(f"Using {len(sub_scripts)} existing submission scripts")
            gen_scripts = 0
    
    if gen_scripts:
        print("Generating subsample and submission scripts")
        print("Loading catalog")
        #fn = "../data/hsc_catalogs/pdr2_wide_icmod_21.0_21.5_fdfc_bsm_shape.fits"
        fn = "../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5.csv"
        if fn.endswith(".fits"):
            hdul = fits.open(fn, memmap=True)
            alldata = hdul[1].data
        elif fn.endswith(".csv"):
            alldata = pd.read_csv(fn)
        print(f"Full catalog: {len(alldata)} objects")

        data = clean_data(alldata)        
        if nsample=='all':
            indices_sample = np.arange(len(data))
        else:
            indices_sample = get_indices(nsample, len(data), exclude=exclude)
            data = data.iloc[indices_sample]

        start = 0
        end = batch_size
        moredata = 1
        batch_num = 0

        sub_scripts = []
        
        while moredata:
            print("Batch {}".format(batch_num))
            if end>=len(data):
                end = len(data)
                moredata = 0
            batch = data.iloc[start:end]
            indices = indices_sample[start:end]

            f_name = write_batch_file(batch, indices, s_ang, filters, batch_num)        
            sub_scripts.append(f_name)

            batch_num += 1
            start += batch_size
            end += batch_size

    with Pool(2) as p:
        p.map(process_scripts, sub_scripts)
        
    endtime = time.time()
    print(f"Time: {endtime-starttime}")
    print("Done!")


def get_indices(nsample, ntotal, exclude=None):
    if exclude:
        excl = np.load(exclude)
        #https://stackoverflow.com/questions/3462143/get-difference-between-two-lists
        incl = list(set(range(ntotal)) - set(excl))
    else:
        incl = ntotal
    indices = np.random.choice(incl, size=nsample, replace=False)
    return indices


def process_scripts(f_name):
    download_file(f_name)
    extract_and_rename(f_name)


def write_batch_file(data, indices, s_ang, filters, batch_num):
     
    sub_dir = f"submit_scripts/subs_{tag}"
    f_name = f"{sub_dir}/submit_{tag}-{batch_num}.txt"
    if os.path.isfile(f_name):
        print(f"File {f_name} exists")
        return f_name

    if not os.path.isdir(sub_dir):
        os.mkdir(sub_dir)
    
    das_header = "#? rerun filter ra dec sw sh # column descriptor\n"
    das_list = []
    for i in range(len(data)):
        obj = data.iloc[i]
        ra = obj['ra']
        dec = obj['dec']
        obj_id = obj['object_id']
        for filt in filters:
            row = f" pdr2_wide HSC-{filt} {ra} {dec} {s_ang}asec {s_ang}asec # {obj_id} {indices[i]}"
            das_list.append(row) 
   
    with open(f_name, "w") as f:
        f.write(das_header)
        f.write("\n".join(das_list))
    return f_name


def download_file(args):
    start = time.time()
    submission = args

    new_dir = f"{submission[:-4]}_dir"
    # Check if directory already exists and is not empty - may already have downloaded
    if os.path.isdir(new_dir) and os.listdir(new_dir):
        if os.listdir(new_dir)[0].startswith("arch"):
            return

    print(f"Downloading {submission}...")
    with open("cred.dat", "r") as fc:
        user = fc.readline().strip()
        pw = fc.readline().strip()

    with open(submission, "r") as f:
        while True:
            # https://stackoverflow.com/a/37573701
            r = requests.post("https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout",
                                files={"list":f},
                                auth=(user, pw),
                                stream=False)

            if r.status_code!=200:
                print(f"Failed Download for:{submission}. HTTP Code: {r.status_code}. Waiting 30 seconds...")
                time.sleep(30)
            else:
                break

    total_size = 70*1024*1000
    block_size = 1024
    print("Here we go")
    with open(f'{submission}.tar.gz', 'wb') as f:
        for data in tqdm(r.iter_content(block_size),
                            total=math.ceil(total_size//block_size),
                            unit='KB', unit_scale=True,
                            desc=f"Downloading {submission}"):
            f.write(data)
    
    end = time.time()
    print('Download done! Time:',end-start)
    sys.stdout.flush()


def extract_and_rename(submission):
    print(f"Extracting files for {submission}")
    start = time.time()
    new_dir = f"{submission[:-4]}_dir"
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    
    if os.path.isfile(f"{submission}.tar.gz"):
        with tarfile.TarFile(f"{submission}.tar.gz", "r") as tarball:
            tarball.extractall(new_dir)

    with open(submission) as f:
        lines = [l.strip().split() for l in f.readlines()[1:]]

    # the HSC api names the folder randomly
    # check if folder empty
    if not os.listdir(new_dir):
        print(f"No downloads for {submission}")
        return   
    sub_dir = os.listdir(new_dir)[0]

    img_dir = os.path.join(new_dir, sub_dir)
    completed = []
    nparrs = []
    indices = []
    # Combine fits files into one numpy array per batch
    for f in sorted(os.listdir(img_dir)):
        row_id = int(f.split("-")[0])
        
        obj_details = lines[row_id-2] # hsc counts from 1 and we dropped the header

        obj_id = obj_details[-2]
        obj_idx = obj_details[-1]
        band = obj_details[1].split("-")[1]

        nparrs.append(unpack(f"{img_dir}/{f}"))
        indices.append(int(obj_idx))

        completed.append(obj_details)

    for c in completed:
        lines.remove(c)

    # Remove completed files
    shutil.rmtree(new_dir)
    if os.path.isfile(f"{submission}.tar.gz"):
        os.remove(f'{submission}.tar.gz')
    script_num = submission.split('-')[-1][:-4]

    if len(lines)>0:
        with open(f"{submission}.err.txt", "w") as f:
            f.write("The following objects failed:\n")
            for o in obj_details:
                f.write(",".join(o) + "\n")
    
    # Save as numpy arrays and remove fits files
    npdir = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarrs_{tag}'    
    if not os.path.isdir(npdir):
        os.mkdir(npdir)
    
    np.save(f'{npdir}/hsc_{tag}_{script_num}.npy', nparrs)
    np.save(f'{npdir}/hsc_{tag}_{script_num}_idx.npy', indices) 

    os.rename(submission, f"{submission[:-4]}-done.txt")

    end = time.time()
    print('Extraction done! Time:',end-start)
    sys.stdout.flush()


def unpack(fn):
    size = 96
    with fits.open(fn, memmap=False) as hdul:
        im = hdul[1].data
    centh = im.shape[0]/2
    centw = im.shape[1]/2
    lh, rh = int(centh-size/2), int(centh+size/2)
    lw, rw = int(centw-size/2), int(centw+size/2)
    cropped = im[lh:rh, lw:rw]
    assert cropped.shape[0]==size and cropped.shape[1]==size, f"Wrong size! Still {cropped.shape}"
    return cropped


# Clean sample
def clean_data(data):
    print("Cleaning data")
    start = time.time()
    bands = ['g','r','i','z','y']
    badflags = ['pixelflags_edge',
                'pixelflags_interpolatedcenter',
                'pixelflags_saturatedcenter',
                'pixelflags_crcenter',
                'cmodel_flag']
    for i in range(len(badflags)):
        for b in range(len(bands)):
            flag = '{}_{}'.format(bands[b], badflags[i])
            data = data[data[flag]==0]
    #data = data[data['reliable']!=1]
    end = time.time()
    print(f"Cleaned catalog: {len(data)} objects")
    print('Time clean:',end-start)
    return data 


if __name__=='__main__':
    main()
