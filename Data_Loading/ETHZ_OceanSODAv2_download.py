#!/usr/bin/env python

import os
import logging
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
from bs4 import BeautifulSoup
from tqdm import tqdm
import concurrent.futures
import time
import data_utils as du

# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s: %(message)s',
#         handlers=[
#             logging.FileHandler('download.log'),
#             logging.StreamHandler()
#         ]
#     )
#     return logging.getLogger(__name__)

def download_file(base_url, filename, username, password, download_dir):
    file_url = base_url + filename
    local_filepath = os.path.join(download_dir, filename)

    try:
        with requests.get(
            file_url,
            auth=HTTPDigestAuth(username, password),
            stream=True
        ) as response:
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(local_filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    progress_bar.update(size)

        return True

    except requests.RequestException as e:
        logging.error(f"Error downloading {filename}: {e}")
        return False

def download_netcdf_files(base_url, download_dir, max_workers=5):
    # logger = setup_logging()
    # download_dir = "downloaded_files"
    os.makedirs(download_dir, exist_ok=True)

    try:
        response = requests.get(
            base_url,
            auth=HTTPDigestAuth('','')
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.nc')]

        print(f"Found {len(links)} NetCDF files to download")

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            download_tasks = [
                executor.submit(
                    download_file,
                    base_url,
                    link,
                    '',
                    '',
                    download_dir
                )
                for link in links
            ]

            successful_downloads = 0
            failed_downloads = 0

            for future in concurrent.futures.as_completed(download_tasks):
                if future.result():
                    successful_downloads += 1
                else:
                    failed_downloads += 1

        total_time = time.time() - start_time

        # logger.info(f"Download Summary:")
        # logger.info(f"Total files: {len(links)}")
        # logger.info(f"Successful downloads: {successful_downloads}")
        # logger.info(f"Failed downloads: {failed_downloads}")
        # logger.info(f"Total download time: {total_time:.2f} seconds")

    except Exception as e:
        print(f"Critical error in download process: {e}")





def ethz_oceansoda_v2_download(loc,start_yr,end_yr,variable,web='https://data.up.ethz.ch/shared/ESA-OHOA/OceanSODA_ETHZ_HR-v2023.01-full_carbsys'):
    folder_loc = os.path.join(loc,variable)
    du.makefolder(folder_loc)
    yr = start_yr



    while yr <=end_yr:
        folder_loc = os.path.join(loc,variable,str(yr))
        du.makefolder(folder_loc)
        download_netcdf_files(web+'/'+variable+'/'+str(yr)+'/',folder_loc,max_workers=3)
        yr = yr+1

def main():
    ethz_oceansoda_v2_download('D:/Data/ETHZ_OceanSODA_v2',1982,2022,'h_free')

if __name__ == "__main__":
    main()
