# adapted from scraper at https://github.com/mas-dse-c6sander/DSE_Cohort2_Traffic_Capstone/blob/master/cohort1/traffic/src/scraper.py

import getpass
import json
import logging
import os
# import pickle
import re
import sys
import time
import traceback
from http.cookiejar import LWPCookieJar

import mechanize
from bs4 import BeautifulSoup

FORMAT = '%(asctime)s [%(levelname)-8s] %(message)s'
formatter = logging.Formatter(FORMAT)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
log = logging.Logger('scraper')
log.addHandler(handler)

BASE_URL = 'http://pems.dot.ca.gov'
START_YEAR = 2018
END_YEAR = 2021
BASE_DIR = 'caltrans'
PICKLE_FILENAME = BASE_DIR + '/completed_files.pkl'
DELAY = 2
# UPLOAD = True
S3_DIR = 's3://dse-grp3-capstone-data'

# define the types of files we want
FILE_TYPES = {'station_5min', 'meta', 'chp_incidents_day'}

# Setup download location
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

# Browser
br = mechanize.Browser()

# Cookie Jar
cj = LWPCookieJar()
br.set_cookiejar(cj)

# Browser options
br.set_handle_equiv(True)
br.set_handle_referer(True)
br.set_handle_robots(False)
br.set_handle_redirect(mechanize.HTTPRedirectHandler)

# Follows refresh 0 but not hangs on refresh > 0
br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)

# Want debugging messages?
# br.set_debug_http(True)
# br.set_debug_redirects(True)
# br.set_debug_responses(True)

log.info('Requesting initial page...')

# User-Agent (this is cheating!  But we need data!)
mozilla = 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1)'
gecko = 'Gecko/2008071615'
fedora = 'Fedora/3.0.1-1.fc9'
firefox = 'Firefox/3.0.1'
br.addheaders = [('User-agent', f'{mozilla} {gecko} {fedora} {firefox}')]
br.open(BASE_URL + '/?dnode=Clearinghouse')

log.info('Opened initial page')

br.select_form(nr=0)
br.form['username'] = getpass.getuser('Enter PeMS username')
br.form['password'] = getpass.getpass('Enter PeMS password')

log.info('Logging in...')

br.submit()

return_html = br.response().read()
soup = BeautifulSoup(return_html)
log.debug(soup)

log.info('Logged in.')

# Extract the script with JSON-like structure containing valid request parameter values
script = soup.find('script', text=re.compile('YAHOO\\.bts\\.Data'))
j = re.search(
    r'^\s*YAHOO\.bts\.Data\s*=\s*({.*?})\s*$',
    script.string, flags=re.DOTALL | re.MULTILINE
).group(1)

# The structure is not valid JSON.  The keys are not quoted. Enclose the keys in quotes.
j = re.sub(r'{\s*(\w)', r'{"\1', j)
j = re.sub(r',\s*(\w)', r',"\1', j)
j = re.sub(r'(\w):', r'\1":', j)

# Now that we have valid JSON, parse it into a dict
data = json.loads(j)
assert data['form_data']['reid_raw']['all'] == 'all'  # sanity check

ft = {k: data['form_data'][k].values() for k in data['labels'].keys()}

copySet = set(FILE_TYPES)

# filetype -> year -> district -> month -> set of completed files
# completedFiles = {}

# if os.path.exists(PICKLE_FILENAME):
#     f = open(PICKLE_FILENAME, 'rb')
#     completedFiles = pickle.load(f)
#     f.close()
#     log.info('Restored state from pickle file')

try:
    for fileType in FILE_TYPES:
        # completedFiles.setdefault(fileType, {})

        for year in [str(x) for x in range(START_YEAR, END_YEAR + 1)]:
            # completedFiles[fileType].setdefault(year, {})

            for d in ft[fileType]:
                # fileSet = completedFiles[fileType][year].setdefault(d, set())
                url = f'{BASE_URL}/?srq=clearinghouse'
                options = {
                    'district_id': d,
                    'yy': year,
                    'type': fileType,
                    'returnformat': 'text'
                }

                for key, value in options.items():
                    url += f'&{key}={value}'

                br.open(url)
                json_response = br.response().read()
                responseDict = json.loads(json_response)

                if not responseDict:
                    options = f'district: {d}, year: {year}, filetype: {fileType}'
                    log.info(f'No data available for {options}')
                    continue

                data = responseDict['data']

                for month in data.keys():
                    destDir = f'{BASE_DIR}/{fileType}/{year}/d{d}/'

                    if not os.path.exists(destDir):
                        os.makedirs(destDir)

                    for link in data[month]:
                        filename = link['file_name']

                        # if filename not in fileSet:
                        if os.system(f'aws s3 ls {S3_DIR}/{destDir + filename}') != 0:
                            download_link = f"{BASE_URL}{link['url']}"
                            log.info('Starting to download %s', download_link)
                            br.retrieve(download_link, destDir + filename)[0]
                            log.info('Downloaded %s', filename)
                            # fileSet.add(link['file_name'])

                            # if UPLOAD:
                            cmd = f'aws s3 mv {destDir + filename} {S3_DIR}/{destDir}'
                            log.info('Uploading %s to S3', filename)
                            os.system(cmd)
                            log.info('Uploaded %s', filename)

                            time.sleep(DELAY)
                        else:
                            log.debug('Already downloaded %s.', filename)

        copySet.remove(fileType)
except Exception:
    log.error(traceback.format_exc())
# finally:
#     pickle.dump(completedFiles, open(PICKLE_FILENAME, 'wb'), 2)

# Sanity check to make sure all filetypes were downloaded.
# If not, the scraper needs to updated.
if len(copySet) > 0:
    log.error('Could not complete downloads of filetypes %s', list(copySet))
