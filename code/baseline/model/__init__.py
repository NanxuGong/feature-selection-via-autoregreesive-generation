import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('/root/NIPSAutoFS/code/baseline')
import feature_env
from MARLFS import gen_marlfs
from GFS import gen_gfs
from KBest import gen_kbest
from LASSO import gen_lasso
from mRMR import gen_mrmr
from RFE import gen_rfe
from SARLFS import gen_sarlfs, gen_sarlfs_init
from LASSONet import gen_lassonet
from MCDM import gen_mcdm
