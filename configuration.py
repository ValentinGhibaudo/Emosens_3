import sys,os
import getpass

from pathlib import Path

if getpass.getuser() == 'samuel' and  sys.platform.startswith('linux'):
    base_cmo = '/home/samuel/mnt/CRNLDATA//crnldata/cmo'
    
elif getpass.getuser() in ('samuel.garcia', 'valentin.ghibaudo') and  sys.platform.startswith('linux'):
    base_cmo = '/crnldata/cmo/'

elif getpass.getuser() == 'matthias' and  sys.platform.startswith('linux'):
    base_cmo = '/home/matthias/smb4k/CRNLDATA/crnldata/cmo/'


elif getpass.getuser() == 'valentin' and  sys.platform.startswith('linux'):
    base_cmo = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/'

elif sys.platform.startswith('win'):
    base_cmo = 'N:/cmo/'

    p1 = 'N:/cmo/scripts/physio/'
    p2 = 'N:/cmo/Etudiants/NBuonviso2022_Emosens1_O+O-_Valentin/ghibtools/'
    
    sys.path = [ p1, p2] + sys.path



base_cmo = Path(base_cmo)
base_folder = base_cmo / 'Etudiants/NBuonviso2023_Emosens3_OdeurSon_Valentin_Matthias'
data_path = base_folder / 'Data' 

precomputedir = base_folder / 'precompute'
