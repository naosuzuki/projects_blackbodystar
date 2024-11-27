import numpy
import sys
from iminuit import Minuit
import scipy
import astropy
import astropy.units as u
from astropy.io import ascii
import speclite
import speclite.filters
import pandas as pd
from scipy import optimize
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

c=2.99792458e10 # cm
h=6.6260755e-27 # erg s

surveydict={"sdss":5,"des":5,"gaia":3,"ps1":5,"galex":2,"wise":4}

sdssfilter={0:"sdss2010-u",1:"sdss2010-g",2:"sdss2010-r",3:"sdss2010-i",4:"sdss2010-z"}
sdssdict={0:"SDSS_u",1:"SDSS_g",2:"SDSS_r",3:"SDSS_i",4:"SDSS_z"}
sdssdict_err={0:"SDSS_u_err",1:"SDSS_g_err",2:"SDSS_r_err",3:"SDSS_i_err",4:"SDSS_z_err"}

decamfilter={0:"decamDR1-g",1:"decamDR1-r",2:"decamDR1-i",3:"decamDR1-z",4:"decamDR1-Y"}
decamdict={0:"DES_g",1:"DES_r",2:"DES_i",3:"DES_z",4:"DES_y"}
decamdict_err={0:"DES_g_err",1:"DES_r_err",2:"DES_i_err",3:"DES_z_err",4:"DES_y_err"}

galexfilter={0:"galex-fuv",1:"galex-nuv"}
galexdict={0:"GALEX_fuv_mag",1:"GALEX_nuv_mag"}
galexdict_err={0:"GALEX_fuv_mag_err",1:"GALEX_nuv_mag_err"}

wisefilter={0:"wise2010-W1",1:"wise2010-W2",2:"wise2010-W3",3:"wise2010-W4"}

gaiafilter={0:"gaiadr3-G",1:"gaiadr3-BP",2:"gaiadr3-RP"}
gaiadict={0:"phot_g_mean_ABmag",1:"phot_bp_mean_ABmag",2:"phot_rp_mean_ABmag"}
gaiadict_err={0:"phot_g_mean_ABmag_err",1:"phot_bp_mean_ABmag_err",2:"phot_rp_mean_ABmag_err"}

twomassfilter={0:"twomass-J",1:"twomass-H",2:"twomass-Ks"}
twomassdict={0:"2MASS_j",1:"2MASS_h",2:"2MASS_k"}
twomassdict_err={0:"2MASS_j_err",1:"2MASS_h_err",2:"2MASS_k_err"}

class starDB:
# class star is for a star
# coordinates and magnitudes are stored here
   def __init__(self):
    self.starname       ='BBJ000000+000000'
    self.radeg          =0.0
    self.decdeg         =0.0
    self.objid          =0
    self.source_id      =0
# GAIA
    self.flag_gaia      =numpy.zeros(3,dtype=int)
    self.mag_gaia       =-1.0*numpy.ones([3])
    self.magerr_gaia    =-1.0*numpy.ones([3])
# GALEX
    self.flag_galex     =numpy.zeros(2,dtype=int)
    self.mag_galex      =-1.0*numpy.ones([2])
    self.magerr_galex   =-1.0*numpy.ones([2])
# GALEX FUV
    self.flag_galexfuv  =0
    self.mag_galexfuv   =-1.0*numpy.ones([1])
    self.magerr_galexfuv=-1.0*numpy.ones([1])
# GALEX NUV
    self.flag_galexnuv  =0
    self.mag_galexnuv   =-1.0*numpy.ones([1])
    self.magerr_galexnuv=-1.0*numpy.ones([1])
# SDSS
    self.flag_sdss      =numpy.zeros(5,dtype=int)
    self.mag_sdss       =-1.0*numpy.ones([5])
    self.magerr_sdss    =-1.0*numpy.ones([5])
# Dark Energy Survey
    self.flag_des       =numpy.zeros(5,dtype=int)
    self.mag_des        =-1.0*numpy.ones([5])
    self.magerr_des     =-1.0*numpy.ones([5])
# PanStarrs1
    self.flag_ps1       =numpy.zeros(5,dtype=int)
    self.mag_ps1        =-1.0*numpy.ones([5])
    self.magerr_ps1     =-1.0*numpy.ones([5])
# 2MASS
    self.flag_twomass   =numpy.zeros(3,dtype=int)
    self.mag_twomass    =-1.0*numpy.ones([3])
    self.magerr_twomass =-1.0*numpy.ones([3])

def read_bblist(i,star):
#  csvfile='../data/FINAL_TABLE_BEST_CANDIDATES_WITH_PHOTOMETRY_20241107.csv'
   csvfile='../data/FINAL_TABLE_BEST_CANDIDATES_WITH_PHOTOMETRY.csv'
   df=pd.read_csv(csvfile)
   star.radeg=df['ra'].iloc[i] ; star.decdeg=df['dec'].iloc[i]
   star.objid=df['objid'].iloc[i] ; star.source_id=df['source_id'].iloc[i]
# Gaia Data
   for j in range(3):
      if(df[gaiadict[j]].iloc[i]>0.0): 
         star.flag_gaia[j]=1; star.mag_gaia[j]=df[gaiadict[j]].iloc[i] ; star.magerr_gaia[j]=df[gaiadict_err[j]].iloc[i]
# GALEX Data
   for j in range(2):
      if(df[galexdict[j]].iloc[i]>0.0): 
         star.flag_galex[j]=1; star.mag_galex[j]=df[galexdict[j]].iloc[i] ; star.magerr_galex[j]=df[galexdict_err[j]].iloc[i]
# SDSS
   for j in range(5):
      if(df[sdssdict[j]].iloc[i]>0.0): 
         star.flag_sdss[j]=1
         star.mag_sdss[j]=df[sdssdict[j]].iloc[i] ; star.magerr_sdss[j]=df[sdssdict_err[j]].iloc[i]
# Decam DES
   for j in range(5):
      if(df[decamdict[j]].iloc[i]>0.0): 
         star.flag_des[j]=1
         star.mag_des[j]=df[decamdict[j]].iloc[i] ; star.magerr_des[j]=df[decamdict_err[j]].iloc[i]
# 2MASS
   for j in range(3):
      if(df[twomassdict[j]].iloc[i]>0.0): 
        star.flag_twomass[j]=1
        star.mag_twomass[j]=df[twomassdict[j]].iloc[i] ; star.magerr_twomass[j]=df[twomassdict_err[j]].iloc[i]
# PanStarrs
   if(df['PS1_g'].iloc[i]>0.0 and df['PS1_r'].iloc[i]>0.0):
      star.flag_des=1
      star.mag_ps1[0]=df['PS1_g'].iloc[i]   ; star.magerr_ps1[0]=df['PS1_g_err'].iloc[i]
      star.mag_ps1[1]=df['PS1_r'].iloc[i]   ; star.magerr_ps1[1]=df['PS1_r_err'].iloc[i]
      star.mag_ps1[2]=df['PS1_i'].iloc[i]   ; star.magerr_ps1[2]=df['PS1_i_err'].iloc[i]
      star.mag_ps1[3]=df['PS1_z'].iloc[i]   ; star.magerr_ps1[3]=df['PS1_z_err'].iloc[i]
      star.mag_ps1[4]=df['PS1_y'].iloc[i]   ; star.magerr_ps1[4]=df['PS1_y_err'].iloc[i]

def func_blackbody_flux(x,a,t):
# Input
#   a: normalization (no unit but x 1.0e-23)
#   x: wavelength (Angstrom)
#   t: Temperature (Teff, Kelvin)
# Output
#   f_lambda : erg/cm^2/s/Angstrom
#   1.0e-23*(1.0e8)**4 ; 1.0e8 remains for erg/s/cm^2/"Angstrom"
#   1.0e-23*(1.0e8)**4 : -23+32=1.0e9 
    return 1.0e9*a*2.0*h*c**2/x**5/(numpy.exp(143877505.592/x/t)-1.0)

def func_blackbody_abmag(x,a,t):
# Inputs
#   x: Wavelength Array (Angstrom)
#   a: normalization ( x 1.0e-23)
#   t: Temperature (Kelvin)
# Outputs
#   mag : AB magnitude
#   y   : flux (f_lambda) in erg/cm^2/Angstrom 
#   fnu=f_lambda*lambda**2/c  lambda is in Ang while c is in cm
#   one needs to cancel out Ang =1.0e-8 cm, the other for erg/cm^2/s/"Ang"
    y=func_blackbody_flux(x,a,t)
    return 2.5*numpy.log10(x**2*y*1.0e-8/c)-48.6

def find_blackbody_spectruminABmag(wave,norm,teff):
# Inputs
#   wave: Wavelength Array (Angstrom)
#   norm: normalization ( x 1.0e-23)
#   teff: Temperature (Kelvin)
# Outputs
#   y   : flux (f_lambda) in erg/cm^2/Angstrom 
#   mag : AB magnitude
    y=func_blackbody_flux(wave,norm,teff)
#   fnu=f_lambda*lambda**2/c  lambda is in Ang while c is in cm
#   one needs to cancel out Ang =1.0e-8 cm, the other for erg/cm^2/s/"Ang"
    mab=-2.5*numpy.log10(wave**2*y*1.0e-8/c)-48.6
    return [y,mab]

def chisquared(a,teff):
    [y,abmag]=find_blackbody_spectruminABmag(w,a,teff)
    flux=y * u.erg / (u.cm**2 * u.s * u.Angstrom)

    chisq=0.0
# GALEX FUV
    modelmag_galex=response_galex.get_ab_magnitudes(flux, wave)
    for j in range(2):
       if(bbstar.flag_galex[j]==1): 
          chisq+=((modelmag_galex[galexfilter[j]][0]-bbstar.mag_galex[j])/bbstar.magerr_galex[j])**2
# SDSS
    modelmag_sdss=response_sdss.get_ab_magnitudes(flux, wave)
    for j in range(5):
       if(bbstar.flag_sdss[j]==1): 
          chisq+=((modelmag_sdss[sdssfilter[j]][0]-bbstar.mag_sdss[j])/bbstar.magerr_sdss[j])**2
# GAIA
    modelmag_gaia=response_gaia.get_ab_magnitudes(flux, wave)
    for j in range(3):
       if(bbstar.flag_gaia[j]==1):
          chisq+=((modelmag_gaia[gaiafilter[j]][0]-bbstar.mag_gaia[j])/bbstar.magerr_gaia[j])**2
# 2MASS
    modelmag_twomass=response_twomass.get_ab_magnitudes(flux, wave)
    for j in range(3):
       if(bbstar.flag_twomass[j]==1):
          chisq+=((modelmag_twomass[twomassfilter[j]][0]-bbstar.mag_twomass[j])/bbstar.magerr_twomass[j])**2

    return chisq


# Load Filter Response Functions
response_sdss=speclite.filters.load_filters('sdss2010-*')
response_galex=speclite.filters.load_filters('galex-*')
response_decam=speclite.filters.load_filters('decamDR1-*')
response_wise=speclite.filters.load_filters('wise2010-*')
response_gaia=speclite.filters.load_filters('gaiadr3-*')
response_twomass=speclite.filters.load_filters('twomass-*')

#for i in range(31):
#for i in range(1):
for i in range(2):
  bbstar=starDB()
  read_bblist(i,bbstar)
  print('GAIA',bbstar.mag_gaia)
  print('GALEX FUV',bbstar.mag_galexfuv)
  print('GALEX NUV',bbstar.mag_galexnuv)
  print('SDSS',bbstar.mag_sdss)
  print('PS1',bbstar.mag_ps1)
  print('DES',bbstar.mag_des)
  print('2MASS',bbstar.mag_twomass)
  a=1.0 ;  teff=10000.0
  w=1000.0+numpy.arange(1400)*20.0
  wave=w * u.Angstrom
  [y,abmag]=find_blackbody_spectruminABmag(w,a,teff)
  flux=y * u.erg / (u.cm**2 * u.s * u.Angstrom)

  chisq=chisquared(a,teff)
  print('chisq=',chisq) 
  m=Minuit(chisquared,a=a,teff=teff)
  #m.simplex()
  m.migrad()
  m.hesse()
  # Best Fit Values
  bestnorm=m.values["a"] ; bestteff=m.values["teff"]
  normerr=m.errors["a"]  ; tefferr=m.errors["teff"]
  chisq=chisquared(bestnorm,bestteff)
#  print(i,bestnorm,normerr,bestteff,tefferr,chisq)
  print("BBFIT",bbstar.objid,"%8.4f"%(bestnorm),"%8.4f"%(normerr),"%10.2f"%(bestteff),"%10.2f"%(tefferr),"%8.2f"%(chisq))

