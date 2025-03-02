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

# This is a program for blackbody SED fit
# Jan 1st  2025 :  Note Saturday, Mar 1st 2025

# Dec 10th 2024 : Revision on github
# Written by Nao Suzuki in November 2024

# Speed of Light in cgs
c=2.99792458e10 # cm
# Planck Constant in cgs
h=6.6260755e-27 # erg s


# Dictionaries
surveydict={"sdss":5,"des":5,"gaia":3,"ps1":5,"galex":2,"wise":4}

# SDSS : Astropy Filter Names
sdssfilter={0:"sdss2010-u",1:"sdss2010-g",2:"sdss2010-r",3:"sdss2010-i",4:"sdss2010-z"}
sdssdict={0:"SDSS_u",1:"SDSS_g",2:"SDSS_r",3:"SDSS_i",4:"SDSS_z"}
sdssdict_err={0:"SDSS_u_err",1:"SDSS_g_err",2:"SDSS_r_err",3:"SDSS_i_err",4:"SDSS_z_err"}

#SMSS : Sky Mapper Survey
smssfilter={0:"smss-u",1:"smss-v",2:"smss-g",3:"smss-r",4:"smss-i",5:"smss-z"}
smssdict={0:"SMSS_u",1:"SMSS_v",2:"SMSS_g",3:"SMSS_r",4:"SMSS_i",5:"SMSS_z"}
smssdict_err={0:"SMSS_u_err",1:"SMSS_v_err",2:"SMSS_g_err",3:"SMSS_r_err",4:"SMSS_i_err",5:"SMSS_z_err"}

# PanStarrs : 
ps1filter={0:"ps1-g",1:"ps1-r",2:"ps1-i",3:"ps1-z",4:"ps1-y"}
ps1dict={0:"PS1_g",1:"PS1_r",2:"PS1_i",3:"PS1_z",4:"PS1_y"}
ps1dict_err={0:"PS1_g_err",1:"PS1_r_err",2:"PS1_i_err",3:"PS1_z_err",4:"PS1_y_err"}

# DECam DES : Astropy Filter Names
decamfilter={0:"decamDR1-g",1:"decamDR1-r",2:"decamDR1-i",3:"decamDR1-z",4:"decamDR1-Y"}
decamdict={0:"DES_g",1:"DES_r",2:"DES_i",3:"DES_z",4:"DES_y"}
decamdict_err={0:"DES_g_err",1:"DES_r_err",2:"DES_i_err",3:"DES_z_err",4:"DES_y_err"}

# GALEX : Astropy Filter Names
galexfilter={0:"galex-fuv",1:"galex-nuv"}
galexdict={0:"GALEX_fuv_mag",1:"GALEX_nuv_mag"}
galexdict_err={0:"GALEX_fuv_mag_err",1:"GALEX_nuv_mag_err"}

# WISE : Astropy Filter Names
wisefilter={0:"wise2010-W1",1:"wise2010-W2",2:"wise2010-W3",3:"wise2010-W4"}

# GAIA : Astropy Filter Names
gaiafilter={0:"gaiadr3-G",1:"gaiadr3-BP",2:"gaiadr3-RP"}
gaiadict={0:"phot_g_mean_ABmag",1:"phot_bp_mean_ABmag",2:"phot_rp_mean_ABmag"}
gaiadict_err={0:"phot_g_mean_ABmag_err",1:"phot_bp_mean_ABmag_err",2:"phot_rp_mean_ABmag_err"}

# 2MASS : Astropy Filter Names
twomassfilter={0:"twomass-J",1:"twomass-H",2:"twomass-Ks"}
twomassdict={0:"2MASS_j",1:"2MASS_h",2:"2MASS_k"}
twomassdict_err={0:"2MASS_j_err",1:"2MASS_h_err",2:"2MASS_k_err"}

class starDB:
# This program is for blackbody fit to the blackbody candidates
# class star is for a star, it stores coordinates, magnitudetes and their flags
#
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
# SMSS
    self.flag_smss      =numpy.zeros(6,dtype=int)
    self.mag_smss       =-1.0*numpy.ones([6])
    self.magerr_smss    =-1.0*numpy.ones([6])
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
# Reading Blackbody Star List
# RA DEC, objid, source_id
#
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
# SMSS
   for j in range(6):
      if(df[smssdict[j]].iloc[i]>0.0): 
         star.flag_smss[j]=1
         star.mag_smss[j]=df[smssdict[j]].iloc[i] ; star.magerr_smss[j]=df[smssdict_err[j]].iloc[i]
# SDSS
   for j in range(5):
      if(df[sdssdict[j]].iloc[i]>0.0): 
         star.flag_sdss[j]=1
         star.mag_sdss[j]=df[sdssdict[j]].iloc[i] ; star.magerr_sdss[j]=df[sdssdict_err[j]].iloc[i]
# PanStarrs
   for j in range(5):
      if(df[ps1dict[j]].iloc[i]>0.0): 
         star.flag_ps1[j]=1
         star.mag_ps1[j]=df[ps1dict[j]].iloc[i] ; star.magerr_ps1[j]=df[ps1dict_err[j]].iloc[i]
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

def load_smss():
# SkyMapper Filter Response is not listed in astropy
# Loading SkyMapper Filters
# Loading Sky Mapper Response Function : Nov 28th 2024
   df_u=pd.read_csv('../filters/SkyMapper_SkyMapper.u.dat',delim_whitespace=True,names=['wave','u'])
   wave_u=df_u['wave'].to_numpy(); band_u=df_u['u'].to_numpy()
   df_v=pd.read_csv('../filters/SkyMapper_SkyMapper.v.dat',delim_whitespace=True,names=['wave','v'])
   wave_v=df_v['wave'].to_numpy(); band_v=df_v['v'].to_numpy()
   df_g=pd.read_csv('../filters/SkyMapper_SkyMapper.g.dat',delim_whitespace=True,names=['wave','g'])
   wave_g=df_g['wave'].to_numpy(); band_g=df_g['g'].to_numpy()
   df_r=pd.read_csv('../filters/SkyMapper_SkyMapper.r.dat',delim_whitespace=True,names=['wave','r'])
   wave_r=df_r['wave'].to_numpy(); band_r=df_r['r'].to_numpy()
   df_i=pd.read_csv('../filters/SkyMapper_SkyMapper.i.dat',delim_whitespace=True,names=['wave','i'])
   wave_i=df_i['wave'].to_numpy(); band_i=df_i['i'].to_numpy()
   df_z=pd.read_csv('../filters/SkyMapper_SkyMapper.z.dat',delim_whitespace=True,names=['wave','z'])
   wave_z=df_z['wave'].to_numpy(); band_z=df_z['z'].to_numpy()

   smss_u=speclite.filters.FilterResponse(wavelength=wave_u*u.Angstrom,\
          response=band_u,meta=dict(group_name='smss',band_name='u'))
   smss_v=speclite.filters.FilterResponse(wavelength=wave_v*u.Angstrom,\
          response=band_v,meta=dict(group_name='smss',band_name='v'))
   smss_g=speclite.filters.FilterResponse(wavelength=wave_g*u.Angstrom,\
          response=band_g,meta=dict(group_name='smss',band_name='g'))
   smss_r=speclite.filters.FilterResponse(wavelength=wave_r*u.Angstrom,\
          response=band_r,meta=dict(group_name='smss',band_name='r'))
   smss_i=speclite.filters.FilterResponse(wavelength=wave_i*u.Angstrom,\
          response=band_i,meta=dict(group_name='smss',band_name='i'))
   smss_z=speclite.filters.FilterResponse(wavelength=wave_z*u.Angstrom,\
          response=band_z,meta=dict(group_name='smss',band_name='z'))
   response_smss=speclite.filters.load_filters('smss-u','smss-v','smss-g','smss-r','smss-i','smss-z')
   return [response_smss]

def load_ps1():
# PanStarrs Filter Response is not listed in astropy
# Loading PS1 Filters
# Reading PanStarr's Filter Response : Nov 27th 2024
   df=pd.read_csv('../filters/ps1filter.txt',delim_whitespace=True,comment='#',
                  names=['wav','all','g','r','i','z','y','wp1','aero','ray','mol'])
   wave=df['wav'].to_numpy()*10.0
   band_g=df['g'].to_numpy()
   band_r=df['r'].to_numpy()
   band_i=df['i'].to_numpy()
   band_z=df['z'].to_numpy()
   band_y=df['y'].to_numpy()
   ps1_g=speclite.filters.FilterResponse(wavelength=wave*u.Angstrom,\
          response=band_g,meta=dict(group_name='ps1',band_name='g'))
   ps1_r=speclite.filters.FilterResponse(wavelength=wave*u.Angstrom,\
          response=band_r,meta=dict(group_name='ps1',band_name='r'))
   ps1_i=speclite.filters.FilterResponse(wavelength=wave*u.Angstrom,\
          response=band_i,meta=dict(group_name='ps1',band_name='i'))
   ps1_z=speclite.filters.FilterResponse(wavelength=wave*u.Angstrom,\
          response=band_z,meta=dict(group_name='ps1',band_name='z'))
   ps1_y=speclite.filters.FilterResponse(wavelength=wave*u.Angstrom,\
          response=band_y,meta=dict(group_name='ps1',band_name='y'))
   response_ps1=speclite.filters.load_filters('ps1-g','ps1-r','ps1-i','ps1-z','ps1-y')
#  speclite.filters.plot_filters(ps1)
   return [response_ps1]

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
# Chisquared Calculation Body
# Note : this is not the minimization
    [y,abmag]=find_blackbody_spectruminABmag(w,a,teff)
    flux=y * u.erg / (u.cm**2 * u.s * u.Angstrom)

    chisq=0.0
# GALEX
    modelmag_galex=response_galex.get_ab_magnitudes(flux, wave)
    for j in range(2):
       if(bbstar.flag_galex[j]==1): 
          chisq+=((modelmag_galex[galexfilter[j]][0]-bbstar.mag_galex[j])/bbstar.magerr_galex[j])**2
# SDSS
    modelmag_sdss=response_sdss.get_ab_magnitudes(flux, wave)
    for j in range(5):
       if(bbstar.flag_sdss[j]==1): 
          chisq+=((modelmag_sdss[sdssfilter[j]][0]-bbstar.mag_sdss[j])/bbstar.magerr_sdss[j])**2
# SMSS
    modelmag_smss=response_smss.get_ab_magnitudes(flux, wave)
    for j in range(6):
       if(bbstar.flag_smss[j]==1): 
          chisq+=((modelmag_smss[smssfilter[j]][0]-bbstar.mag_smss[j])/bbstar.magerr_smss[j])**2
# PanStarrs1
    modelmag_ps1=response_ps1.get_ab_magnitudes(flux, wave)
    for j in range(5):
       if(bbstar.flag_ps1[j]==1): 
          chisq+=((modelmag_ps1[ps1filter[j]][0]-bbstar.mag_ps1[j])/bbstar.magerr_ps1[j])**2
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

# Program Starts from here
# Filter Preparations

# Loading PanStarrs Filters (not included in astropy)
[response_ps1]=load_ps1()
# Loading SkyMapper Filters
[response_smss]=load_smss()

# Loading Filter Response Functions from astropy
response_sdss=speclite.filters.load_filters('sdss2010-*')
response_galex=speclite.filters.load_filters('galex-*')
response_decam=speclite.filters.load_filters('decamDR1-*')
response_wise=speclite.filters.load_filters('wise2010-*')
response_gaia=speclite.filters.load_filters('gaiadr3-*')
response_twomass=speclite.filters.load_filters('twomass-*')

# Fitting Blackbody Spectra to Ryan's 31 new blackbody stars
# Loop for individual blackbody star (31 stars from Ryan Cooke's list)
#
#
for i in range(31):
#for i in range(1):  As a test
# Define BBstar Class
  bbstar=starDB()
# Reading BBstar info from csv
  read_bblist(i,bbstar)
  print('RA, Dec',bbstar.radeg,bbstar.decdeg)
  print('GAIA',bbstar.mag_gaia)
  print('GALEX',bbstar.mag_galex)
  print('SDSS',bbstar.mag_sdss)
  print('SMSS',bbstar.mag_smss)
  print('PS1',bbstar.mag_ps1)
  print('DES',bbstar.mag_des)
  print('2MASS',bbstar.mag_twomass)

# Default Parameter Setup
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
