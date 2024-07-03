class Header:
    
    def __init__(self, tel_unit=1): # telescope unit (1,2,3,4)
        self.tel_unit = tel_unit
        self.dict = dict(
            obj      = 'OBJECT',
            instr    = 'INSTRUME',
            date     = 'DATE-OBS',
            # bjd      = 'HIERARCH ESO QC BJD',
            airmass  = f'HIERARCH ESO TEL{self.tel_unit} AIRM AVG', # WARNING: ADD this keyword manually from START / END airmass
            seeing   = f'HIERARCH ESO TEL{self.tel_unit} IA FWHM', # [arcsec]
            # ccf_rv       = 'HIERARCH ESO QC CCF RV', # Radial velocity [km/s]
            # ccf_rv_err   = 'HIERARCH ESO QC CCF RV ERROR', # Uncertainty on radial velocity [km/s]
            berv     = 'HIERARCH ESO QC BERV', # [km/s]
            ccf_fwhm     = 'HIERARCH ESO QC CCF FWHM', # CCF FWHM [km/s] 
            ccf_fwhm_err = 'HIERARCH ESO QC CCF FWHM ERROR', # Uncertainty on CCF FWHM [km/s]
            cont     = 'HIERARCH ESO QC CCF CONTRAST', # CCF contrast [%]                
            cont_err = 'HIERARCH ESO QC CCF CONTRAST ERROR', # CCF contrast error [%] 
            bis      = 'HIERARCH ESO QC CCF BIS SPAN', # CCF bisector span [km/s]     
            bis_err  = 'HIERARCH ESO QC CCF BIS SPAN ERROR', # CCF bisector span err
            exptime  = 'EXPTIME',
            snr50    = 'HIERARCH ESO QC ORDER50 SNR', 
            snr102   = 'HIERARCH ESO QC ORDER102 SNR', # order at 5500 A
            snr138   = 'HIERARCH ESO QC ORDER138 SNR', # order at lambda = 6560 A (Halpha)
            iwv      = f'HIERARCH ESO TEL{self.tel_unit} AMBI IWV AVG', # Integrated water vapour [mm]
            humidity = f'HIERARCH ESO TEL{self.tel_unit} AMBI RHUM', # ambient relative humidity [%]
            # snr      = 'SNR', # Median Signal-to-noise ratio over orders
            spec_rv  = 'HIERARCH ESO OCS OBJ RV',
            ra       = 'RA',      # [deg]
            dec      = 'DEC',     # [deg]
            prog_id  = 'HIERARCH ESO OBS PROG ID',
            file_type    = 'HIERARCH ESO PRO CATG',

        )
    def keys(self):
        return self.dict.keys()
    
    @property
    def unique_keys(self):
        # These keys we store as a single value (not a list)
        return ['obj', 'instr','ra', 'dec', 'prog_id']
    
    
# class EspressoHeader(Header):
    
    
    
    
    

if __name__ == '__main__':
    eso = Header()