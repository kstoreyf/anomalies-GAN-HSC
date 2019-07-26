from astropy.io import fits

print("Loading")
fn = "data/pdr2_wide_icmod_21.0_21.5_fdfc_bsm_shape.fits"
hdul = fits.open(fn, memmap=True)
print("Data")
data = hdul[1].data
print("Ext")
data_ext = data[data['i_extendedness_value']==1]
print("Writing")
data_ext.writeto("data/pdr2_wide_shape_ext.fits")
print("Written")

