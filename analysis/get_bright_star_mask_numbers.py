from astropy.table import Table

from caterpillar import catalog

#base_dir = '/Users/ksf/code/kavli/anomalies-GAN-HSC'
base_dir = '/scratch/ksf293/anomalies'
mask_dir = f'{base_dir}/data/hsc_masks/'

anom_file = f'{base_dir}/data/hsc_catalogs/anomaly_catalog_hsc_full.fits'
anom_cat = Table.read(anom_file)
#anom_brightstar = catalog.filter_through_bright_star_mask(anom_file, mask_dir)
anom_msk = catalog.filter_through_bright_star_mask(
    anom_file, mask_dir, filter_type='inside', output_suffix='masked')

n_total = len(anom_cat)
anoms_total = len(anom_cat[anom_cat['discriminator_score_normalized'] > 3])
print("Total everywhere:", n_total)
print("Anoms everywhere:", anoms_total)

print("Total inside mask:", len(anom_msk))
print("% total inside mask:", len(anom_msk)/n_total)
anoms_inmask = len(anom_msk[anom_msk['discriminator_score_normalized'] > 3])
print("Anomalies inside mask:", anoms_inmask)
print("% Anomalies inside mask:", anoms_inmask/anoms_total)
