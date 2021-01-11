To make webpage with images:

1. make folder of thumbnails of desired image set, e.g. at 'thumbnails/cosmos_targets/cosmos_1sig_interesting/{tag}
2. write_archive_table.py: This will write a nice table of objects to put in tables as {tag}.csv, as well as a table formatted for the archive server
3. upload table at https://irsa.ipac.caltech.edu/data/COSMOS/index_cutouts.html, then download fits results with wget command that it produces to the folder webpage/fits_images/{tag}
4. make_archive_thumbnails.py with proper tag
5. web_images.py with proper tag

