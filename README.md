# anomalies-GAN-HSC

In this work, we trained a Wasserstein generative adversarial network (WGAN) to detect anomalous galaxy images in the Hyper Suprime-Cam (HSC) survey.
Check out our [short paper](https://arxiv.org/abs/2012.08082) at the 2020 NeurIPS Machine Learning for Physical Sciences workshop, and our [full paper](https://arxiv.org/abs/2105.02434) which has been submitted to MNRAS.
You can explore our anomalous object sample and find more interesting galaxies at [https://weirdgalaxi.es](https://weirdgalaxi.es).
For any questions or feedback, open an [issue](https://github.com/kstoreyf/anomalies-GAN-HSC/issues) or email [k.sf@nyu.edu](mailto:k.sf@nyu.edu).

This project was initiated at the [Kavli Summer Program in Astrophysics](https://kspa.soe.ucsc.edu/archives/2019) at UC Santa Cruz in July 2019.

## A quick overview

GANs are a class of generative model that have been shown to be adept at anomaly detection.
The GAN learns a latent space representation of the entire training set data; essentially, images that are poorly represented by this learned representation are more anomalous with respect to the training set.
Further, the discriminator is trained to distinguish between real and fake (generated) images, and more anomalous real images tend to look more fake, so it is natural at picking out weird images.

We trained our WGAN on nearly 1 million galaxy images from the HSC survey.
The use of the Wasserstein distance, combined with a gradient penalty, stabilizes the training to avoid the failure modes that make GANs infamous.
We set our trained WGAN to generate its best reconstruction of each image from its latent space using a straightforward optimization.
Based on the pixel values of this reconstruction by the generator and the features extracted by the discriminator, we assigned each image an anomaly score.
We found that the discriminator selects for more scientifically interesting anomalies, while the generator is adept at finding optical artifacts and noise.
We use the discriminator score to select a high anomaly sample of ~13,000 objects with score higher than 3-sigma above the mean.

The challenge with anomaly detection in astrophysics is to identify *interesting* anomalies; many of the high-scoring images identified by our WGAN were in fact optical artifacts or noisy observations, even among those selected by the discriminator.
To characterize our anomalies, we trained a convolutional autoencoder (CAE) to reduce the dimensionality of the residual images between the real image and the WGAN reconstruction.
This allowed us to isolate the information relevant to the anomalousness of the images, and perform clustering via a UMAP embedding (similar to T-SNE).
We used this approach to identify many anomalous objects of scientific interest, including galaxy mergers, tidal disruption features, and galaxies with extreme star formation.
We performed follow-up observations of several of these objects to confirm their scientific interest; we found one particularly interesting object, which we conclude is likely metal-poor dwarf galaxy with a nearby extremely blue, enriched HII region.

Our WGAN-CAE-UMAP approach is flexible and scalable, and we hope that it will spur novel discoveries in the increasingly large astronomical surveys of the coming years.

## Data access

All of the data used in this work are publicly available.
The notebook [`anomaly_catalog_demo.ipynb`](https://github.com/kstoreyf/anomalies-GAN-HSC/blob/master/notebooks/anomaly_catalog_demo.ipynb) demonstrates the data access and use.
The data files are also described below.

We have provided derived catalogs of the objects in our data set and their anomaly scores as pickle files.
These are available in this repo in the [`anomaly_catalogs`](https://github.com/kstoreyf/anomalies-GAN-HSC/tree/master/anomaly_catalogs) subfolder, and include the full catalog ([`anomaly_catalog_hsc_full.p`](https://github.com/kstoreyf/anomalies-GAN-HSC/tree/master/anomaly_catalogs/anomaly_catalog_hsc_full.p)) as well as the high-anomaly catalog used in the paper ([`anomaly_catalog_hsc_disc3sig.p`](https://github.com/kstoreyf/anomalies-GAN-HSC/tree/master/anomaly_catalogs/anomaly_catalog_hsc_disc3sig.p)).
The high-anomaly selection is based on a discriminator score greater than 3 sigma above the mean (discriminator_score_normalized > 3).
For both of these, the columns are: `[idx,HSC_object_id,ra,dec,discriminator_score_normalized,generator_score_normalized,combined_score_normalized]`.

We also provide the full result catalogs, which include the original images, the GAN reconstructions, and the residuals. 
These are larger, and can be downloadd from Google Drive at [this link](https://drive.google.com/drive/folders/1jPR7Quv_KCT_fJ1sCpny6YKF3T9cAUaa?usp=sharing). 
The high-anomaly catalog (`results_gri_lambda0.3_3sigd.h5`) is 1.1GB and can be downloaded with (on the command line):
```
wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1mYh0GqNxca2sIuyqk4Q8H3aQGy4nh738' -O 'results_gri_lambda0.3_3sigd.h5'
```
The full catalog (`results_gri_lambda0.3.h5`) is 73GB (careful here!!) and can be downloaded with
```
wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1mLGJ9fq1PVLuT0z972kQE_ktGtZH0hUr' -O 'results_gri_lambda0.3.h5'
```
These have the following columns: `['anomaly_scores', 'anomaly_scores_sigma', 'disc_scores', 'disc_scores_sigma', 'gen_scores', 'gen_scores_sigma', 'idxs', 'object_ids', 'reals', 'reconstructed', 'residuals']`.
(Note that `disc_scores_sigma` are the same as `discriminator_score_normalized` in the pickle files, and similar for the other score definitions.

The results of the convolutional autoencoder (CAE) on the high-anomaly sample are also provided, in this repo in the [`anomaly_catalogs`](https://github.com/kstoreyf/anomalies-GAN-HSC/tree/master/anomaly_catalogs) subfolder.
This includes the CAE results on the residuals between the real and reconstructed images ([`autoencoded_disc3sigma_residuals.npy`](https://github.com/kstoreyf/anomalies-GAN-HSC/tree/master/anomaly_catalogs/autoencoded_disc3sigma_residuals.npy)) as well on the real images for comparison ([`autoencoded_disc3sigma_reals.npy`](https://github.com/kstoreyf/anomalies-GAN-HSC/tree/master/anomaly_catalogs/autoencoded_disc3sigma_reals.npy)).
These contain the 64-dimensional latent-space representations, the image indices (idxs), and normalized discriminator scores.


## Authors

- [Kate Storey-Fisher](https://github.com/kstoreyf)
- [Marc Huertas-Company](https://github.com/mhuertascompany)
- Alexie Leauthaud
- [Nesar Ramachandra](https://github.com/nesar)
- [Francois Lanusse](https://github.com/EiffL)
- [Yifei Luo](https://github.com/yluo54301)
- [Song Huang](https://github.com/dr-guangtou)
- [J. Xavier Prochaska](https://github.com/profxj)

