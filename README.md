# anomalies-GAN-HSC

In this work, we trained a Wasserstein generative adversarial network (WGAN) to detect anomalous galaxy images in the Hyper Suprime-Cam (HSC) survey.
Check out our short [paper](https://arxiv.org/abs/2012.08082) at the 2020 NeurIPS Machine Learning for Physical Sciences workshop; full paper coming soon.
You can explore our anomalous object sample and find more interesting galaxies at [https://weirdgalaxi.es](https://weirdgalaxi.es).
For any questions or feedback, open an [issue](https://github.com/kstoreyf/anomalies-GAN-HSC/issues) or email [k.sf@nyu.edu](mailto:k.sf@nyu.edu).

This project was initiated at the [Kavli Summer Program in Astrophysics](https://kspa.soe.ucsc.edu/archives/2019) at UC Santa Cruz in July 2019.

## A Quick Overview

GANs are a class of generative model that have been shown to be adept at anomaly detection.
The GAN learns a latent space representation of the entire training set data; essentially, images that are poorly represented by this learned representation are more anomalous with respect to the training set.
Further, the discriminator is trained to distinguish between real and fake (generated) images, and more anomalous real images tend to look more fake, so it is natural at picking out weird images.

We trained our WGAN on nearly 1 million galaxy images from the HSC survey.
The use of the Wasserstein distance, combined with a gradient penalty, stabilizes the training to avoid the failure modes that make GANs infamous.
We set our trained WGAN to generate its best reconstruction of each image from its latent space using a straightforward optimization.
Based on this reconstruction, we assigned each image an anomaly score.
We found that around 9,000 objects score higher than 3-sigma above the mean, and we took these to be our anomalous object sample.

The challenge with anomaly detection in astrophysics is to identify *interesting* anomalies; many of the high-scoring images identified by our WGAN were in fact pipeline errors or noisy observations.
To characterize our anomalies, we trained a convolutional autoencoder (CAE) to reduce the dimensionality of the residual images between the real image and the WGAN reconstruction.
This allowed us to isolate the information relevant to the anomalousness of the images, and perform clustering via a UMAP embedding (similar to T-SNE).
We used this approach to identify many objects of scientific interest, including galaxy mergers, tidal disruption features, and galaxies with extreme star formation.
We performed follow-up observations of several of these objects to confirm their scientific interest; we found one particularly interesting object, a metal-poor dwarf galaxy with potential gaseous outflows!
Keep an eye out for our paper with details on this exciting find.

Our WGAN-CAE-UMAP approach is flexible and scalable, and we hope that it will spur novel discoveries in our increasingly large astronomical surveys.

## Authors

- [Kate Storey-Fisher](https://github.com/kstoreyf)
- [Marc Huertas-Company](https://github.com/mhuertascompany)
- Alexie Leauthaud
- [Nesar Ramachandra](https://github.com/nesar)
- [Francois Lanusse](https://github.com/EiffL)
- [Yifei Luo](https://github.com/yluo54301)
- [Song Huang](https://github.com/dr-guangtou)

