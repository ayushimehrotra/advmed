# AdvMed: Detecting Adversarial Attacks in Medical Deep Learning Systems

This is the repository for AdvMed, which was accepted to the International Symposium on Biomedical Imaging (ISBI) 2024 and the International Science and Engineering Fair (ISEF) 2024.

## Dependencies

The following are the packages and their versions needed to run the code

* `torch - 2.1.0`
* `torchvision - 0.16`
* `cleverhans - 4.0`
* `captum`
* `matplotlib`
* `pandas`
* `numpy`

## Download Datasets

`skin-dataset.ipynb` can be used to download the HAM10000 dataset. It will create the appropriate directory that is used by the rest of the files.

`diabetic-dataset.ipynb` can be used to download the alternative verision of the Diabetic Retinoscopy Contest Dataset. It will create the appropriate directory that is used by the rest of the files.

## Create and Run Model

`skin_model.ipynb` is used to create the ResNet50 model and conduct transfer learning using the ImageNet weights on the skin lesion dataset

`diabetic-model.ipynb` is used create the ResNet50 model and conduct transfer learning using the ImageNet weights on the diabetic retinoscopy dataset

## Run Attacks and Defense 

`skin_stats.ipynb` and `diabetic_stats.ipynb` can be used to generate the standard attacks PGD and FGSM, and run the statisical defense using the GRAD and IG 

`skin_noise.ipynb` and `diabetic_noise.ipynb` can be used to generate the attacks and run the sensitivity based defense using the GRAD and IG 

## Run Attack against Defense

`skin_adv2.ipynb` and `diabetic_adv2.ipynb` can be used to generate the Adv2 attack that tailors the perturbed image against the explainable AI method so it looks unperturbed.
