# Histopathologic Cancer Detection
## Introduction
Cancer is a disease in which cells multiply uncontrollably and crowd out the normal cells. In biopsies, pathologists
provide the histopathologic assessment of the microscopic structure of the tissue and make final diagnosis by applying visual
inspection of histopathological samples under the microscope and aim to differentiate between normal, and malignant cells.
Manual detection is a tedious, tiring task and most likely to comprise human error, as most parts of the cell are frequently part of
irregular random and arbitrary visual angles. The goal of this project is to identify whether a tumor is benign or malignant in nature, as
malignant tumors are cancerous and should be treated as soon as possible to reduce and prevent further complications.
<p  align="center">
<img src="images/dataset_visualization.jpg" alt="about dataset"><br>
<i>(Image source : Colab file)</i>
</p>

## About the Dataset
The [dataset](https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB) contains 6 gzipped HDF5 files. The files contain histopathologic scans of lymph node sections in the form of multidimensional arrays of scientific or numerical data. The description of the dataset is as follows:
|File Name|Content|Size|
|--|--|--|
|[Camelyonpatch_level_2_split_train_x.h5.gz](https://drive.google.com/file/d/1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2/view)|262144 images|6.1 GB|
|[Camelyonpatch_level_2_split_train_y.h5.gz](https://drive.google.com/file/d/1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG/view)|262144 labels|21 KB|
|[Camelyonpatch_level_2_split_valid_x.h5.gz](https://drive.google.com/file/d/1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3/view)|32768 images|0.8 GB|
|[Camelyonpatch_level_2_split_valid_y.h5.gz](https://drive.google.com/file/d/1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO/view)|32768 labels|3 KB|
|[Camelyonpatch_level_2_split_test_x.h5.gz](https://drive.google.com/file/d/1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_/view)|32768 images|0.8 GB|
|[Camelyonpatch_level_2_split_test_y.h5.gz](https://drive.google.com/file/d/17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP/view)|32768 labels|3 KB|
## Methodology
### Dataset Preparation
- The given dataset was in the form of gzipped HDF5 files. So, in order to perform dataset exploration we first unzipped the file,
uploaded it on google drive, and then loaded the .h5 file into the ‘datasets.PCAM’ function of torchvision (version = 0.12)
library and used the transform attribute to obtain the images in tensor format.
- Next, in order to make the obtained dataset iterable, we passed it through the dataloader function.
- The images obtained were of the dimensions 3x96x96.
- We also converted the images in tensor format to a dataframe with features columns as the pixels and target columns as labels in
order to use the sklearn library models like Random Forest Classifier, LightGBM etc.
### Dataset Preprocessing
- Most of the pixels in the image are redundant and do not contribute substantially. it is required to eliminate them to avoid
unnecessary computational overhead. This can be achieved by compression techniques.
- This is necessary to remove redundancy from the input data which only contributes to the computational complexity of the
network without providing any significant improvements in the result.
- The compression technique implemented by us is image resizing. We resized both the dimensions to half, thereby maintaining
the aspect ratio but reduced the area to 1/4th.
### Dimension Reduction Techniques
- Principal Component Analysis (PCA) : It is one of the most commonly used unsupervised machine learning algorithms that
increases interpretability but at the same time minimizes information loss. It is a statistical procedure that uses an orthogonal
transformation and converts a set of correlated variables to a set of uncorrelated variables.
- Linear Discriminant Analysis (LDA) : It is supervised dimensionality reduction technique which accounts for the intraclass and
interclass variations as well to increase/maintain the separability of the classes after the dimensionality reduction. We tried to do
the LDA but the colab file was crashing as the RAM was getting full.
### Feature Reduction Techniques
- Sequential Feature Selection (SFS) : Attempted but didn’t adopt it as it was taking too much time to run.
Reason: The time complexity factor of SFS technique is O(n!). In our data, n = 6912. Hence the n factorial (n!) of such a big
value increases the time complexity to a huge extent. The run time exceeded 7 hours, which made it not possible to attempt.
## Evaluation of Models
|Models implemented|Accuracy|Specificity|Precision|Recall|
|--|--|--|--|--|
|Transfer Learning|0.85757|0.90867|0.89819|0.80643|
|Convolutional Neural Network|0.74716|0.92434|0.88271|0.56982|
|Multi-layer Perceptron|0.680175|0.86956|0.78983|0.49063|
|Random Forest Classifier(with PCA)|0.69393|0.84991|0.65|0.85|
|LightGBM Classifier(with PCA)|0.7275|0.7578|0.71|0.76|
|Support Vector Machine(with PCA)|0.5233|0.0469|1.00|0.05|
## Result and Analysis
We chose ‘specificity’ as the metric for evaluation as it denotes the chance of correctly classifying negative samples thereby maximizing the surety of positive samples not going undetected. While training the deep learning models (MLP, CNN and Transfer Learning model), the model with highest specificity is saved and it turns out to be the model with lowest validation loss. From the loss vs epoch curves for the Deep Learning frameworks(Linear and CNN), it can be observed that after a certain number of epochs the training loss is decreasing whereas the validation loss is increasing, this implies that the model started to overfit after a certain number of epochs.

<p  align="center">
<img src="images/ROC_TL.jpg" alt="ROC_TL"><br>
<i>(Image source : Colab file)</i>
</p>From the attached ROC curve, we can see that the validation AUC is less than that of training AUC in the case of ‘Transfer Learning’ model. Hence, we can say that the Transfer learning model is not getting overfitted.

From ROC curves and the evaluation table, it is quite evident that the Transfer Learning model is performing the best as the AUC/specificity is coming out to be maximum in that case. Since our evaluation criteria is specificity, we will go with the Transfer learning model, however it can be observed from the evaluation
table above that the transfer learning model is outperformed over other models in terms of accuracy, precision and recall as well.
## Launching the Project
- One needs to save the weights of the model which performed the best. In our case, the weights of the best model are : [Weights](https://drive.google.com/file/d/1YpfoeXjKVwuurWWN2aoelA5hrB_1N3-z/view)
- Run the following command in the terminal : 
  ```
  streamlit run app.py
  ```
## Contributors
|Name|Branch|Institute|
|--|--|--|
|[Vedant A Sontake](https://github.com/Vedant-02)|EE|IIT Jodhpur|
|[Debdut Saini](https://github.com/Debdut0122)|CS|IIT Jodhpur|
|[Suborno Biswas](https://github.com/subornobiswas)|EE|IIT Jodhpur|
## References
We referred to the following research papers and documentations:
- [Cancer diagnosis in histopathological image: CNN based approach](https://www.sciencedirect.com/science/article/pii/S2352914819301133)
- [Transfer learning based histopathologic image classification for breast cancer detection](https://link.springer.com/article/10.1007/s13755-018-0057-x)
- [Pytorch](https://pytorch.org/docs/stable/nn.html)
- [Scikit learn](https://scikit-learn.org/stable/user_guide.html)
