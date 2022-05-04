# Histopathologic Cancer Detection

This is the course project of PRML 2022. The contributors are:

i) Debdut Saini (B20CS011)
ii) Vedant A. Sontake (B20EE095)
iii) Suborno Biswas (B20EE069)

<h3>WorkFlow:</h3>

The dataset contains almost 2.6 lakhs of training images, so, it is quite difficult task to make a csv file out of it.
So, we used PyTorch library's inbuilt `torch.utils.data.Dataset` and `torch.utils.data.dataloader` to make a dataset to train our neural networks. 

Models we tried:

i. MLP

ii. 3- layered conv network along with fully connected layers (Ref: Cancer diagnosis in histopathological image: CNN based approach)

What we will try:

i. Transfer Learning Based deep neural network training

Since 2.6 lakhs images are difficult to handle, we will choose around 26k images where class distribution will be almost 50%, to get a new dataset. We will preprocess that data, like reshaping the image and normalizing the pixels values for better convergence. Once we get the dataset, the data obtained will be reduced to lower dimension using PCA.The dimensionally reduced data will be computationally efficient, so we can apply various models to it.

ii. SVM can be applied with 'rbf' kernel.

iii. LDA can be applied for classification.

iv. LightGBM and XGBoost/or other boosting algorithm can be applied.

v. Random forest and different ensemble learning methods can be tried.

vi. Decision Tree can be tried, but it is expected to give poor result when compared to Random Forest.

## Note: We can not disclose the code as of now.

