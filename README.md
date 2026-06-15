# Fabric Anomaly Detection

In this project, we address fabric anomaly detection using the [MVTec AD 2 dataset](https://arxiv.org/abs/2503.21622). We tackle image anomaly detection with two types of autoencoders:  

- **Convolutional autoencoder (CAE)**  
- **Variational autoencoder (VAE)**

To handle the high resolution images we experiment with two distinct input strategies:

- Sliding window
- Resizing

Models are trained using MSE and SSIM loss functions, and performance is evaluated using AUROC and PR AUC metrics.

CAE weights are available [here](https://liveunibo-my.sharepoint.com/:f:/g/personal/gio_formichella_studio_unibo_it/IgBBzT6lCy6bQqKqJaQpCXuVAVcaWroaAQbdVmzBgoJi8Nc?e=yyKXQQ)

VAE weights are available [here](https://liveunibo-my.sharepoint.com/personal/jacopo_dardini_studio_unibo_it/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjacopo%5Fdardini%5Fstudio%5Funibo%5Fit%2FDocuments%2Fvae%2Ept&parent=%2Fpersonal%2Fjacopo%5Fdardini%5Fstudio%5Funibo%5Fit%2FDocuments&ga=1)