# ML application on ESP32-S3-EYE (Image classification for Plant Health Monitoring) 
Hardware Setup

![Setup Block Diagram](images/setup_Block%20Diagram.jpg)


The workflow is as follows:

<img src="images/workflow_Detection.png" width="200">


CNN Model architecture

![Setup Block Diagram](images/model_architecture.png)

Dataset

<img src="images/dataset.png" width="400">

Model Performance

<div align="center">
<img src="images/confusion_matrix.jpg" width="200">
</div>

<div align="center">
<img src="images/train_validation_accuracy.png" width="200">
<img src="images/train_validation_loss.png" width="200">
</div>

<img src="images/performance_metrics.png" width="400">

Images after data augmentation

<img src="images/data_augmented.png" width="400">

Sample Predictions

<img src="images/sample_predict.png" width="400">

RESULTS:

Detection Result of Mint leaves for various categories on LCD

<div align="center">
<img src="images/Fresh.png" width="200">
<img src="images/Dried.png" width="200">
<img src="images/Spoiled.png" width="200">
<img src="images/unknown.png" width="200">
</div>

Serial Port output

<img src="images/serial_port.jpg" width="400">