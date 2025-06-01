# 👗 Zalando Fashion Clothing Classification  

![Dashboard](https://github.com/ShaikhBorhanUddin/Fashion-Image-Classification-Project/blob/main/Images/zalando_title.png?raw=true)

This project focuses on classifying Zalando fashion articles using deep learning models trained on the Fashion MNIST dataset. It aims to automate clothing categorization and contribute to intelligent fashion recommendation systems.


## 📂 Folder Structure  
The following project structure is maintained in this repository:  
```bash
Zalando Project
|
├── Dataset/ # If it is not available in this folder due to file size limitation of Github, please follow the Colab or Kaggle link
├── Src/ # Source code for dataset upload, training, evaluation, and visualization for every model used
├── Images/ # Output images, graphs, confusion matrix, etc.
├── Requirements.txt # Python dependencies
├── Licence
└── README.md # Outline and overview of this project
```
## 📦 Dataset Overview  
[Dataset](https://www.kaggle.com/datasets/dqmonn/zalando-store-crawl)

![Dashboard](https://github.com/ShaikhBorhanUddin/Fashion-Image-Classification-Project/blob/main/Images/dataset_image_distribution.png?raw=true)

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/Zalando_dataset_sample.png?raw=true)  

## ⚗️ Experiments

## 📊 Results  

| Model         | Accuracy | F1 Score | Loss   | Precision | Recall  |
|---------------|----------|----------|--------|-----------|---------|
| DenseNet121   | 0.7250   | 0.6671   | 0.5354 | 0.7406    | 0.7058  |
| MobileNetV2   | 0.6578   | 0.6535   | 0.8897 | 0.6789    | 0.6463  |
| ResNet101V2   | 0.6883   | 0.6995   | 0.8445 | 0.6929    | 0.6822  |
| VGG19         | 0.6230   | 0.6074   | 0.7610 | 0.6725    | 0.5580  |

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/densenet121_performance.png?raw=true)  

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/mobilenetv2_performance.png?raw=true)  

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/resnet_performance.png?raw=true)  

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/vgg_accuracy.png?raw=true)  

## 📈 ROC Curve Analysis  
![dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/Zalando_ROC.png?raw=true)

## 📉 Confusion Matrix  
![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/Zalando_CM.png?raw=true)

The DenseNet121 model shows strong performance, particularly for the 'shirt' class, with 473 correct predictions and minimal confusion. The 'longsleeve' class is also fairly well classified with 153 correct predictions and limited spillover into other categories. However, the 'sweatshirt' and 'sweatshirt_female' classes show considerable misclassification — especially 'sweatshirt', which is often confused with 'hoodies' and 'longsleeve', indicating feature overlap. Notably, many 'sweatshirt_female' instances are being classified as 'hoodies_female', likely due to visual similarities in female upper-body garments.

MobileNetV2 struggles more with class separability compared to DenseNet121. The 'shirt' class again performs well (477 correct), but misclassifications increase across other categories. For instance, 'hoodies' has 24 misclassified as 'hoodies_female', and 'sweatshirt_female' is highly confused with 'hoodies_female' (143 predictions). The 'longsleeve' class suffers from significant ambiguity, with predictions spreading across 'shirt', 'sweatshirt', and even 'hoodies'. This implies that MobileNetV2 has a harder time distinguishing fine-grained clothing details, likely due to a lighter architecture and reduced representational capacity.

ResNet101V2 shows balanced performance across most categories, with stronger diagonal dominance than MobileNetV2, and slightly better than DenseNet121 in some areas. The 'shirt' class is highly accurate with 470 correct predictions. However, 'sweatshirt_female' still faces confusion, notably with 'hoodies_female' (107 misclassifications), and 'sweatshirt' often gets misclassified as 'longsleeve' and 'shirt'. ResNet101V2 does a better job distinguishing between 'hoodies' and 'sweatshirt', likely due to deeper feature extraction layers, but subtle gender-based visual differences (like 'female' labels) remain a consistent challenge across all models.
## 🖼️ Visualizations  
![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/DenseNet121_viz.png?raw=true)  

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/mobilenetv2_viz.png?raw=true)  

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/resnet_viz.png?raw=true)  

## 🛍️ Practical Applications

## 🧪 How to Run Locally

## ⚠️ Limitations

## 🧰 Tools and Technology Used

## 📄 License

## 📬 Contact



