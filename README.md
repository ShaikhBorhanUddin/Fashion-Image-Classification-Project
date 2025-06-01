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

The performance metrics of the DenseNet101 model, as visualized across training and fine-tuning phases, show clear improvements during fine-tuning. Initially, training and validation accuracy and precision fluctuate moderately but gradually improve over 40 epochs. After fine-tuning begins, both training and validation accuracy (green and red curves) continue to rise, with the fine-tune validation accuracy reaching above 0.75. Similarly, loss decreases significantly in the fine-tuning phase, indicating better convergence. Precision and recall also improve, with fine-tune validation recall nearing 0.75. The most notable improvement is in the F1 score, which steadily increases during fine-tuning for both training and validation sets, indicating a balanced performance in terms of precision and recall. Overall, fine-tuning significantly enhances model generalization and robustness, as seen in the smoother and better-performing validation metrics.

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/mobilenetv2_performance.png?raw=true)  

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/resnet_performance.png?raw=true)  

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/vgg_accuracy.png?raw=true)  

## 📈 ROC Curve Analysis  
![dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/Zalando_ROC.png?raw=true)

The ROC curves for DenseNet121 (left panel) demonstrate strong class separability, particularly for the 'shirt' and 'longsleeve' classes, which nearly hug the top-left corner — indicating excellent true positive rates with low false positives. The 'hoodies_female' and 'sweatshirt_female' classes perform slightly lower but still show decent AUC behavior. The curves suggest that DenseNet121 is a well-calibrated model overall, achieving a good balance between sensitivity and specificity for most classes. The tight grouping near the top indicates robustness in multi-class discrimination.

The MobileNetV2 ROC curves (middle panel) show slightly weaker performance compared to DenseNet121. The 'shirt' and 'longsleeve' classes again achieve high AUCs, reflecting consistent model strength in recognizing these categories. However, the 'sweatshirt_female' and 'hoodies' curves drop more steeply, showing reduced ability to distinguish these classes from others. This suggests MobileNetV2 struggles more with complex or visually similar clothing categories — likely due to its lightweight architecture, which limits its depth of feature extraction compared to deeper networks.

The ResNet101V2 ROC curves (right panel) are competitive and overall slightly stronger than MobileNetV2, though not as tight as DenseNet121. The 'shirt', 'longsleeve', and 'sweatshirt_female' classes display high AUC values, indicating effective detection. While there is still a modest drop for 'hoodies' and 'hoodies_female', the curves remain relatively steep with limited false positives. ResNet101V2 shows a balanced and generalized classification capacity across multiple categories, thanks to its deeper architecture and rich feature representation.


## 📉 Confusion Matrix  
![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/Zalando_CM.png?raw=true)

The DenseNet121 model shows strong performance, particularly for the 'shirt' class, with 473 correct predictions and minimal confusion. The 'longsleeve' class is also fairly well classified with 153 correct predictions and limited spillover into other categories. However, the 'sweatshirt' and 'sweatshirt_female' classes show considerable misclassification — especially 'sweatshirt', which is often confused with 'hoodies' and 'longsleeve', indicating feature overlap. Notably, many 'sweatshirt_female' instances are being classified as 'hoodies_female', likely due to visual similarities in female upper-body garments.

MobileNetV2 struggles more with class separability compared to DenseNet121. The 'shirt' class again performs well (477 correct), but misclassifications increase across other categories. For instance, 'hoodies' has 24 misclassified as 'hoodies_female', and 'sweatshirt_female' is highly confused with 'hoodies_female' (143 predictions). The 'longsleeve' class suffers from significant ambiguity, with predictions spreading across 'shirt', 'sweatshirt', and even 'hoodies'. This implies that MobileNetV2 has a harder time distinguishing fine-grained clothing details, likely due to a lighter architecture and reduced representational capacity.

ResNet101V2 shows balanced performance across most categories, with stronger diagonal dominance than MobileNetV2, and slightly better than DenseNet121 in some areas. The 'shirt' class is highly accurate with 470 correct predictions. However, 'sweatshirt_female' still faces confusion, notably with 'hoodies_female' (107 misclassifications), and 'sweatshirt' often gets misclassified as 'longsleeve' and 'shirt'. ResNet101V2 does a better job distinguishing between 'hoodies' and 'sweatshirt', likely due to deeper feature extraction layers, but subtle gender-based visual differences (like 'female' labels) remain a consistent challenge across all models.
## 🖼️ Visualizations  
![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/DenseNet121_viz.png?raw=true)  

The Grad-CAM and Score-CAM visualizations for DenseNet121 reveal effective spatial attention in identifying key garment features for classification. In the Grad-CAM outputs (top row), the model accurately focuses on the central torso area, including hoods and sleeves, to distinguish 'hoodies' and 'hoodies_female', showing localized activation around relevant textures and shapes. The attention maps are centered and tight, suggesting robust feature extraction. The Score-CAM visualizations (bottom row) for 'longsleeve' and 'shirt' also demonstrate clear attention to the full upper body, emphasizing the shape and fit of sleeves and collars. These heatmaps are more uniformly distributed compared to Grad-CAM, capturing broader regions of importance. Overall, both techniques confirm that DenseNet121 not only classifies accurately but also bases decisions on meaningful, interpretable visual cues from the clothing items.  

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/mobilenetv2_viz.png?raw=true)  

The visual explanations from MobileNetV2 show how it interprets clothing categories with Grad-CAM and Grad-CAM++ techniques. In the first row, the Grad-CAM visualizations highlight the central torso and chest area, capturing logos, text, and sleeve length, which helps the model correctly classify 'hoodies' and 'hoodies_female'. The attention is strong but a bit more diffused than DenseNet121, especially around the arms and shoulders. In the Grad-CAM++ examples (second row), the heatmaps demonstrate more precise and fine-grained focus, particularly around the midsection and sleeve cuffs, contributing to the accurate prediction of 'longsleeve' and even challenging examples like a 'hoodie' misclassified as 'hoodies_female'. While Grad-CAM++ improves localization, it also shows sensitivity to subtle features like gender-specific styling. Overall, MobileNetV2 demonstrates solid interpretability, with Grad-CAM++ offering slightly more refined insights than Grad-CAM, though some class confusion still exists for visually similar clothing items.  

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/resnet_viz.png?raw=true)  

This visualization illustrates how the ResNet101V2 model interprets clothing categories using Grad-CAM (first pair) and Grad-CAM++ (remaining three pairs):

- The Grad-CAM result (top-left) for the 'longsleeve' category shows moderate focus on the upper torso and chest area, with some spread across the arms. While effective, the heatmap is less concentrated, reflecting the broader receptive fields in deeper ResNet layers.

In contrast, Grad-CAM++ visualizations offer more localized attention:

- For the 'hoodies' example (top-right), the focus is primarily on the logo and hoodie structure—particularly the hood and chest print—indicating strong alignment with visual cues relevant to classification.

- The 'hoodies_female' sample (bottom-left) shows heat focused on the face, hood, and logo area. While this attention overlaps with gender-specific cues, it may also indicate some reliance on facial attributes.

- The 'sweatshirt' image misclassified as 'hoodies' (bottom-right) reveals attention on the hoodie-style drawstrings and upper torso, showing that the model prioritizes visual hoodie-like features even if the category is technically different.

Overall, ResNet101V2 with Grad-CAM++ produces tighter, more discriminative heatmaps, effectively localizing clothing details like logos, hoods, and necklines. However, its tendency to misclassify sweatshirt-style tops as hoodies suggests some feature overlap confusion, especially when garments share visual attributes.








## 🛍️ Practical Applications

## 🧪 How to Run Locally

## ⚠️ Limitations

## 🧰 Tools and Technology Used

## 📄 License

## 📬 Contact



