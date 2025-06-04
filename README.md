# 👗 Zalando Fashion Clothing Classification  
<p align="left">
  <img src="https://img.shields.io/badge/Made%20With-Colab-blue?logo=googlecolab&logoColor=white&label=Made%20With" alt="Made with Colab">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/github/repo-size/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification" alt="Repo Size">
  <img src="https://img.shields.io/github/last-commit/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification" alt="Issues">
  <img src="https://img.shields.io/badge/Data%20Visualization-Python-yellow?logo=python" alt="Data Visualization: Python">
  <img src="https://img.shields.io/badge/Version%20Control-Git-orange?logo=git" alt="Version Control: Git">
  <img src="https://img.shields.io/badge/Host-GitHub-black?logo=github" alt="Host: GitHub">
  <img src="https://img.shields.io/badge/Result%20Visualization-GradCAM%20|%20GradCAM++%20|%20ScoreCAM-red?style=flat" alt="Result Visualization: GradCAM, GradCAM++, ScoreCAM">
  <img src="https://img.shields.io/github/forks/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification?style=social" alt="Forks">
  <img src="https://img.shields.io/badge/Project-Completed-brightgreen" alt="Project Status">
</p>

![Dashboard](https://github.com/ShaikhBorhanUddin/Fashion-Image-Classification-Project/blob/main/Images/zalando_title.png?raw=true)

The Zalando Fashion Clothing Classification project aims to automatically classify various types of clothing items using deep learning techniques. This project leverages transfer learning with state-of-the-art convolutional neural network (CNN) architectures—including DenseNet121, MobileNetV2, and ResNet101V2—to accurately identify apparel types such as shirts, hoodies, longsleeves, and sweatshirts, with further distinction based on gender (e.g., hoodies_female, sweatshirt_female).

The goal is to develop a robust multi-class image classification model that can be used in real-world fashion retail and e-commerce scenarios, such as automatic product tagging, visual search systems, and inventory management. The dataset used in this project consists of labeled fashion images sourced from Zalando, cleaned and preprocessed for training, validation, and testing purposes.

Multiple performance metrics such as accuracy, F1-score, precision, recall, loss, confusion matrix, and ROC curves were used to evaluate the models. Additionally, interpretability techniques like Grad-CAM, Grad-CAM++, and Score-CAM were applied to understand model decisions at the visual level. The project is structured to be modular, reproducible, and suitable for integration into larger fashion-tech pipelines.



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
## 🧾 Dataset Overview

The [Dataset](https://www.kaggle.com/datasets/dqmonn/zalando-store-crawl) used in this project consists of fashion clothing images scraped from **Zalando.com**, a popular European e-commerce platform that offers a wide range of clothing, shoes, and accessories for men, women, and kids. The original images, available through a Kaggle dataset, had a high resolution of **606 × 875 pixels**, capturing fine-grained apparel details such as texture, fit, and patterns. To make the dataset compatible with pre-trained deep learning models such as **DenseNet121**, **ResNet101V2**, and **MobileNetV2**, all images were resized to **224 × 224 pixels**, which is the standard input size for most ImageNet-based transfer learning architectures. This resizing was done while preserving the aspect ratio and essential visual features to retain classification accuracy. The preprocessed and **scaled-down version** of the dataset can be accessed via the following [Colab Link](https://drive.google.com/drive/folders/1GzTLc50hYA64IhYc-bplxrFVko6tQkv4) used in this project.  

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/Zalando_dataset_sample.png?raw=true)  

The dataset includes various clothing categories such as **shirts, longsleeves, hoodies, sweatshirts**, and their **gender-specific** counterparts (e.g., *hoodies_female*, *sweatshirt_female*). The images are clean, front-facing product photos with minimal background clutter, making them ideal for training robust image classifiers. Image distribution of the dataset is visualized in the following bar chart.

![Dashboard](https://github.com/ShaikhBorhanUddin/Fashion-Image-Classification-Project/blob/main/Images/dataset_image_distribution.png?raw=true)

## ⚗️ Experiments  

This project evaluates the performance of four pre-trained CNN architectures—**DenseNet121**, **MobileNetV2**, **ResNet101V2**, and **VGG19**—for fashion clothing image classification. Each model was fine-tuned (except VGG19) using transfer learning with customized fully connected layers and trained on the scaled-down Zalando dataset.

**DenseNet121** was configured with a dense layer of 512 neurons and ReLU activation, followed by a softmax output layer for multi-class classification. Initially, the base model was frozen and trained with a learning rate of `0.001` using the Adam optimizer. Later, the base model was unfrozen and fine-tuned with a reduced learning rate of `1e-4`. A dropout of `0.5` was applied to reduce overfitting. The model was trained for `40 epochs` with a batch size of `256`, resulting in approximately **7.56 million trainable parameters**.

**MobileNetV2**, a lightweight model, followed the same architecture setup as DenseNet121 but used a larger batch size of `512` and was trained for `30 epochs`. Its compact design resulted in only **2.9 million parameters**, making it suitable for faster inference and deployment in low-resource environments.

**ResNet101V2** also shared the same architecture structure and training methodology as DenseNet121 but was significantly deeper and more complex, with **43.6 million parameters**. It was trained for `40 epochs` with a batch size of `256`, enabling the model to learn high-level features at greater depth.

**VGG19** was used without fine-tuning (i.e., the base model remained frozen). It was trained for `70 epochs` with a batch size of `256`, resulting in **20.2 million parameters**. Unlike the other models, VGG19 was evaluated solely as a feature extractor, and no additional training on its convolutional base was conducted.

These experiments were designed to assess trade-offs between model complexity, training time, and classification performance across a real-world fashion dataset.


## 📊 Results  

Performance Matrix summary for all tested models are given in the following chart.

| Model         | Accuracy | F1 Score | Loss   | Precision | Recall  |
|---------------|----------|----------|--------|-----------|---------|
| DenseNet121   | 0.7250   | 0.6671   | 0.5354 | 0.7406    | 0.7058  |
| MobileNetV2   | 0.6578   | 0.6535   | 0.8897 | 0.6789    | 0.6463  |
| ResNet101V2   | 0.6883   | 0.6995   | 0.8445 | 0.6929    | 0.6822  |
| VGG19         | 0.6230   | 0.6074   | 0.7610 | 0.6725    | 0.5580  |

Detailed analysis of individual models are discussed below.  

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/densenet121_performance.png?raw=true)  

The performance metrics of the DenseNet101 model, as visualized across training and fine-tuning phases, show clear improvements during fine-tuning. Initially, training and validation accuracy and precision fluctuate moderately but gradually improve over 40 epochs. After fine-tuning begins, both training and validation accuracy (green and red curves) continue to rise, with the fine-tune validation accuracy reaching above 0.75. Similarly, loss decreases significantly in the fine-tuning phase, indicating better convergence. Precision and recall also improve, with fine-tune validation recall nearing 0.75. The most notable improvement is in the F1 score, which steadily increases during fine-tuning for both training and validation sets, indicating a balanced performance in terms of precision and recall. Overall, fine-tuning significantly enhances model generalization and robustness, as seen in the smoother and better-performing validation metrics.

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/mobilenetv2_performance.png?raw=true)  

The MobileNetV2 model's performance metrics demonstrate substantial gains following fine-tuning. Initially, the training and validation accuracy improve but remain somewhat noisy, especially in the validation set. However, after fine-tuning begins, the fine-tune training and validation accuracy (green and red curves) continue to increase, with validation accuracy nearing 0.7. The loss curves show consistent decreases, with fine-tuning significantly reducing loss for both training and validation. Precision and recall also benefit from fine-tuning, though the validation precision shows some stagnation and noise compared to recall, which trends steadily upward. Notably, the F1 score demonstrates a smooth and consistent increase across both training and validation, with fine-tuned validation scores exceeding 0.6, indicating improved overall classification performance. Compared to earlier epochs, fine-tuning clearly enhances the model’s generalization ability and stability.

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/resnet_performance.png?raw=true)  

The performance plots for the ResNet101V2 model reveal notable improvements during the fine-tuning phase. Initially, training and validation accuracy (blue and orange lines) show moderate and somewhat noisy gains, stabilizing around 0.65–0.7. However, after fine-tuning, both the fine-tune training and validation accuracy (green and red lines) significantly improve, with validation accuracy reaching over 0.75. Loss decreases consistently during fine-tuning, reflecting better optimization and convergence. Precision remains strong throughout and further improves with fine-tuning, while recall and F1 score also increase steadily — especially notable in the validation F1 score, which surpasses 0.68. Overall, ResNet101V2 benefits greatly from fine-tuning, achieving strong and stable generalization performance with less variance than earlier training phases.  

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/vgg_accuracy.png?raw=true)  

The accuracy graph for the VGG19 model before fine-tuning shows gradual improvement over 70 epochs but ultimately yields underwhelming results compared to other models. Training accuracy fluctuates between 60% and 70% after the initial sharp rise, while validation accuracy plateaus early and remains consistently around 62–64%, showing minimal improvement despite the extended training duration. The persistent gap and instability in training accuracy without corresponding gains in validation performance suggest limited generalization and potential overfitting. Due to this lackluster performance, especially in contrast with models like ResNet101V2 or DenseNet101, other performance metrics such as loss, precision, recall, and F1-score are omitted from this section as they are unlikely to provide further meaningful insights without fine-tuning.

## 📈 ROC Curve Analysis  

This section presents the ROC curves for the tested models (DenseNet121, MobileNetV2, and ResNet101V2) to visually compare their capability to distinguish between classes in a multi-class fashion classification setting.  

![dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/Zalando_ROC.png?raw=true)

The ROC curves for DenseNet121 (left panel) demonstrate strong class separability, particularly for the 'shirt' and 'longsleeve' classes, which nearly hug the top-left corner — indicating excellent true positive rates with low false positives. The 'hoodies_female' and 'sweatshirt_female' classes perform slightly lower but still show decent AUC behavior. The curves suggest that DenseNet121 is a well-calibrated model overall, achieving a good balance between sensitivity and specificity for most classes. The tight grouping near the top indicates robustness in multi-class discrimination.

The MobileNetV2 ROC curves (middle panel) show slightly weaker performance compared to DenseNet121. The 'shirt' and 'longsleeve' classes again achieve high AUCs, reflecting consistent model strength in recognizing these categories. However, the 'sweatshirt_female' and 'hoodies' curves drop more steeply, showing reduced ability to distinguish these classes from others. This suggests MobileNetV2 struggles more with complex or visually similar clothing categories — likely due to its lightweight architecture, which limits its depth of feature extraction compared to deeper networks.

The ResNet101V2 ROC curves (right panel) are competitive and overall slightly stronger than MobileNetV2, though not as tight as DenseNet121. The 'shirt', 'longsleeve', and 'sweatshirt_female' classes display high AUC values, indicating effective detection. While there is still a modest drop for 'hoodies' and 'hoodies_female', the curves remain relatively steep with limited false positives. ResNet101V2 shows a balanced and generalized classification capacity across multiple categories, thanks to its deeper architecture and rich feature representation.

## 🔢 Confusion Matrix

The confusion matrix provides a detailed breakdown of each model’s classification performance by showing the number of correct and incorrect predictions for every class. It helps to identify specific categories where the model performs well and where it struggles, such as confusing visually similar apparel types. In this project, confusion matrices were generated for DenseNet121, MobileNetV2, and ResNet101V2. These matrices offer insights into inter-class misclassifications and support a deeper understanding of each model's strengths and weaknesses in multi-class fashion clothing classification.

![Dashboard](https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification/blob/main/Images/Zalando_CM.png?raw=true)

The DenseNet121 model shows strong performance, particularly for the 'shirt' class, with 473 correct predictions and minimal confusion. The 'longsleeve' class is also fairly well classified with 153 correct predictions and limited spillover into other categories. However, the 'sweatshirt' and 'sweatshirt_female' classes show considerable misclassification — especially 'sweatshirt', which is often confused with 'hoodies' and 'longsleeve', indicating feature overlap. Notably, many 'sweatshirt_female' instances are being classified as 'hoodies_female', likely due to visual similarities in female upper-body garments.

MobileNetV2 struggles more with class separability compared to DenseNet121. The 'shirt' class again performs well (477 correct), but misclassifications increase across other categories. For instance, 'hoodies' has 24 misclassified as 'hoodies_female', and 'sweatshirt_female' is highly confused with 'hoodies_female' (143 predictions). The 'longsleeve' class suffers from significant ambiguity, with predictions spreading across 'shirt', 'sweatshirt', and even 'hoodies'. This implies that MobileNetV2 has a harder time distinguishing fine-grained clothing details, likely due to a lighter architecture and reduced representational capacity.

ResNet101V2 shows balanced performance across most categories, with stronger diagonal dominance than MobileNetV2, and slightly better than DenseNet121 in some areas. The 'shirt' class is highly accurate with 470 correct predictions. However, 'sweatshirt_female' still faces confusion, notably with 'hoodies_female' (107 misclassifications), and 'sweatshirt' often gets misclassified as 'longsleeve' and 'shirt'. ResNet101V2 does a better job distinguishing between 'hoodies' and 'sweatshirt', likely due to deeper feature extraction layers, but subtle gender-based visual differences (like 'female' labels) remain a consistent challenge across all models.

## 🖼️ Visualizations  

To enhance interpretability and understand model decision-making, this project incorporates advanced model explainability techniques using **Grad-CAM**, **Grad-CAM++**, and **Score-CAM**. These visualization methods highlight important regions in input images that the model focuses on while making predictions. Grad-CAM and Grad-CAM++ generate class-specific heatmaps by leveraging gradients flowing into the final convolutional layers, while Score-CAM provides a gradient-free approach by using the model's confidence scores to produce more robust attention maps. Visualizations were generated for DenseNet121, MobileNetV2, and ResNet101V2 models across various fashion categories. These saliency maps offer valuable insights into the visual reasoning process of each model and validate whether predictions are based on relevant clothing features, such as sleeves, collars, and overall shape.

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

The Zalando Fashion Clothing Classification model can be directly applied in real-world fashion and e-commerce systems to improve efficiency and user experience. For example, online retailers like Zalando, ASOS, or Amazon Fashion can use this model to automatically classify and tag clothing items—such as hoodies, long sleeves, sweatshirts, and gender-specific apparel—when sellers upload new product images. This eliminates the need for manual labeling, reduces human error, and ensures consistency across large inventories. It also enhances the accuracy of product filters and search results, helping customers find desired items faster. In recommendation systems, the model can be used to tailor suggestions based on past purchases or browsing behavior by identifying clothing categories in product images. Additionally, fashion visual search engines can utilize this model to allow users to upload a photo and discover similar-looking items instantly. The model also supports mobile apps that manage digital wardrobes or offer virtual styling advice by detecting and categorizing garments in user photos. In backend operations, it can be integrated into analytics tools to monitor inventory by category, gender, or seasonal demand. Overall, this model offers scalable, automated, and intelligent clothing recognition for a wide range of fashion-tech applications.

## 🛠️ How to Run Locally

To run this project locally, first clone the repository using `git clone https://github.com/ShaikhBorhanUddin/Zalando-Fashion-Clothing-Classification.git` and navigate into the project folder. Create a virtual environment using `python -m venv venv` and activate it (`source venv/bin/activate` on Linux/macOS or `venv\Scripts\activate` on Windows). Install all required dependencies with `pip install -r requirements.txt`. Next, download the preprocessed dataset from the linked Colab notebook and place it in the correct directory as referenced in the scripts or notebooks. Launch Jupyter using `jupyter notebook` and open any model-specific notebook (e.g., `DenseNet121_Fashion_Classification.ipynb`) to train or evaluate. Note that `.h5` model weight files larger than 25MB are not stored on GitHub; you’ll need to manually download them from the shared Colab or Google Drive link provided in the notebook comments.

## ⚠️ Limitations  

The original Zalando dataset was approximately 1.21 GB and contained high-resolution clothing images. To accommodate resource-constrained environments such as Google Colab, a scaled-down version of the dataset (224x224 pixels) was used for training and evaluation. While this made model training feasible, the reduced resolution may have led to a loss of fine-grained features, impacting the models’ ability to distinguish between visually similar categories.

Some overlap was observed between hoodie and sweatshirt samples, which introduced ambiguity into the classification task. This overlap particularly affected visual interpretation tools like Grad-CAM, Grad-CAM++, and Score-CAM, leading to less precise heatmaps and occasional misclassification. Additionally, the dataset was imbalanced, with a higher concentration of shirt and sweatshirt images compared to other categories. This bias may have skewed model predictions and contributed to performance disparities across classes.

Overall, while the models performed reasonably well, these limitations highlight the need for higher-quality labeling, more balanced class distributions, and potentially higher-resolution inputs for further improvement.


## 🧰 Tools and Technology Used  

This project leverages a range of tools and technologies to build, train, evaluate, and visualize deep learning models for fashion clothing classification:

- **Python 3.10** – Core programming language used throughout the project.
- **TensorFlow & Keras** – Used for building and fine-tuning deep learning models such as DenseNet121, MobileNetV2, ResNet101V2, and VGG19.
- **NumPy & Pandas** – For data manipulation, preprocessing, and analysis.
- **Matplotlib & Seaborn** – For visualizing performance metrics, confusion matrices, and ROC curves.
- **OpenCV** – For image processing and resizing.
- **Grad-CAM, Grad-CAM++, and Score-CAM** – For model interpretability and visual explanations.
- **Google Colab Pro** – Used for model training with access to powerful GPUs (A100).
- **Jupyter Notebook** – For modular, interactive experimentation and visualization.
- **Git & GitHub** – For version control and project collaboration.

These tools together provide a robust pipeline for developing and analyzing deep learning models in a real-world image classification context.


## 📄 License

## 📬 Contact



