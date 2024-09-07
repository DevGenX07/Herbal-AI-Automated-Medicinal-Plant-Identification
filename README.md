# Herbal AI : Automated-Medicinal-Plant-Identification

# ABSTRACT: 
The proper identification of plant species has major benefits for a wide range of stakeholders ranging from forestry services, botanists, taxonomists, physicians, pharmaceutical laboratories, organizations fighting for endangered species, government and the public at large. Consequently, this has fueled an interest in developing automated systems for the recognition of different plant species. A fully automated method for the recognition of medicinal plants using computer vision and machine learning techniques has been presented. In this project we explore feature vectors from both the front and back side of a green leaf along with morphological features to arrive at aunique optimum combination of features that maximizes the identification rate. A database of medicinal plant leaves is created from scanned images of front and back side of leaves of commonly used medicinal plants. Leaves from different medicinal plant species were collected and photographed using a webcam in a laboratory setting. A large number of features were extracted from each leaf such as its length, width, perimeter, area, number of vertices, color, perimeter and area of hull. Several derived features were then computed from these attributes.  The best results were obtained from a random forest classifier using a 10-fold cross-validation technique. With an accuracy of 90.1%, the random forest classifier performed better than other machine learning approaches such as the k-nearest neighbor, naïve Bayes, support vector machines and neural networks. These results are very encouraging and future work will be geared towards using a larger dataset and high-performance computing facilities to investigate the performance of deep learning neural networks to identify medicinal plants used in primary health care. To the best of our knowledge, this work is the first of its kind to have created a unique image dataset for medicinal plants.  It is anticipated that a web-based or mobile computer system for the automatic recognition of medicinal plants will help the local population to improve their knowledge on medicinal plants, help taxonomists to develop more efficient species identification techniques and will also contribute significantly in the protection of endangered species.

# INTRODUCTION
The world bears thousands of plant species, many of which have medicinal values, others are close to extinction, and still others that are harmful to man. Not only are plants an essential resource for human beings, but they form the base of all food chains. The medicinal plants are used mostly in herbal, ayurvedic and folk medicinal manufacturing. Herbal plants are plants that can be used for alternatives to cure diseases naturally. About 80% of people in the world still depend on traditional medicine. Meanwhile, according to herbal plants are plants whose plant parts (leaves, stems, or roots) have properties that can be used as raw materials in making modern medicines or traditional medicines. These medicinal plants are often found in the forest. There are various types of herbal plants that we can know through the identification of these herbs, one of which is using identification through the leaves. and protect plant species, it is crucial to study and classify plants correctly. Combinations of a small subset amounting to 1500 of these plants are used in Herbal medicines of different systems of India. Specifically, commercial Ayurvedic preparations use 500 of these plants. Over 80% of plants used in ayurvedic formulations are collected from the forests and wastelands whereas the remaining are cultivated in agricultural lands. More than 8000 plants of Indian origin have been found to be of medicinal value.

# PROBLEM DEFINITION
Deep learning is one of the major subfields of machine learning framework. Machine learning is the study of design of algorithms, inspired from the model of human brain. Deep learning is becoming more popular in data science fields like robotics, artificial intelligence (AI), audio & video recognition and image recognition. Artificial neural network is the core of deep learning methodologies. Deep learning is supported by various libraries such as Theano, TensorFlow, Caffe, Mx net etc., Keras is one of the most powerful and easy to use python library, which is built on top of popular deep learning libraries like TensorFlow, Theano, etc., for creating deep learning models. Detection of correct medicinal leaves can help botanists, taxonomists and drug manufacturers to make quality drug and can reduce the side effects caused by the wrong drug delivery. To identify the leaves of the plants, a type of artificial neural network called convolutional neural network (CNN) is used. The architecture we used here is Densenet121, which is a convolutional neural network that is a powerful model capable of achieving high accuracies on challenging datasets. To address these challenges, there is a pressing need for an automated system that can accurately and efficiently identify medicinal plants based on their visual characteristics. Leveraging advancements in deep learning and machine learning techniques, this study aims to develop a robust and scalable solution for automated medicinal plant identification. The primary objective is to create a model that can analyze plant images, extract discriminative features, and classify medicinal plant species accurately and rapidly, thereby mitigating the limitations of manual identification methods.By employing a combination of deep convolutional neural networks (CNNs) for feature extraction and machine learning algorithms for classification, this research seeks to create a hybrid system capable of revolutionizing the identification process. This system aims to provide a reliable and time-efficient means of identifying medicinal plants, contributing significantly to the fields of healthcare, pharmaceuticals, and biodiversity conservation.

# METHODOLOGY

  # DATA COLLECTION: 
  Dataset Acquisition: Acquired a diverse dataset containing images of various medicinal plants, encompassing different species and parts of the plants (leaves, flowers, roots). Data Annotation: Manually labeled 
  the dataset with accurate plant species or medicinal part Information for supervised learning.
  # DATA PRE-PROCESSING
  Data Cleaning: Removed duplicates, irrelevant images, and performed augmentation techniques (resizing, cropping, rotations) to enhance dataset variability and balance classes. Image Processing: Standardized 
  images by resizing and normalizing to optimize model performance.
  # Model Development by VGG16:
  Feature Extraction: Utilized transfer learning with pre-trained VGG-16 CNN architecture to extract relevant features from the images.Model Selection and Training: Explored various architectures and 
  hyperparameters, selecting the best-performing model through a train-validation split. Trained the model on the training set and optimized using the validation set. 
  # Model Validation and Testing by VGG16:
  Evaluation Metrics: Utilized metrics such as accuracy, precision, recall, and F1-score to assess the model's performanceTest Set Evaluation: Validated the model's performance on a separate test set to ensure 
  its generalizability. 
  # Model Deployment By CNN: 
  Feature Extraction: Utilized transfer learning with pre-trained CNN architecture to extract relevant features from the images. Model Selection and Training: Explored various architectures and hyperparameters, 
  selecting the best-performing model through a train-validation split. Trained the model on the training set and optimized using the validation set.
  # Model Validation and Testing by CNN:
  Evaluation Metrics: Utilized metrics such as accuracy, precision, recall, and F1-score to assess the model's performance. Test Set Evaluation: Validated the model's performance on a separate test set to ensure 
  its generalizability. 
  # Model Deployment and Image Detection:
  Successfully integrated the trained model. After that User click pictures of plants to detect what type of plant is that. Also, after the successful detection it shows some relevant information about the 
  detected plant like its scientific name, the habitat of the detected plant etc. After that, it also giveslinks to external web pages for detail information about the detected plant.

# PROPOSED METHOD
The proposed method for the automated medicinal plant identification problem involves the use of a hybrid system combining deep convolutional neural networks (CNNs) for feature extraction and machine learning algorithms for classification. The overarching goal is to leverage advancements in deep learning and machine learning techniques to develop a robust and scalable solution. Here's a breakdown of the proposed method:
  # Data Collection and Preprocessing:
  Collect a diverse dataset of medicinal plant images, ensuring it covers various species and visual characteristics.Preprocess the images by standardizing sizes, normalizing pixel values, and applying 
  augmentation techniques to increase the diversity of the training data.
  # Deep Convolutional Neural Network (CNN) Architecture:
  Utilize the Densenet121 architecture as the core convolutional neural network for feature extraction. Fine-tune the pre-trained Densenet121 on the medicinal plant dataset to adapt it to the specific 
  characteristics of the target plants.
  # Transfer Learning:
  Leverage transfer learning to take advantage of the knowledge gained by the Densenet121 model on a large dataset (e.g., ImageNet). Reuse the learned features from Densenet121 and fine-tune the model to 
  specialize in medicinal plant identification.
  # Feature Extraction:
  Extract high-level features from the last few layers of the Densenet121 model. These features should capture the distinctive visual characteristics of medicinal plants.
  # Machine Learning Classifier:
  Integrate a machine learning classifier, such as a support vector machine (SVM) or a random forest, to interpret the extracted features and perform the final classification. Train the classifier on the 
  extracted features, optimizing it for accurate and rapid identification.
  # Model Evaluation and Optimization:
  Split the dataset into training, validation, and testing sets to evaluate the model's performance. Employ metrics such as accuracy, precision, recall, and F1 score to assess the model's effectiveness. Fine-tune 
  hyperparameters and conduct cross-validation to optimize the overall performance.
  # Deployment and Integration:
  Develop an interface for the model to accept images for identification.Deploy the model in a scalable and accessible manner, considering potential integration with healthcare, pharmaceutical, or conservation 
  systems.
  # Continuous Improvement:
  Implement mechanisms for continuous learning and improvement, allowing the model to adapt to new data and improve over time. Monitor the model's performance in real-world scenarios and update it as needed. By 
  combining the strengths of deep learning for feature extraction and traditional machine learning for classification, this hybrid system aims to provide an accurate, efficient, and scalable solution for 
  automated medicinal plant identification.

# RESULT AND DISCUSSION
In the realm of machine learning, the surge in popularity of deep learning, particularly in the form of convolutional neural networks (CNNs), has catalysed transformative advancements across various disciplines. This study represents a pioneering effort to leverage deep learning methodologies, specifically the Densenet121 architecture, for the automated identification of medicinal plants. The integration of machine learning techniques, alongside meticulous data collection and preprocessing, has resulted in a hybrid system poised to revolutionize the identification process, offering implications for healthcare, pharmaceuticals, and biodiversity conservation.

  # Data Collection and Preprocessing: A Foundation for Success:
  The journey begins with the acquisition of a diverse dataset, a digital compendium encapsulating the visual diversity of various medicinal plants. Encompassing different species and plant parts such as leaves, 
  flowers, and roots, this dataset becomes the backbone of the subsequent model development. The manual annotation of this dataset with accurate information on plant species or medicinal parts lays thegroundwork 
  for supervised learning, ensuring that the model learns from a labelled dataset.Data preprocessing, a critical phase in the machine learning pipeline, involves the curation of the dataset to enhance its 
  quality and effectiveness. The removal of duplicates and irrelevant images streamlines the dataset, fostering a cleaner and more efficient learning process. Augmentation techniques, including resizing, 
  cropping, and rotations, contribute to dataset variability and class balance, addressing potential biases that may arise during model training. Standardizing images through resizing and normalization ensures 
  consistency in the input data, a crucial factor in the optimization of model performance.
  # Model Development: Harnessing the Power of Densenet121:
  The core of the deep learning methodology lies in the choice of the neural network architecture. In this study, the Densenet121 architecture, known for its capacity to achieve high accuracies on challenging 
  datasets, was selected as the foundation for model development. The architecture's ability to capture intricate features within images positions it as a powerful tool for the nuanced task of identifying 
  medicinal plants. Transfer learning, a technique where a pre-trained model is used as the starting point for a new task, was employed with the pre-trained VGG-16 CNN architecture. This facilitated feature 
  extraction from the medicinal plant images, enabling the model to leverage knowledge gained from broader image datasets.Model selection and training involved a meticulous exploration of various architectures 
  and hyperparameters. The selection of the best-performing model through a train-validation split, followed by fine-tuning and optimization using the validation set, contributed to the model's ability to 
  generalize well to new and unseen data.
  # Model Validation and Testing: Gauging Performance Metrics:
  The validation and testing phases are critical steps in evaluating the efficacy and reliability of the developed model. Utilizing a range of evaluation metrics, including accuracy, precision, recall, and F1- 
  score, provides a comprehensive understanding of the model's performance across various dimensions.The test set evaluation, conducted on a separate dataset not encountered during training, serves as a robust 
  measure of the model's generalization capabilities. The successful validation on an unseen dataset underscores the model's ability to accurately identify medicinal plants under real-world conditions.
  # Model Deployment and Image Detection: Bringing Identification to the User:
  The successful integration of the trained model into a deployable system marks the culmination of the study. The user-friendly interface allows individuals, from botanists to enthusiasts, to actively 
  participate in the identification process. By simply capturing images of medicinal plants, users initiate the identification process, a testament to the accessibility and practicality of the developed system.
  Upon successful detection, the system provides users with a wealth of information about the identified plant, including its scientific name and habitat. This not only aids in immediate recognition but also 
  contributes to educational initiatives, fostering a deeper understanding of the botanical world. The integration of external links to web pages further enriches the user experience, enabling users to explore 
  detailed information about the detected plant.

# CONCLUSION:
In this study, a deep-learning-based system was proposed to perform a real-time species identification of medicinal plants found in the Borneo region. The proposed system addressed some of the key challenges when training deep learning models, such as small training samples with long-tile class imbalance distribution of the species data. Techniques such as class weighting and the use of focal loss function were applied to improve the learning process of the model. The results showed that the proposed system could significantly improve the performance of the deep learning model by more than 10% accuracy compared to the baseline model. However, performance accuracy was slightly dropped when the system was tested on the actual samples by using the developed mobile application in real time. In the future, we intend to further improve the system’s performance by improving the sample collection of the training data. Furthermore, to make the system more useful, we intend to increase the number of species, as the Borneo region is a high species diversity spot.

In this paper we have implemented a technique for medicinal plant identification using random forest algorithm, an ensemble supervises machine learning algorithm based on color, texture and geometrical features to identify the correct species of medicinal plant. The combination of shape, color and texture features result in correct leaf identification accuracy of 94.54 %. The results shown in this technique are very promising and thus indicate the aptness of this algorithm for medicinal plant identification systems. this work can be extended to a larger number of Plants species with improved accuracy in future.






































