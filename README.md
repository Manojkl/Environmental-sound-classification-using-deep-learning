# Environmental sound classification using deep learning

Autonomous robots is a field of artificial intelligence which deals with designing of robots that can perform a task without any intervention from external sources. Autonomous robots will have a huge impact on our lives at home, industries and public places. There is a need for these robots to understand the surrounding environment to exhibit intelligence behaviour. One of the ways robots perceive the surroundings is through sound. The mechanical control technology of robots has grown in considerable rate in recent years. However their ability to perceive the surrounding environment especially through auditory scenes are still in the nascent stage. Acoustic scene classification supplement the image based classification in
many ways such as microphone is omni-directional in nature compared with the limited camera view angle and audio signal require low computation resource and lower bandwidth. A robot fitted with a microphone can listen and interact with humans at any angle by just analyzing the sound signal from the source and it can enhance the application domain of behavioral and assistive autonomous robots. A number of researchers are working on the intelligent sound recognition(ISR) system to provide the ability to understand the real surrounding environment to the robot. The goal of environmental sound classification systems is to analyze human auditory awareness characteristics and embedding such percept ability in autonomous robots.  

# Abstract

The advancement of machine learning approaches increased research in the domain of environmental sound classification tasks. However, the performance of the current traditional models trained with default parameters and deep learning technique trained with a small number of the dataset is unsatisfactory. Existing approaches described in the literature work fail to address the shortage of dataset and improvement techniques. In this research work, two approaches were implemented, firstly six traditional machine learning methods namely K Nearest Neighbor (KNN), Support Vector Machine (SVM), Naive Bayes (NB), Random Forest (RF), Gradient Boosting (GB), and XGBoost (XGBoost) are trained with Mel Frequency Cepstral Coefficient (MFCCs) features and hyperparameter is tuned to improve the performance. Secondly, the AudioSet-pre-trained VGGish architecture is modified and the weights are adjusted to the Urbansound8K and DCASE2018 dataset. The log-mel spectrogram features are used to retrain the modified VGGish network. Freezing more layers of the modified network resulted in increased performance. A traditional gradient boosting method achieves 72.83% testing accuracy on the DCASE2018 dataset and the modified VGGish network achieves 96.56% accuracy on the Urbansound8K dataset.

``` 
@article{Manoj-ESC-DL,
    title = "Environmental sound classification using deep learning",
    author = "Manoj Kolpe Lingappa", 
    month = "01",
    year = "2021",
    url = "https://drive.google.com/file/d/1st5ZnUT3vR0PO387kY6fGzdJDzfZKIIK/view?usp=sharing",
}
``` 
