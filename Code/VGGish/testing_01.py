from keras.models import load_model
import numpy as np
# import model_evaluation_utils as meu
from sklearn import metrics

model = load_model("/home/mkolpe2s/rand/CNN/VGGish/vggish_keras/Performance/vgg_training_all_data_01.h5")

values = np.load("/scratch/mkolpe2s/Test_log_mel_value_all.npy")
test_labels = np.load("/scratch/mkolpe2s/Test_class_value_all.npy")
# values = np.load("/scratch/mkolpe2s/embeddings/test_embedded_value_new_01.npy")

predictions = model.predict_classes(values, verbose=0)
class_value = []
labels = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "enginge_idling", "gun_shot", "jackhammer", "siren", "street_music"]

for  i in predictions:
    
    class_value.append(labels[i])
    # print(i)

print("/home/mkolpe2s/rand/CNN/VGGish/vggish_keras/Performance/vgg_training_all_data_01.h5")
print("/scratch/mkolpe2s/Test_log_mel_value_all.npy")
print("/scratch/mkolpe2s/Test_class_value_all.npy")

report = metrics.classification_report(y_true=test_labels, y_pred=class_value) 
print(report)

# meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=class_value, classes=list(set(test_labels)))