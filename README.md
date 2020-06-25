# deep-learning-for-disaster-scene-understanding
Applied deep learning method to understanding the disaster scenes for damage level and disaster type

## "models": includes 4 trained models
- Diaster Type detector with ResNet50
- Diaster Type detector with SSD
- Damage Level detector with ResNet50
- Damage Level detector with SSD
## "ground_truth"
Testing ground turth files in .csv format
## "inference_results.py"
- Load trained model for different purpose (Disater Type/Damage Level)
- Detect the interesting region with bounding boxes
- Predict relative class
- Return class information in .csv file
## "evaluate.py"
- Load inference results (.csv)
- Plot ROC curve and Precision-Recall curve 
