# deep-learning-for-disaster-scene-understanding
Applied deep learning method to understanding the disaster scenes for damage level and disaster type
- Build a novel pos-diaster image dataset which includes 3 multiple diaster data type: tornado, tsunami and earthquake, and also labeled the damage level of object buldings
- Adjusted Faster R-CNN to realized the diaster object detection and give the buiding damage level prediction
  - replaced the orging zf features extractor by ResNet50 and SSD

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
