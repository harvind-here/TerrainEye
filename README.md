# TerrainEye: Fine-Tuned Image classifier model. 

A Terrain classifier model achieved using Transfer learning of a pre-trained ResNet50v2 model.
This repo contains the trained models, codes, files and datasets of terrain images used.

## Skills:
<img src="https://img.shields.io/badge/Machine Learning-3776AB?style=flat-square&logo=ML&logoColor=white" alt="Machine Learning"> <img src="https://img.shields.io/badge/FineTuning-3776AB?style=flat-square&logo=finetuning&logoColor=white" alt="Fine-Tuning">  <img src="https://img.shields.io/badge/Data Preprocessing-3776AB?style=flat-square&logo=Data&logoColor=white" alt="Data Preprocessing">
# Desired Outcomes:

Terrain Recognition-  land-type and terrain-type recognition from aerial view by a flying vehicle such as drone.



Classification Report:
                   precision    recall  f1-score   support

          Airport       0.97      0.98      0.97       248
         BareLand       0.98      0.93      0.95       202
    BaseballField       1.00      1.00      1.00       147
            Beach       1.00      1.00      1.00       258
           Bridge       1.00      0.99      0.99       238
           Center       0.98      0.95      0.97       177
           Church       0.88      0.98      0.93       162
       Commercial       0.98      0.99      0.98       233
 DenseResidential       0.99      0.98      0.99       274
           Desert       0.99      0.98      0.99       201
         Farmland       1.00      1.00      1.00       256
           Forest       0.98      0.98      0.98       163
       Industrial       0.99      0.95      0.97       262
           Meadow       0.93      0.98      0.95       184
MediumResidential       0.95      0.99      0.97       193
         Mountain       1.00      1.00      1.00       230
             Park       0.97      0.97      0.97       240
          Parking       1.00      1.00      1.00       269
       Playground       1.00      0.99      0.99       254
             Pond       0.99      1.00      0.99       268
             Port       0.99      0.99      0.99       256
   RailwayStation       0.98      0.97      0.97       175
           Resort       0.96      0.90      0.93       190
            River       0.99      1.00      0.99       272
           School       0.94      0.94      0.94       199
SparseResidential       0.99      0.99      0.99       174
           Square       0.97      0.97      0.97       192
          Stadium       0.99      0.95      0.97       172
     StorageTanks       0.99      0.99      0.99       212
          Viaduct       1.00      1.00      1.00       260

         accuracy                           0.98      6561
        macro avg       0.98      0.98      0.98      6561
     weighted avg       0.98      0.98      0.98      6561




Dataset citation: https://arxiv.org/pdf/1608.05167.pdf