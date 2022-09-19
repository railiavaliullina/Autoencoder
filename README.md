# Autoencoder

## About The Project

1) Implementation of the following models:

       - undercomplete autoencoder,
       - overcomplete sparse autoencoder,
       - overcomplete contractive autoencoder.

2) implementation of KNN for anomaly detection task,
3) implementation of Local Outlier Factor (LOF) for anomaly detection task.


## Getting Started

1) Undercomplete Autoencoder;

To run undercomplete autoencoder you need to:

- go to `configs/model_config.py` file and set parameter `cfg.autoencoder_type = AEType.undercomplete`;

- go to `configs/train_config.py` and set parameter cfg.experiment_name = 'undercomplete_ae',

- then to run `executor/executor.py` file. 
       
As a result, there will be validation step on the best epoch, then knn and lof, after which the training will continue.
  
Best achieved result:

        MSE after 100 training epochs:  0.6484451278855529.
  
        KNN on test set: 
            the best k = 100
            the best AP = 0.10504789744435714
            the best F1-score = 0.18622568824579536
            precision-recall curve and confusion matrix are in:
            
                     /saved_files/plots/undercomplete_ae/knn/
  
        LOF on test set: 
            the best k = 100
            the best AP = 0.10351424232852682
            the best F1-score = 0.1857264070896578
            precision-recall curve and confusion matrix are in:
            
                     /saved_files/plots/undercomplete_ae/lof/

2) Overcomplete Sparse Autoencoder


To run overcomplete sparse autoencoder you need to:

- go to `configs/model_config.py` file and set parameters `cfg.autoencoder_type = AEType.overcomplete`, cfg.reg_type = RegType.sparse;

- go to `configs/train_config.py` and set parameter cfg.experiment_name = 'overcomplete_ae_sparse_reg',

- then to run `executor/executor.py` file. 
       
As a result, there will be validation step on the best epoch, then knn and lof, after which the training will continue.

Best achieved result:

        MSE after 100 training epochs: 1.0028213624712787.

        KNN on test set: 
            the best k = 96
            the best AP = 0.1694976385837381
            the best F1-score = 0.23116438356164384
            precision-recall curve and confusion matrix are in:
            
                     /saved_files/plots/overcomplete_ae_sparse_reg/knn/

        LOF on test set: 
            the best k = 96
            the best AP = 0.16261158079500324
            the best F1-score = 0.23076923076923075
            precision-recall curve and confusion matrix are in:
            
                     /saved_files/plots/overcomplete_ae_sparse_reg/lof/


3) Overcomplete Contractive Autoencoder



To run overcomplete sparse autoencoder you need to:

- go to `configs/model_config.py` file and set parameters `cfg.autoencoder_type = AEType.overcomplete`, cfg.reg_type = RegType.contractive;

- go to `configs/train_config.py` and set parameter cfg.experiment_name = 'overcomplete_ae_contractive_reg',

- then to run `executor/executor.py` file. 
       
As a result, there will be validation step on the best epoch, then knn and lof, after which the training will continue.

Best achieved result:
        
        MSE after 100 training epochs: 0.6182704493358498.

        KNN on test set: 
            the best k = 96
            the best AP = 0.11054209234208252
            the best F1-score = 0.19816009305276513
            precision-recall curve and confusion matrix are in:
            
                     /saved_files/plots/overcomplete_ae_contractive_reg/knn/

        LOF on test set: 
            the best k = 94
            the best AP = 0.10772625557194751
            the best F1-score = 0.1938424144217136
            precision-recall curve and confusion matrix are in:
            
                     /saved_files/plots/overcomplete_ae_contractive_reg/lof/
            
