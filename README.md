### Домашнее задание №3 (5+3 баллов). Срок выполнения: до 04.11.21.

- Реализовать undercomplete autoencoder (h=100);

       Для запуска undercomplete autoencoder нужно в `configs/model_config.py`
       установить параметр `cfg.autoencoder_type = AEType.undercomplete`, 
       в `configs/train_config.py` установить параметр cfg.experiment_name = 'undercomplete_ae',
       затем запустить файл `executor/executor.py`. Запустится валидация на лучшей эпохе, затем knn и lof, 
       после чего продолжится обучение.
  
        MSE, полученный после 100 эпох обучения:  0.6484451278855529.
  
        KNN на тестовой выборке: 
            лучший k = 100
            лучший AP = 0.10504789744435714
            лучший F1-score = 0.18622568824579536
            precision-recall кривая и confusion matrix в папке `/saved_files/plots/undercomplete_ae/knn/`.
  
        LOF на тестовой выборке: 
            лучший k = 100
            лучший AP = 0.10351424232852682
            лучший F1-score = 0.1857264070896578
            precision-recall кривая и confusion matrix в папке `/saved_files/plots/undercomplete_ae/lof/`.

- Реализовать sparse autoencoder с h=3x32x32
  
        Для запуска overcomplete sparse autoencoder нужно в `configs/model_config.py`
        cfg.autoencoder_type = AEType.overcomplete
        cfg.reg_type = RegType.sparse, 
        в `configs/train_config.py` установить параметр cfg.experiment_name = 'overcomplete_ae_sparse_reg',
        затем запустить файл `executor/executor.py`.

        MSE, полученный после 100 эпох обучения: 1.0028213624712787.

        KNN на тестовой выборке: 
            лучший k = 96
            лучший AP = 0.1694976385837381
            лучший F1-score = 0.23116438356164384
            precision-recall кривая и confusion matrix в папке `/saved_files/plots/overcomplete_ae_sparse_reg/knn/`.

        LOF на тестовой выборке: 
            лучший k = 96
            лучший AP = 0.16261158079500324
            лучший F1-score = 0.23076923076923075
            precision-recall кривая и confusion matrix в папке `/saved_files/plots/overcomplete_ae_sparse_reg/lof/`.

- Сравнить полученные результаты пунктов 1 и 2 с результатами домашних заданий 1 и 2
        

  Ссылка на табличку:
  https://docs.google.com/spreadsheets/d/1T1pJvS-rOqyfDc7TYVEHM5i39yjV8Cb5UHZBYsxmULc/edit?usp=sharing


- (Дополнительно): Реализовать contractive autoencoder и выполнить те же шаги, что для остальных

        Для запуска overcomplete contractive autoencoder нужно в `configs/model_config.py`
        cfg.autoencoder_type = AEType.overcomplete
        cfg.reg_type = RegType.contractive, 
        в `configs/train_config.py` установить параметр cfg.experiment_name = 'overcomplete_ae_contractive_reg',
        затем запустить файл `executor/executor.py`.
        
        MSE, полученный после 100 эпох обучения: 0.6182704493358498.

        KNN на тестовой выборке: 
            лучший k = 96
            лучший AP = 0.11054209234208252
            лучший F1-score = 0.19816009305276513
            precision-recall кривая и confusion matrix в папке `/saved_files/plots/overcomplete_ae_contractive_reg/knn/`.

        LOF на тестовой выборке: 
            лучший k = 94
            лучший AP = 0.10772625557194751
            лучший F1-score = 0.1938424144217136
            precision-recall кривая и confusion matrix в папке `/saved_files/plots/overcomplete_ae_contractive_reg/lof/`.

- Ссылка на папку с пиклами:
    https://drive.google.com/drive/folders/1_cbHpY6BoBWpO7HDuh6-n9ANXSZfZh-I?usp=sharing
  
Все папки с гугл диска нужно добавить в папку `/saved_files/`.
