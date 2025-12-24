# big-cats-cv-classification-ops

## Важно
### Убедитись что путь к репозиторию не содержит символов КИРИЛИЦЫ иначе возникнет ошибка из-за особенностей устройства dvc 
#### (PS. так как у меня инициализирован dvc (dvc init -f  --subdir) в поддиректрории корневой папки, где лежит .git, то из-за этого и возникает ошибка. Если бы .dvc был создан в корнейвой папке репозитория, рядом с .git, ошибок из-за кирилицы бы небыло)

## Архитектура cv-classification-dvc-pipeline

├── data/
│   ├──data_from_iNaturalist/observations.csv
│   ├──dataset.csv
│   └──dataset_images/... # папки с классами изображений 
├── src/                     
│   ├── prepare.py           
│   ├── download.py          
│   └── train.py             
├── dvc.yaml                 
└──params.yaml              
