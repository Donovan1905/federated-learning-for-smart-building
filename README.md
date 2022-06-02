# federated-learning-for-smart-building



This project comes from a student research project. The members of the group are the following :

		- Donovan HOANG : donovan.hoang@viacesi.fr
		- Clément AMARY : clement.amary@viacesi.fr
		- Adeline BLUM : adeline.blum@viacesi.fr
		- Luc ANTONI : luc.antoni@viacesi.fr

You can find the corresponding paper in the project root `./Aggregation_algorithms_use_in_FL_for_smart_Buildings.pdf`.

# Installation



## Python package

```bash
pip install -r requirements.txt
```



## Dataset import

Due to the maximum file size of GitHub your have to replace the `./_data` folder by the one provided in the zip archive of the project.

## Run main

```bash
python ./main.py
```



# Folders



**`./learning`** : Data wrangling, separate to local dataset, local models training

**`./aggregation`** : Aggregation algorithm for global models

**`./wrangling`** : All the dataset wrangling (fillingNaN, scaling, encoding...) 



