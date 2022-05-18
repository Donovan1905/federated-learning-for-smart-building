# federated-learning-for-smart-building



# Installation



## Python package

```shell
pip install -r requirements.txt
```



## Postgresql

First download sql dump dataset from : https://www.data.gouv.fr/fr/datasets/r/5411ee53-8ff1-4dfd-9908-4af53513e7de

Then install and configure postgresql and postgis:

```shell
sudo apt install postgresql postgis
```

To configure the database run the `./_data/quickstart.sql` and `./_data/ext.sql` scripts :

```shell
psql -U postgres -h localhost -f <path_to_quickstart_script>
psql -U postgres -h localhost -f <path_to_ext_script>
```

Import sql dump with the following command : 

```shell
psql -U postgres -h localhost -d bdnb -f <path_to_dump>
```



# Folders



**`./learning`** : Data wrangling, separate to local dataset, local models training

**`./aggregation`** : Aggregation algorithm for global models