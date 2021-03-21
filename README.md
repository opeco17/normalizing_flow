# Normalizing Flow
Unofficial implementation of [normalizing flow](http://proceedings.mlr.press/v37/rezende15.pdf).

[Here](https://qiita.com/yuto_hito/items/62192b4dd1cd9cbaa170) is an explanatory article written in Japanese.


## Target Distribution
<img src="https://user-images.githubusercontent.com/46510874/71621211-4dd39700-2c11-11ea-8067-1f1f6b545ac7.png" width="25%"> <img src="https://user-images.githubusercontent.com/46510874/71621232-7360a080-2c11-11ea-983a-5b197d625985.png" width="25%"> <img src="https://user-images.githubusercontent.com/46510874/71621308-ed912500-2c11-11ea-83cd-aeb48eb762d8.png" width="25%">

## Samples from Normalizing Flow
<img src="https://user-images.githubusercontent.com/46510874/71621432-8c1d8600-2c12-11ea-9391-a0fcfe83fc76.png" width="25%"> <img src="https://user-images.githubusercontent.com/46510874/71621451-a22b4680-2c12-11ea-81be-5ee225a73fd8.png" width="25%"> <img src="https://user-images.githubusercontent.com/46510874/71621458-aeaf9f00-2c12-11ea-96ee-d49e57492797.png" width="25%">

# How To Use (with Docker)
```
git clone git@github.com:opeco17/normalizing-flow.git
cd normalizing-flow

# Build and run Docker container
docker-compose up -d

# Training Normalizing Flow
# Executed result will be output to normalizing-flow/src/figure/ at local machine
docker exec -it normalizing-flow_nf_1 python3 train.py
```

# How To Use (without Docker)
```
git clone git@github.com:opeco17/normalizing-flow.git
cd normalizing-flow

# Install requirements
pip3 install -r requirements.txt

# Training Normalizing Flow
# Executed result will be output to normalizing-flow/src/figure/ at local machine
cd src
python3 train.py
```
