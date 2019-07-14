sudo apt install docker.io
sudo docker pull pytorch/pytorch
sudo docker build . --tag=despegar 
docker run despegar

sudo docker run despegar python classify.py --filename ../weights/densenet_Adam_1e-06_45ep16bs_1562875559weigthed.pth --datadir data

30 mins

https://hub.docker.com/r/pytorch/pytorch/

