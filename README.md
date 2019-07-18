## EC2019 - Competencia de Datos - Despegar

### Clasificador de Imágenes de Hoteles 

#### Ejecución local

##### Descargar los pesos y copiarlos a:
https://drive.google.com/file/d/1nuEaOCjDzf9b1LtChksesBbRdZ0RdWtz/view?usp=sharing (200MB aprox)

 `weights/resnet152_SGD_0.0001_LRSchedReduceLROnPlateau_20ep16bs_1563217792.pth`

##### Para crear el container
docker build -t letyrodri .

##### Para correr el container
Importante! Dado que el clasificador usa pytorch se precisa un subdirectorio extra: `/images/0`.

docker run --mount source=host_path_images,target=/images/0,type=bind letyrodri

##### Para copiar la solucion del container a la maquina host
sacar el container_id haciendo docker 
- docker ps -a 
- docker cp containerid:/app/solution-letyrodri.csv ./solution-letyrodri.csv

