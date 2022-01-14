# GAL-iD

Galaxy's shape identification with the help of AI

## Start the project

To run the database, the api and the streamlit web application, simply run:
```shell
docker-compose up
```

Then, navigate to [http://127.0.0.1:8501/](http://127.0.0.1:8501/)

### GPU Support

If you want use your GPU with tensorflow, make sure the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) is installed
and simply run:
```shell
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up
```