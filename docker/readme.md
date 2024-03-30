### Some useful commands

Build image with
```
$ docker build -f ./docker/fedora39Dockerfile -t meshlib/fedora39-build-server .
$ docker build -f ./docker/ubuntu22Dockerfile -t meshlib/ubuntu22-build-server .
```

Run a temporary container:
```
$ docker run --rm -it meshlib/fedora39-build-server bash
```
Run a container in background:
```
$ docker run -d --name angry_fedora meshlib/fedora39-build-server tail -f /dev/null
```
[Start an existing container](https://docs.docker.com/engine/reference/commandline/container_start/) and "Attach STDOUT/STDERR and forward signals" (-a), "Attach container's STDIN" (-i):
```
$ docker container start -ai angry_fedora
```
Attach to a running (!) container as root:
```
$ docker exec -u 0 -it angry_fedora bash
```
Show all containers:
```
$ docker ps -a
```
Delete all containers:
```
$ docker rm -f $(docker ps -a -q)
```


