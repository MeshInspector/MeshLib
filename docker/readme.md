### Some usefull commands

Build image with
```
$ docker build -f ./docker/fedoraDockerfile -t meshrus/fedora-build-server .
$ docker build -f ./docker/ubuntuDockerfile -t meshrus/ubuntu-build-server .
```

[Start container](https://docs.docker.com/engine/reference/commandline/container_start/) and "Attach STDOUT/STDERR and forward signals" (-a), "Attach container's STDIN" (-i):
```
$ docker container start -ai angry_fedora
```
Attach to running (!) container as root:
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
