info:
	@echo "clean: removes docker containers and images"
	@echo "build: builds docker image"
	@echo "pytest: runs pytest in container"
	@echo "deploy: starts container to run on server"
	
clean:
	-docker rm --force happyday
	-docker rmi --force happyday-image

build:
	docker build -t happyday-image --build-arg user=$DAV_USER,pwd=$DAV_PASSWORD .

pytest:
	docker run --rm happyday-image /bin/bash -c "cd happyday-service; pytest"

deploy:
	docker run -d -p 80:5000 --name happyday happyday-image

all: clean build test deploy
