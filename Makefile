info:
	@echo "clean: removes docker containers and images"
	@echo "build: builds docker image"
	@echo "test: runs pytest in container"
	@echo "deploy: starts container to run on server"
	
clean:
	docker rm happyday
	docker rmi happyday-image

build:
	docker build -t happyday-image .

test: clean
	docker run --rm happyday-image /bin/bash -c "cd happyday-service; pytest"

deploy: clean build test
	docker run -d -p 80:5000 --name happyday happyday-image
