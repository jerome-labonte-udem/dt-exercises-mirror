all: test

test:
	docker-compose down --volumes
	docker-compose build
	docker-compose up

clean:
	docker-compose down --volumes

push:
	docker push courchesnea/dt-middleware:test

run:
	docker run courchesnea/dt-middleware:test

build:
	docker build -t courchesnea/dt-middleware:test .
