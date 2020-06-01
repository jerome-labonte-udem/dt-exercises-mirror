repo=aidonode-aido2_lf-gym_bridge
# repo=$(shell basename -s .git `git config --get remote.origin.url`)
branch=$(shell git rev-parse --abbrev-ref HEAD)
tag=duckietown/$(repo):$(branch)

build:
	docker build --pull -t $(tag) .

build-no-cache:
	docker build --pull  -t $(tag)  --no-cache .

push: build
	docker push $(tag)

test-data1-direct:
	./dummy_image_filter.py < test_data/in1.json > test_data/out1.json

test-data1-docker:
	docker run -i $(tag) < test_data/in1.json > test_data/out1.json


test-data1-direct-more:
	cat  test_data/in1.json | \
		./dummy_image_filter.py --name node1  | \
		./dummy_image_filter.py --name node2  \
		> test_data/out1.json
