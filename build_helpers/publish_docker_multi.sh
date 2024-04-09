#!/bin/sh

# The below assumes a correctly setup docker buildx environment

IMAGE_NAME=tradescopeorg/tradescope
CACHE_IMAGE=tradescopeorg/tradescope_cache
# Replace / with _ to create a valid tag
TAG=$(echo "${BRANCH_NAME}" | sed -e "s/\//_/g")
TAG_PLOT=${TAG}_plot
TAG_TRADEAI=${TAG}_tradeai
TAG_TRADEAI_RL=${TAG_TRADEAI}rl
TAG_PI="${TAG}_pi"

PI_PLATFORM="linux/arm/v7"
echo "Running for ${TAG}"
CACHE_TAG=${CACHE_IMAGE}:${TAG_PI}_cache

# Add commit and commit_message to docker container
echo "${GITHUB_SHA}" > tradescope_commit

if [ "${GITHUB_EVENT_NAME}" = "schedule" ]; then
    echo "event ${GITHUB_EVENT_NAME}: full rebuild - skipping cache"
    # Build regular image
    docker build -t tradescope:${TAG} .
    # Build PI image
    docker buildx build \
        --cache-to=type=registry,ref=${CACHE_TAG} \
        -f docker/Dockerfile.armhf \
        --platform ${PI_PLATFORM} \
        -t ${IMAGE_NAME}:${TAG_PI} \
        --push \
        --provenance=false \
        .
else
    echo "event ${GITHUB_EVENT_NAME}: building with cache"
    # Build regular image
    docker pull ${IMAGE_NAME}:${TAG}
    docker build --cache-from ${IMAGE_NAME}:${TAG} -t tradescope:${TAG} .

    # Pull last build to avoid rebuilding the whole image
    # docker pull --platform ${PI_PLATFORM} ${IMAGE_NAME}:${TAG}
    # disable provenance due to https://github.com/docker/buildx/issues/1509
    docker buildx build \
        --cache-from=type=registry,ref=${CACHE_TAG} \
        --cache-to=type=registry,ref=${CACHE_TAG} \
        -f docker/Dockerfile.armhf \
        --platform ${PI_PLATFORM} \
        -t ${IMAGE_NAME}:${TAG_PI} \
        --push \
        --provenance=false \
        .
fi

if [ $? -ne 0 ]; then
    echo "failed building multiarch images"
    return 1
fi
# Tag image for upload and next build step
docker tag tradescope:$TAG ${CACHE_IMAGE}:$TAG

docker build --build-arg sourceimage=tradescope --build-arg sourcetag=${TAG} -t tradescope:${TAG_PLOT} -f docker/Dockerfile.plot .
docker build --build-arg sourceimage=tradescope --build-arg sourcetag=${TAG} -t tradescope:${TAG_TRADEAI} -f docker/Dockerfile.tradeai .
docker build --build-arg sourceimage=tradescope --build-arg sourcetag=${TAG_TRADEAI} -t tradescope:${TAG_TRADEAI_RL} -f docker/Dockerfile.tradeai_rl .

docker tag tradescope:$TAG_PLOT ${CACHE_IMAGE}:$TAG_PLOT
docker tag tradescope:$TAG_TRADEAI ${CACHE_IMAGE}:$TAG_TRADEAI
docker tag tradescope:$TAG_TRADEAI_RL ${CACHE_IMAGE}:$TAG_TRADEAI_RL

# Run backtest
docker run --rm -v $(pwd)/tests/testdata/config.tests.json:/tradescope/config.json:ro -v $(pwd)/tests:/tests tradescope:${TAG} backtesting --datadir /tests/testdata --strategy-path /tests/strategy/strats/ --strategy StrategyTestV3

if [ $? -ne 0 ]; then
    echo "failed running backtest"
    return 1
fi

docker images

docker push ${CACHE_IMAGE}:$TAG
docker push ${CACHE_IMAGE}:$TAG_PLOT
docker push ${CACHE_IMAGE}:$TAG_TRADEAI
docker push ${CACHE_IMAGE}:$TAG_TRADEAI_RL

docker images

if [ $? -ne 0 ]; then
    echo "failed building image"
    return 1
fi
