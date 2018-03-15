#!/usr/bin/env bash
# make necessary directories
mkdir -p ./cache/mscoco_image_features
wget -O ./cache/mscoco_image_features/inception_v3.zip http://www.cs.jhu.edu/~jhmao/open_source_data/tf_mrnn/inception_v3.zip
unzip ./cache/mscoco_image_features/inception_v3.zip -d ./cache/mscoco_image_features/
rm ./cache/mscoco_image_features/inception_v3.zip