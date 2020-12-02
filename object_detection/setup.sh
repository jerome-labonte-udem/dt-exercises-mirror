DATA_COLLECTION_DIR=./utils/data_collection
DATASET_DIR=./dataset
rm -rf temp || true
rm -rf $DATA_COLLECTION_DIR/gym_duckietown || true
rm -rf ./requirements.txt || true
git clone https://github.com/Velythyl/gym-duckietown.git -b object_detection temp && cp temp/requirements.txt ./requirements.txt && cp -r ./temp/src/gym_duckietown $DATA_COLLECTION_DIR/gym_duckietown && rm -rf temp
pip install -r requirements.txt
mkdir -p $DATASET_DIR
cd dataset
wget https://www.dropbox.com/s/bpd535fzmj1pz5w/duckietown%20object%20detection%20dataset-20201129T162330Z-001.zip?dl=0
mv duckietown\ object\ detection\ dataset-20201129T162330Z-001.zip\?dl=0 duckietown_object_detection_dataset.zip
unzip duckietown_object_detection_dataset.zip -d real_data
cd ../utils/data_collection
./data_transfer.py
cd ../..
echo "setup finished successfully"
