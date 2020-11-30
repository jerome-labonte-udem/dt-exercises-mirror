DATA_COLLECTION_DIR=./utils/data_collection
rm -rf temp || true
rm -rf $DATA_COLLECTION_DIR/gym_duckietown || true
rm -rf ./requirements.txt || true
git clone git@github.com:Velythyl/gym-duckietown.git -b object_detection temp && cp temp/requirements.txt ./requirements.txt && cp -r ./temp/src/gym_duckietown $DATA_COLLECTION_DIR/gym_duckietown && rm -rf temp
wget https://drive.google.com/drive/folders/1cTBoKrXJb0kajBGxhuBxJpbKaotHPX7O -P dataset
