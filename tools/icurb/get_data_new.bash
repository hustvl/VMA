export TEMP_RAW_DIR=./data/nyc/temp_raw_tiff
mkdir -p ${TEMP_RAW_DIR}

# download raw data
wget -O Queens.zip https://s3.amazonaws.com/sa-static-customer-assets-us-east-1-fedramp-prod/data.cityofnewyork.us/Map_Downloads/2016_orthos/boro_queens_sp16.zip
unzip -d ./Queens Queens.zip
rm -rf ./Queens.zip
mv ./Queens/*.jp2 ${TEMP_RAW_DIR}
rm -rf ./Queens

wget -O Manhattan.zip https://s3.amazonaws.com/sa-static-customer-assets-us-east-1-fedramp-prod/data.cityofnewyork.us/Map_Downloads/2016_orthos/boro_manhattan_sp16.zip
unzip -d ./Manhattan Manhattan.zip
rm -rf ./Manhattan.zip
mv ./Manhattan/*.jp2 ${TEMP_RAW_DIR}
rm -rf ./Manhattan

wget -O Kings.zip https://s3.amazonaws.com/sa-static-customer-assets-us-east-1-fedramp-prod/data.cityofnewyork.us/Map_Downloads/2016_orthos/boro_brooklyn_sp16.zip
unzip -d ./Kings Kings.zip
rm -rf ./Kings.zip
mv ./Kings/*.jp2 ${TEMP_RAW_DIR}
rm -rf ./Kings

wget -O Richmond.zip https://s3.amazonaws.com/sa-static-customer-assets-us-east-1-fedramp-prod/data.cityofnewyork.us/Map_Downloads/2016_orthos/boro_staten_island_sp16.zip
unzip -d ./Richmond Richmond.zip
rm -rf ./Richmond.zip
mv ./Richmond/*.jp2 ${TEMP_RAW_DIR}
rm -rf ./Richmond

wget -O Bronx.zip https://s3.amazonaws.com/sa-static-customer-assets-us-east-1-fedramp-prod/data.cityofnewyork.us/Map_Downloads/2016_orthos/boro_bronx_sp16.zip
unzip -d ./Bronx Bronx.zip
rm -rf ./Bronx.zip
mv ./Bronx/*.jp2 ${TEMP_RAW_DIR}
rm -rf ./Bronx

# crop raw data
export CROPPED_DIR=./data/nyc/cropped_tiff_1
mkdir -p ${CROPPED_DIR}
python ./tools/icurb/generate_cropped_tiff.py --raw_data_dir ${TEMP_RAW_DIR} --out_dir ${CROPPED_DIR}

rm -rf ${TEMP_RAW_DIR}