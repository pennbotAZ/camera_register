INPUT_DIR=$1
# extract features
colmap feature_extractor  --database_path $INPUT_DIR/db.db --image_path $INPUT_DIR/images --ImageReader.single_camera 1 --ImageReader.camera_model RADIAL
# match features
colmap exhaustive_matcher --database_path $INPUT_DIR/db.db  --SiftMatching.guided_matching 1

# sparse reconstruction
mkdir -p $INPUT_DIR/sparse
colmap mapper --database_path $INPUT_DIR/db.db --image_path $INPUT_DIR/images --output_path $INPUT_DIR/sparse

# undistort images
mkdir $INPUT_DIR/dense
colmap image_undistorter \
    --image_path $INPUT_DIR/images \
    --input_path $INPUT_DIR/sparse/0 \
    --output_path $INPUT_DIR/dense \
    --output_type COLMAP \
    --max_image_size 2000