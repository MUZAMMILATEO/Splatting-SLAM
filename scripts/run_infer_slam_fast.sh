#!/bin/bash 

# Change the absolute path first!
DATA_ROOT_DIR="/home/khanm/workfolder/InstantSplat/assets"
OUTPUT_DIR="output_infer"
DATASETS=(
    sora
)
SCENES=(
    Santorini
)
N_VIEWS=(
    2
)

gs_train_iter=1000

# Folder containing sequential track2 images
TRACK2_DIR="/home/khanm/workfolder/Muzammil_SLAM_pointmap/output/track2/color"

# Function to get the id of an available GPU
get_available_gpu() {
    local mem_threshold=7000
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
    $2 < threshold { print $1; exit }
    '
}

# Function: Run task on specified GPU using the Flask API
run_on_gpu() {
    local GPU_ID=$1
    local DATASET=$2
    local SCENE=$3
    local N_VIEW=$4
    local gs_train_iter=$5
    SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
    GT_POSE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
    IMAGE_PATH=${SOURCE_PATH}images
    MODEL_PATH=./${OUTPUT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views

    # Create necessary directories
    mkdir -p ${MODEL_PATH}
    mkdir -p ${IMAGE_PATH}

    echo "======================================================="
    echo "Starting process: ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
    echo "======================================================="

    # --- Begin loop over image pairs from TRACK2_DIR ---
    # Clean the images folder before starting
    rm -f ${IMAGE_PATH}/*.png

    # Get sorted list of image files from TRACK2_DIR
    files=($(ls ${TRACK2_DIR} | sort))
    num_files=${#files[@]}

    # Loop over image pairs (each iteration uses a pair: file[i] and file[i+1])
    for (( i=0; i < num_files - 1; i++ )); do
        IMG_IDX=$((i+1))
        echo "Processing image pair IMG_IDX=${IMG_IDX}: ${files[i]}, ${files[i+1]}"

        # Copy the pair from TRACK2_DIR to the images folder
        cp "${TRACK2_DIR}/${files[i]}" "${IMAGE_PATH}/"
        cp "${TRACK2_DIR}/${files[i+1]}" "${IMAGE_PATH}/"
        
        # (1) Send an inference request to the Flask API
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sending inference request for IMG_IDX=${IMG_IDX}..."
        response=$(curl -s -X POST -H "Content-Type: application/json" \
            -d "{\"img_idx\": \"${IMG_IDX}\", \"source_path\": \"${SOURCE_PATH}\", \"model_path\": \"${MODEL_PATH}\", \"n_views\": \"${N_VIEW}\", \"co_vis_dsp\": \"true\", \"depth_thre\": \"0.01\", \"infer_video\": \"true\", \"conf_aware_ranking\": \"true\", \"focal_avg\": \"true\"}" \
            http://localhost:5000/infer)
        echo "Response: $response"

        # Remove the processed images to prepare for the next iteration
        rm -f ${IMAGE_PATH}/*.png
        
        
        
        
        
        # (2) Train: jointly optimize pose for current image pair
        # If IMG_IDX is a multiple of 5, run training on the cumulative image set.
        if (( IMG_IDX % 7 == 0 )); then
            echo "IMG_IDX ${IMG_IDX} is a multiple of 7. Preparing cumulative image set for training."

            # Copy all images processed so far (from index 0 to i+1) into IMAGE_PATH
            for (( j=0; j <= i+1; j++ )); do
                cp "${TRACK2_DIR}/${files[j]}" "${IMAGE_PATH}/"
            done

            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running training on cumulative image set (IMG_IDX=${IMG_IDX})..."
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train_slam.py \
                -s ${SOURCE_PATH} \
                -m ${MODEL_PATH} \
                -r 1 \
                --n_views ${N_VIEW} \
                --iterations ${gs_train_iter} \
                --pp_optimizer \
                --optim_pose \
                > ${MODEL_PATH}/02_train_${IMG_IDX}.log 2>&1
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed for cumulative set (IMG_IDX=${IMG_IDX}). Log saved in ${MODEL_PATH}/02_train_${IMG_IDX}.log"
            
            # (3) Render-Video (if needed; currently commented out)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rendering training views for IMG_IDX=${IMG_IDX}..."
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render_slam.py \
                -s ${SOURCE_PATH} \
                -m ${MODEL_PATH} \
                -r 1 \
                --n_views ${N_VIEW} \
                --iterations ${gs_train_iter} \
                --infer_video \
                > ${MODEL_PATH}/03_render_${IMG_IDX}.log 2>&1
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering completed for IMG_IDX=${IMG_IDX}. Log saved in ${MODEL_PATH}/03_render_${IMG_IDX}.log"
            
            # Delete the cumulative image files after training
            rm -f ${IMAGE_PATH}/*.png
        fi
            # For IMG_IDX not a multiple of 5, delete just the current pair.

        # Wait briefly before processing the next pair
        sleep 2
    done
    # --- End loop over image pairs ---

    echo "======================================================="
    echo "Task completed: ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
    echo "======================================================="
}

# Main loop to schedule tasks
total_tasks=$((${#DATASETS[@]} * ${#SCENES[@]} * ${#N_VIEWS[@]}))
current_task=0

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do
            current_task=$((current_task + 1))
            echo "Processing task $current_task / $total_tasks"

            # Get available GPU
            GPU_ID=$(get_available_gpu)
            echo "Task completed: ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
            # If no GPU is available, wait and retry
            # while [ -z "$GPU_ID" ]; do
            #     echo "[$(date '+%Y-%m-%d %H:%M:%S')] No GPU available, waiting 60 seconds before retrying..."
            #     sleep 60
            #     GPU_ID=$(get_available_gpu)
            # done

            # Run the task in the background
            (run_on_gpu $GPU_ID "$DATASET" "$SCENE" "$N_VIEW" "$gs_train_iter") &
            # Wait 10 seconds before starting the next task
            sleep 10
        done
    done
done

# Wait for all background tasks to complete
wait

echo "======================================================="
echo "All tasks completed! Processed $total_tasks tasks in total."
echo "======================================================="
