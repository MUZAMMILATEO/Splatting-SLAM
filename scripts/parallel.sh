#!/bin/bash 

# Change the absolute path first!
DATA_ROOT_DIR="/home/khanm/workfolder/InstantSplat/assets"
OUTPUT_DIR="output_infer"
DATASETS=(sora)
SCENES=(Santorini)
N_VIEWS=(2)

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

# Function: Run task on specified GPU using the Flask API and perform training when needed.
run_on_gpu() {
    local GPU_ID=$1
    local DATASET=$2
    local SCENE=$3
    local N_VIEW=$4
    local gs_train_iter=$5
    local IMG_IDX=$6
    local TMP_FOLDER=$7

    # Define paths
    SOURCE_PATH="${DATA_ROOT_DIR}/${DATASET}/${SCENE}/"
    MODEL_PATH="./${OUTPUT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views"
    IMAGE_PATH="${SOURCE_PATH}images"

    # Create necessary directories
    mkdir -p "${MODEL_PATH}"
    mkdir -p "${IMAGE_PATH}"

    echo "======================================================="
    echo "Processing IMG_IDX=${IMG_IDX} in ${TMP_FOLDER} on GPU ${GPU_ID}"
    echo "======================================================="
    
    echo "Listing contents of ${TMP_FOLDER}:"
    ls -l "${TMP_FOLDER}"

    # (1) Send an inference request to the Flask API
    response=$(curl -s -X POST -H "Content-Type: application/json" \
        -d "{\"img_idx\": \"${IMG_IDX}\", \"source_path\": \"${SOURCE_PATH}\", \"model_path\": \"${MODEL_PATH}\", \"tmp_folder\": \"${TMP_FOLDER}\", \"n_views\": \"${N_VIEW}\", \"co_vis_dsp\": \"true\", \"depth_thre\": \"0.01\", \"infer_video\": \"true\", \"conf_aware_ranking\": \"true\", \"focal_avg\": \"true\"}" \
        http://localhost:5000/infer)
    echo "Response: $response"

    # (2) Train: jointly optimize pose for current image pair
    # If IMG_IDX is a multiple of 7, run training on the cumulative image set.
    if (( IMG_IDX % 7 == 0 )); then
        echo "IMG_IDX ${IMG_IDX} is a multiple of 7. Waiting for any running training tasks to finish..."
        # Wait until no training task is running (lock file not present)
        while [ -f /tmp/training.lock ]; do
            sleep 10
        done

        # Create the lock file to block other training tasks
        touch /tmp/training.lock

        # Optional pause before training
        sleep 30
        
        # ***** Insert GPU memory check here *****
        memory_threshold=7000
        while true; do
            gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU_ID})
            echo "GPU ${GPU_ID} memory used: ${gpu_mem} MB"
            if (( gpu_mem < memory_threshold )); then
                break
            fi
            sleep 10
        done
        # ***** End GPU memory check *****
        
        echo "IMG_IDX ${IMG_IDX} is a multiple of 7. Preparing cumulative image set for training."

        # Rebuild the sorted file list locally
        files_local=($(ls "${TRACK2_DIR}" | sort))
        # Copy all images processed so far (from index 0 to IMG_IDX-1) into IMAGE_PATH
        for (( j=0; j < IMG_IDX; j++ )); do
            cp "${TRACK2_DIR}/${files_local[j]}" "${IMAGE_PATH}/"
        done

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running training on cumulative image set (IMG_IDX=${IMG_IDX})..."
        CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train_slam.py \
            -s "${SOURCE_PATH}" \
            -m "${MODEL_PATH}" \
            -r 1 \
            --n_views ${N_VIEW} \
            --iterations ${gs_train_iter} \
            --pp_optimizer \
            --optim_pose \
            > "${MODEL_PATH}/02_train_${IMG_IDX}.log" 2>&1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed for cumulative set (IMG_IDX=${IMG_IDX}). Log saved in ${MODEL_PATH}/02_train_${IMG_IDX}.log"

        # (3) Render-Video (if needed)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rendering training views for IMG_IDX=${IMG_IDX}..."
        CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render_slam.py \
            -s "${SOURCE_PATH}" \
            -m "${MODEL_PATH}" \
            -r 1 \
            --n_views ${N_VIEW} \
            --iterations ${gs_train_iter} \
            --infer_video \
            > "${MODEL_PATH}/03_render_${IMG_IDX}.log" 2>&1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering completed for IMG_IDX=${IMG_IDX}. Log saved in ${MODEL_PATH}/03_render_${IMG_IDX}.log"
            
        # Delete the cumulative image files after training
        rm -f "${IMAGE_PATH}"/*.png

        # Remove the lock file so that subsequent training tasks can proceed
        rm -f /tmp/training.lock
    fi
    # End training block
}

# Main loop to schedule tasks
total_tasks=$((${#DATASETS[@]} * ${#SCENES[@]} * ${#N_VIEWS[@]}))
current_task=0

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        # Define SOURCE_PATH in the main loop too
        SOURCE_PATH="${DATA_ROOT_DIR}/${DATASET}/${SCENE}/"
        for N_VIEW in "${N_VIEWS[@]}"; do
            echo "Processing task for ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters)"

            # Get sorted list of image files from TRACK2_DIR
            files=($(ls "${TRACK2_DIR}" | sort))
            num_files=${#files[@]}

            for (( i=0; i < num_files - 1; i++ )); do
                IMG_IDX=$((i+1))
                TMP_FOLDER="${SOURCE_PATH}tmp/tmp_${IMG_IDX}"
                mkdir -p "${SOURCE_PATH}tmp"
                mkdir -p "${TMP_FOLDER}"
                # sleep 5 (optional pause)

                echo "Copying image pair ${files[i]} and ${files[i+1]} to ${TMP_FOLDER}"
                cp "${TRACK2_DIR}/${files[i]}" "${TMP_FOLDER}/"
                cp "${TRACK2_DIR}/${files[i+1]}" "${TMP_FOLDER}/"

                # Get available GPU
                GPU_ID=$(get_available_gpu)

                # Run the task in the background
                (run_on_gpu "$GPU_ID" "$DATASET" "$SCENE" "$N_VIEW" "$gs_train_iter" "$IMG_IDX" "$TMP_FOLDER") &

                # Wait briefly before launching the next parallel process
                sleep 1
            done
        done
    done
done

# Wait for all background tasks to complete
wait

echo "======================================================="
echo "All tasks completed! Processed $total_tasks tasks in total."
echo "======================================================="
