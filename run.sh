#!/bin/bash

MAX_LAYERS=9
LOG_FILE="temp_run_output.log"

for i in $(seq 2 $MAX_LAYERS); do
    echo ""
    echo "============================================================"
    echo " RUNNING HYBRID VERIFICATION | drop_n = $i"
    echo "============================================================"
    
    # Run python script, merge stderr into stdout, and pipe to tee
    python main_liresnet.py \
        --dataset cifar10_liresnet \
        --batch_size 1 \
        --drop_n $i \
        --epsilon 0.14118 \
        --norm 2 2>&1 | tee $LOG_FILE
    
    # Capture the exit code of the python script (before the tee pipe)
    PYTHON_EXIT_CODE=${PIPESTATUS[0]}
    
    # Check if the output contains standard PyTorch OOM keywords
    if grep -iq "out of memory\|cuda oom\|allocating" $LOG_FILE; then
        echo -e "\n[!] CUDA Out Of Memory (OOM) reached at drop_n = $i."
        echo "    auto_LiRPA cannot handle $i linear layers on this GPU. Stopping sweep."
        break
    fi
    
    # Check if the script failed for any other non-OOM reason
    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        echo -e "\n[!] Process crashed at drop_n = $i with Exit Code $PYTHON_EXIT_CODE. Stopping sweep."
        break
    fi
    
    echo -e "\n[*] Successfully completed drop_n = $i."
done

# Clean up the temporary log file
rm -f $LOG_FILE
echo "Sweep finished."