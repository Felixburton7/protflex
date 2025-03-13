#!/bin/bash

# Define output file
OUTPUT_FILE="/home/s_felix/drFelix/src/protflex/AI_context.txt"

# Start writing to output file
{
    echo "Working Directory: $(pwd)"
    echo ""
    echo "File Structure:"
    tree
    echo ""
    echo "Contents of Relevant Files Below (Ignoring Binary Files):"
    echo "---------------------------------------------------------"
    
    find config cli data models training utils -type f ! -name "*.png" -print0 | while IFS= read -r -d '' file; do
        if file "$file" | grep -qE "text|ASCII|UTF-8"; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done

    echo ""
    echo "======================================="
    echo "Extracting First 10 Lines from Data21 (Ignoring Binary Files)"
    echo "======================================="
    echo ""

    # Navigate to data2 directory
    cd /home/s_felix/drFelix/data_NONE || exit 1
    echo "Data Directory: $(pwd)"
    echo ""
    echo "Folder Structure in Data:"
    tree
    echo ""
    echo "Extracting First 10 Lines from Each File in Data (Excluding Binary & pipeline.log):"
    echo "-------------------------------------------------------------------------------------"

    find . -type f ! -name "pipeline.log" ! -name "*.png" -print0 | while IFS= read -r -d '' file; do
        if file "$file" | grep -qE "text|ASCII|UTF-8"; then
            echo "===== FILE: $file ====="
            head -n 3 "$file"
            echo ""
        fi
    done
} > "$OUTPUT_FILE"