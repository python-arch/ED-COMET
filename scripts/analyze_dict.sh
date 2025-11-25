#!/bin/bash
# Check what tokenization was used by examining the dictionary

echo "========================================================================"
echo "Analyzing fairseq dictionary to detect tokenization method"
echo "========================================================================"

DICT_PATH="/home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/BART-pretraining/candle/data-bin/gdbart-bin/dict.txt"

echo -e "\n1. Dictionary size:"
wc -l "$DICT_PATH"

echo -e "\n2. First 30 tokens:"
head -30 "$DICT_PATH"

echo -e "\n3. Checking for BPE markers (@@):"
echo "  Count of tokens with '@@':"
grep -c '@@' "$DICT_PATH" || echo "0"
echo -e "\n  Sample tokens with '@@':"
grep '@@' "$DICT_PATH" | head -20 || echo "  None found"

echo -e "\n4. Checking for special tokens:"
grep -E '<s>|<pad>|</s>|<unk>|<mask>' "$DICT_PATH"

echo -e "\n5. Random sample of tokens (lines 1000-1020):"
sed -n '1000,1020p' "$DICT_PATH"

echo -e "\n========================================================================"
echo "Analysis complete!"
echo "========================================================================"
