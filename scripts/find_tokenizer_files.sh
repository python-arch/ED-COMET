#!/bin/bash
# Script to find BART/GPT-2 BPE tokenizer files

echo "========================================================================"
echo "Searching for BART/GPT-2 BPE tokenizer files..."
echo "========================================================================"

echo ""
echo "1. Checking BART pretraining directory..."
if [ -d "/home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/BART-pretraining" ]; then
    find /home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/BART-pretraining -name "encoder.json" 2>/dev/null | head -5
    find /home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/BART-pretraining -name "vocab.bpe" 2>/dev/null | head -5
    find /home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/BART-pretraining -name "vocab.json" 2>/dev/null | head -5
    find /home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/BART-pretraining -name "merges.txt" 2>/dev/null | head -5
fi

echo ""
echo "2. Checking fairseq cache directories..."
find ~/.cache -name "encoder.json" 2>/dev/null | head -5
find ~/.cache -name "vocab.bpe" 2>/dev/null | head -5

echo ""
echo "3. Checking for gpt2_bpe directory..."
find /home/ahmedjaheen -type d -name "*gpt2*" 2>/dev/null | head -10

echo ""
echo "4. Checking dict.txt in data-bin..."
if [ -f "/home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/BART-pretraining/candle/data-bin/edbart-bin/dict.txt" ]; then
    echo "Found dict.txt:"
    echo "  Location: /home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/BART-pretraining/candle/data-bin/edbart-bin/dict.txt"
    echo "  Number of entries: $(wc -l < /home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/BART-pretraining/candle/data-bin/edbart-bin/dict.txt)"
    echo "  First 10 lines:"
    head -10 /home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/BART-pretraining/candle/data-bin/edbart-bin/dict.txt
fi

echo ""
echo "========================================================================"
echo "Search complete!"
echo "========================================================================"
