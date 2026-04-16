# Default LIBERO extraction encodes `images` into `libero_all_codes_200`.
# To encode another LIBERO view, override these env vars, e.g.:
# export LIBERO_VIEW_SUBDIR=birdview_images 
# export LIBERO_CODES_SAVE=/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/data_storage/libero_all_birdview_codes_200 \
#   bash scripts/tokenizer/extract_vq_emu3.sh
CUDA_VISIBLE_DEVICES=0 python3 models/tokenizer/emu3_tokenizer.py 0 & 
CUDA_VISIBLE_DEVICES=1 python3 models/tokenizer/emu3_tokenizer.py 1 & 
CUDA_VISIBLE_DEVICES=2 python3 models/tokenizer/emu3_tokenizer.py 2 & 
CUDA_VISIBLE_DEVICES=3 python3 models/tokenizer/emu3_tokenizer.py 3 & 
CUDA_VISIBLE_DEVICES=4 python3 models/tokenizer/emu3_tokenizer.py 4 & 
CUDA_VISIBLE_DEVICES=5 python3 models/tokenizer/emu3_tokenizer.py 5 & 
CUDA_VISIBLE_DEVICES=6 python3 models/tokenizer/emu3_tokenizer.py 6 & 
CUDA_VISIBLE_DEVICES=7 python3 models/tokenizer/emu3_tokenizer.py 7
