# for nyc dataset
CUDA_VISIBLE_DEVICES=0 python main_pretrain.py --dataset nyc --num_iterations 300 --edim 32 --lr 0.001

# for DC dataset
# CUDA_VISIBLE_DEVICES=0 python main_pretrain.py --dataset DC --num_iterations 300 --edim 32 --lr 0.001

# for BM dataset
# CUDA_VISIBLE_DEVICES=0 python main_pretrain.py --dataset BM --num_iterations 300 --edim 32 --lr 0.001
