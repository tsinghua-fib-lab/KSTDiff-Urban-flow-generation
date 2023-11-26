# for nyc dataset
CUDA_VISIBLE_DEVICES=1 python main.py --num_iterations 500 --dataset nyc --lr 0.0001 --train_guidance_every_epochs 1

# for DC dataset
# CUDA_VISIBLE_DEVICES=1 python main.py --num_iterations 200 --dataset DC --lr 0.005 --train_guidance_every_epochs 10

# for BM dataset
# CUDA_VISIBLE_DEVICES=1 python main.py --num_iterations 200 --dataset BM --lr 0.0001 --train_guidance_every_epochs 100
