CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 20000 main_finetune.py \
--dataset 'nwpu' --model 'vit_base_win_rvsa' --input_size 224 --postfix 'sota' \
--batch_size 64 --epochs 200 --warmup_epochs 5 \
--blr 1e-3  --weight_decay 0.05 --split 28 --tag 0 --exp_num 1 \
--finetune '../Model/pretrained/vit-b-checkpoint-1599.pth'