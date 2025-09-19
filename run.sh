# fine-grained contrastive loss

CUDA_VISIBLE_DEVICES=0 nohup python train_agent_fgc.py --data MKG-W --num_epoch 1550 --hidden_dim 1024 --lr 5e-4 --dim 256 --max_txt_token 8 --num_head 4 --emb_dropout 0.9 --vis_dropout 0.4 --txt_dropout 0.1 --num_layer_dec 2 --mu 100 > log_MKG-W.txt &

CUDA_VISIBLE_DEVICES=0 nohup python train_agent_fgc.py --data DB15K --num_epoch 1550 --hidden_dim 1024 --lr 5e-4 --dim 256 --max_vis_token 8 --max_txt_token 4 --num_head 2 --emb_dropout 0.6 --vis_dropout 0.3 --txt_dropout 0.1 --num_layer_dec 1 --mu 1000 > log_DB15K.txt &

