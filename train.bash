python train.py \
--config ./configs/retrieval_cuhk.yaml \
--output_dir output/CUHK \
--max_epoch 30 \
--batch_size_train 36 \
--batch_size_test 64 \
--init_lr 5e-6 \
--k_test 16 \
--epoch_eval 1

