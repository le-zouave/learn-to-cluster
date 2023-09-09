cfg_name=cfg_train_lgcn_fashion
config=$LTU_PATH/learn-to-cluster/ltc/lgcn/configs/$cfg_name.py
work_dir=$LTU_PATH/learn-to-cluster/ltc/data/work_dir

export CUDA_VISIBLE_DEVICES=0
#export PYTHONPATH=.

# train
python $LTU_PATH/learn-to-cluster/ltc/lgcn/main.py \
    --config $config \
	--work_dir $work_dir \
    --phase 'train'

# test
#load_from=${work_dir}/${cfg_name}/latest.pth
#python $LTU_PATH/learn-to-cluster/ltc/lgcn/main.py \
#    --config $config \
#    --phase 'test' \
#    --load_from $load_from \
#	--work_dir ${work_dir}/${cfg_name} \
#    --save_output \
#    --force
