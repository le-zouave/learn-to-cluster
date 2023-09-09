config=$LTU_PATH/learn-to-cluster/ltc/lgcn/configs/cfg_test_lgcn_fashion.py
load_from=$LTU_PATH/learn-to-cluster/ltc/data/pretrained_models/pretrained_lgcn_fashion.pth
work_dir=$LTU_PATH/learn-to-cluster/ltc/lgcn/test_lgcn_fashion_results/

export CUDA_VISIBLE_DEVICES=0
#export PYTHONPATH=.

python $LTU_PATH/learn-to-cluster/ltc/lgcn/main.py \
    --config $config \
    --phase 'test' \
    --load_from $load_from \
	--work_dir $work_dir \
    --save_output \
    --force
