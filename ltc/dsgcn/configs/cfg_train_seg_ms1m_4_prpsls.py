# On 1 TitanX, it takes around 2 hours for training
# metircs (pre, rec, pairwise fscore)
# test on 2 proposal params (gt iop, 0.2-0.8): (99.31, 67.24, 80.19)
# test on 20 proposal params (gt iop, 0.2-0.8): (98.1, 80.87, 88.66)

import os.path as osp
from functools import partial
from ltc.proposals import generate_proposals

use_random_seed = True
featureless=True

# model
model = dict(type='dsgcn',
             kwargs=dict(feature_dim=256,
                         featureless=featureless,
                         stage='seg',
                         reduce_method='no_pool',
                         use_random_seed=use_random_seed,
                         hidden_dims=[512, 64]))

# training args
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer_config = {}

lr_config = dict(
    policy='step',
    step=[15, 24, 28],
)

iter_size = 32
batch_size_per_gpu = 1
test_batch_size_per_gpu = 256
total_epochs = 30
workflow = [('train', 1)]

# misc
workers_per_gpu = 1

checkpoint_config = dict(interval=1)

log_level = 'INFO'
log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'),
])

# post_process
th_pos = -1
th_iou = 1
th_outlier = 0.5
keep_outlier = True

# testing metrics
metrics = ['pairwise', 'bcubed', 'nmi']

# data locations
prefix = './data'
train_name = 'part0_train'
test_name = 'part1_test'

knn_method = 'faiss'
k = 80
step = 0.05
minsz = 3
maxsz = 300

train_thresholds = [0.6, 0.65, 0.7, 0.75]
proposal_params = [
    dict(
        k=k,
        knn_method=knn_method,
        th_knn=th_knn,
        th_step=step,
        minsz=minsz,
        maxsz=maxsz,
    ) for th_knn in train_thresholds
]
feat_path = osp.join(prefix, 'features', '{}.bin'.format(train_name))
label_path = osp.join(prefix, 'labels', '{}.meta'.format(train_name))
proposal_path = osp.join(prefix, 'cluster_proposals')
train_data = dict(wo_weight=False,
                  feat_path=feat_path,
                  label_path=label_path,
                  use_random_seed=use_random_seed,
                  featureless=featureless,
                  th_iop_min=0.3,
                  th_iop_max=0.7,
                  proposal_folders=partial(generate_proposals,
                      params=proposal_params,
                      prefix=prefix,
                      oprefix=proposal_path,
                      name=train_name,
                      dim=model['kwargs']['feature_dim'],
                      no_normalize=False))

test_thresholds = [0.7, 0.75]
proposal_params = [
    dict(
        k=k,
        knn_method=knn_method,
        th_knn=th_knn,
        th_step=step,
        minsz=minsz,
        maxsz=maxsz,
    ) for th_knn in test_thresholds
]
feat_path = osp.join(prefix, 'features', '{}.bin'.format(test_name))
label_path = osp.join(prefix, 'labels', '{}.meta'.format(test_name))
test_data = dict(wo_weight=False,
                 feat_path=feat_path,
                 label_path=label_path,
                 use_random_seed=use_random_seed,
                 use_max_degree_seed=False,
                 featureless=featureless,
                 th_iop_min=0.2,
                 th_iop_max=0.8,
                 proposal_folders=partial(generate_proposals,
                     params=proposal_params,
                     prefix=prefix,
                     oprefix=proposal_path,
                     name=test_name,
                     dim=model['kwargs']['feature_dim'],
                     no_normalize=False))
