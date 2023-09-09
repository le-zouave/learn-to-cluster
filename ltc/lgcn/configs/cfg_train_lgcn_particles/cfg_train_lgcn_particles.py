import os

# data locations
prefix = '/home/charles/Documents/University/LtUAFM/learn-to-cluster/ltc/datasets/gaussian_particles_1x_bg'

train_name = 'particles_train'
test_name = 'particles_test'
knn = 5

knn_method = 'faiss'

train_data = dict(
    dataset_path=prefix,
    feat_path=os.path.join(prefix, 'features', '{}.bin'.format(train_name)),
    label_path=os.path.join(prefix, 'labels', '{}.meta'.format(train_name)),
    # knn_graph_path=os.path.join(prefix, 'knns', train_name,
    #                         '{}_k_{}.npz'.format(knn_method, knn)),
    knn_graph_path=None,
    knn=knn,
    knn_method=knn_method,
    k_at_hop=[5, 5],
    active_connection=5,
    is_norm_feat=True,
    is_sort_knns=True,
)

test_data = dict(
    feat_path=os.path.join(prefix, 'features', '{}.bin'.format(test_name)),
    label_path=os.path.join(prefix, 'labels', '{}.meta'.format(test_name)),
    # knn_graph_path=os.path.join(prefix, 'knns', test_name,
    #                         '{}_k_{}.npz'.format(knn_method, knn)),
    knn_graph_path=None,
    k_at_hop=[5, 5],
    active_connection=5,
    is_norm_feat=True,
    is_sort_knns=True,
    is_test=True,
)

# model
model = dict(type='lgcn', kwargs=dict(feature_dim=6))

# training args
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer_config = {}

lr_config = dict(
    policy='step',
    step=[1, 2, 3],
)

batch_size_per_gpu = 1
total_epochs = 4
workflow = [('train', 1)]

# testing args
max_sz = 50
step = 0.5
pool = 'avg'

metrics = ['pairwise', 'bcubed', 'nmi']

# misc
workers_per_gpu = 1

checkpoint_config = dict(interval=1)

log_level = 'INFO'
log_config = dict(interval=200, hooks=[
    dict(type='TextLoggerHook'),
])
