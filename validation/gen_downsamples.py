from util import downsample_ts


for num_nodes in [200, 500, 999]:
    downsample_ts('slim_2d_continuous_recapitated_mutated.trees', num_nodes, 0)
