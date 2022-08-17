from util import downsample_ts


for num_nodes in [50, 200, 500, 999]:
    downsample_ts('slim_2d_continuous_recapitated_mutated.trees', num_nodes, 0)

downsample_ts('slim_2d_continuous_nocompetition_Ne_2000_sigma_0.5_mate_choice_0.5_max_dist_2_generations_8000_ancients_70_mutated_recapitated_rep_2.trees', 100, 0)
