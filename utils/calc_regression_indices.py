import numpy as np

def calc_regression_indices(presented_lengths, reproduced_lengths, classes_train):
    num_regressions = np.max(classes_train)+1
    slope = np.zeros((num_regressions,))
    intercept = np.zeros((num_regressions,))
    
    for init_state_idx in range(num_regressions):
        x = presented_lengths[classes_train == init_state_idx]
        y = reproduced_lengths[classes_train == init_state_idx]
        params = np.polyfit(x, y, 1)
        slope[init_state_idx] = params[0]
        intercept[init_state_idx] = params[1]
    RI = 1 - slope
    
    return RI, slope, intercept
