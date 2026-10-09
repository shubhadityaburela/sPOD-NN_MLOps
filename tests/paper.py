"""The paper's configuration for the synthetic case: the golden values exist only for these inputs."""

TRAIN_MU = [0.1, 0.15, 0.2, 0.25, 0.3]
TEST_MU = [0.23]
N_FRAMES = 2
SPOD_MODES = 8  # modes kept per sPOD frame
POD_MODES = 18  # 2 frames x 8 modes + 2 shifts: as many outputs as sPOD-NN
SPOD_ITER = 300
