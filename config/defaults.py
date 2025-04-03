from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DEVICE = "cuda"
_C.SEED = 0
_C.DATA_TYPE = 'float'  # or double

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.NAME = "Unet"  # Model name
_C.MODEL.STEPS = 1000

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.SIZE_TRAIN = 32
_C.DATA.NAME = 'cartoon'


# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.MAX_EPOCH = 2

# -----------------------------------------------------------------------------
# INFERENCE
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.BATCH_SIZE = 1
_C.INFERENCE.MODEL_PATH = r'output\model.pth'
_C.INFERENCE.DATA_PATH = r''

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()


# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# OUTPUT
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.DIR = "outputs"

# -----------------------------------------------------------------------------
# LOG
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.DIR = "./log"
_C.LOG.ITER_INTERVAL = 1
_C.LOG.EPOCH_INTERVAL = 10
_C.LOG.OUTPUT_TO_FILE = True # 是否输出到文件，默认输出到控制台
_C.LOG.PREFIX = "default" # 输出到文件的命名前缀