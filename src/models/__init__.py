
from .BasicModel import BasicModel
from .SLBR import SLBR


def basic(**kwargs):
	return BasicModel(**kwargs)

def slbr(**kwargs):
    return SLBR(**kwargs)
