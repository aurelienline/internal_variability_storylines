__author__ = "Aurélien Liné"
__credits__ = ["Aurélien Liné"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Aurélien Liné"
__email__ = "aurelien.line@research.froeliger.eu"

# Global librairies
import matplotlib.pyplot

# Local librairies
from . import cmip6
from . import util
#from .storylines import storylines
from .plot.maps import maplot, plot_shape
from .tools.field import check_lat_lon_names, field_avg, lon_flip, mask_land
from .tools.time import get_season
from .shli.matplotlib import set_colors, set_cmap

set_colors('vibrant')
#set_cmap('BuRd')

# Plot options (dedicated to LaTeX use)
default_width_pt = 455.24411
pixel_in_inches = 72.
matplotlib.pyplot.rcParams.update({
    'figure.figsize': [default_width_pt/pixel_in_inches, default_width_pt/pixel_in_inches],
    'figure.dpi': pixel_in_inches,
    'savefig.dpi': 100,
    'savefig.transparent': True,
    'savefig.bbox': 'tight',
    'font.size' : 9,
    'axes.labelsize': 7,
    'legend.fontsize': 7,
    #'text.usetex' : True,
    #'text.latex.preamble': (
    #    r"\usepackage{mathpazo}", # mathpazo lmodern
    #),
    #'font.family' : 'mathpazo', # mathpazo lmodern
#    #'text.latex.unicode': True,
#    'axes.unicode_minus' :False,
})
pt = 1. / matplotlib.pyplot.rcParams['figure.dpi']
linewidth = default_width_pt / matplotlib.pyplot.rcParams['figure.dpi']

__all__ = (
    # Sub-packages
    'cmip6',
    'util',
    # Top-level functions
    'check_lat_lon_names',
    'field_avg',
    'get_season',
    'lon_flip',
    'maplot',
    'plot_shape',
    'mask_land',
    'set_colors',
    'set_cmap',
    # Classes
    #'storylines'
    # Exceptions
    # Constants
)
