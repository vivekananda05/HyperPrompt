"""Loss exports for the six prompt-learning families used in the paper."""

from .patch_coop_loss_fn import get_coop_loss_fn
from .pixel_coop_loss_fn import get_pixel_coop_loss_fn
from .patch_kgcoop_loss_fn import get_kgcoop_loss_fn
from .pixel_kgcoop_loss_fn import get_pixel_kgcoop_loss_fn
from .patch_maple_loss_fn import get_maple_loss_fn
from .pixel_maple_loss_fn import get_pixel_maple_loss_fn
from .patch_promptsrc_loss_fn import get_promptsrc_loss_fn
from .pixel_promptsrc_loss_fn import get_pixel_promptsrc_loss_fn
from .patch_promptkd_loss_fn import get_promptkd_loss_fn
from .pixel_promptkd_loss_fn import get_pixel_promptkd_loss_fn
from .patch_mmrl_loss_fn import get_mmrl_loss_fn
from .pixel_mmrl_loss_fn import get_pixel_mmrl_loss_fn
from .hyperprompt_coop_loss_fn import get_hyperprompt_coop_loss_fn
from .hyperprompt_kgcoop_loss_fn import get_hyperprompt_kgcoop_loss_fn
from .hyperprompt_maple_loss_fn import get_hyperprompt_maple_loss_fn
from .hyperprompt_promptsrc_loss_fn import get_hyperprompt_promptsrc_loss_fn
from .hyperprompt_promptkd_loss_fn import get_hyperprompt_promptkd_loss_fn
from .hyperprompt_mmrl_loss_fn import get_hyperprompt_mmrl_loss_fn

__all__ = [
    'get_coop_loss_fn',
    'get_pixel_coop_loss_fn',
    'get_kgcoop_loss_fn',
    'get_pixel_kgcoop_loss_fn',
    'get_maple_loss_fn',
    'get_pixel_maple_loss_fn',
    'get_promptsrc_loss_fn',
    'get_pixel_promptsrc_loss_fn',
    'get_promptkd_loss_fn',
    'get_pixel_promptkd_loss_fn',
    'get_mmrl_loss_fn',
    'get_pixel_mmrl_loss_fn',
    'get_hyperprompt_coop_loss_fn',
    'get_hyperprompt_kgcoop_loss_fn',
    'get_hyperprompt_maple_loss_fn',
    'get_hyperprompt_promptsrc_loss_fn',
    'get_hyperprompt_promptkd_loss_fn',
    'get_hyperprompt_mmrl_loss_fn',
]
