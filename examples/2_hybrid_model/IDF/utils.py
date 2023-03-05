import torch


class Base(torch.nn.Module):
    """
    The base class for modules. That contains a disable round mode
    """

    def __init__(self):
        super().__init__()

    def _set_child_attribute(self, attr, value):
        r"""Sets the module in rounding mode.
        This has any effect only on certain modules if variable type is
        discrete.
        Returns:
            Module: self
        """
        if hasattr(self, attr):
            setattr(self, attr, value)

        for module in self.modules():
            if hasattr(module, attr):
                setattr(module, attr, value)
        return self

    def set_temperature(self, value):
        self._set_child_attribute("temperature", value)

    def enable_hard_round(self, mode=True):
        self._set_child_attribute("hard_round", mode)

    def disable_hard_round(self, mode=True):
        self.enable_hard_round(not mode)