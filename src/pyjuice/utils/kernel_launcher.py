import torch
import triton
from typing import Callable, Tuple


class FastJITFunction():
    def __init__(self, fn: Callable):
        self.jit_fn = triton.JITFunction(fn)

        try:
            self.constexpr_ids = [p.num for p in self.jit_fn.params if p.is_constexpr]
            self.constexpr_names = {p.name: p.num for p in self.jit_fn.params if p.is_constexpr}
            self.nonconstexpr_names = [p.name for p in self.jit_fn.params if not p.is_constexpr]
        except AttributeError:
            self.constexpr_ids = self.jit_fn.constexprs
            self.constexpr_names = {self.jit_fn.arg_names[i]: i for i in self.jit_fn.constexprs}
            self.nonconstexpr_names = [self.jit_fn.arg_names[i] for i in range(len(self.jit_fn.arg_names)) if i not in self.jit_fn.constexprs]

        self.constexpr_ids_set = set(self.constexpr_ids)

        self.cache = dict()

    def __getitem__(self, grid: Tuple):
        
        def wrapper(*args, **kwargs):
            signature_list = list()

            for i in self.constexpr_ids:
                if i >= len(args):
                    break
                signature_list.append(args[i])

            for k, v in kwargs.items():
                if k in self.constexpr_names:
                    signature_list.append((self.constexpr_names[k], v))

            if "batch_size" in kwargs:
                signature_list.append(("batch_size", kwargs["batch_size"]))

            grid_length = len(grid)
            grid0, grid1, grid2 = grid[0], grid[1] if grid_length > 1 else 1, grid[2] if grid_length > 2 else 1

            signature = tuple(signature_list)
            if signature in self.cache:
                kernel = self.cache[signature]

                aligned_args = list()
                for i, arg in enumerate(args):
                    if i not in self.constexpr_ids_set:
                        aligned_args.append(arg)

                for k in self.nonconstexpr_names:
                    if k in kwargs:
                        aligned_args.append(kwargs[k])

                kernel[(grid0, grid1, grid2)](*aligned_args)
            else:
                kernel = self.jit_fn[grid](*args, **kwargs)
                self.cache[signature] = kernel

        return wrapper


def triton_jit(fn: Callable):
    return FastJITFunction(fn)
