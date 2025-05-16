from __future__ import annotations

import itertools
from typing import Optional, Iterable


def _make_arg_signature(
        ndim: int, var_symbol: str,
        prefix: Optional[Iterable[str | int] | str | int] = None,
        suffix: Optional[Iterable[str | int] | str | int] = None
) -> str:
    if isinstance(prefix, str | int):
        prefix = [prefix]
    if isinstance(suffix, str | int):
        suffix = [suffix]

    if prefix is None:
        prefix = []
    if suffix is None:
        suffix = []

    args = (f"{var_symbol}{dim + 1}" for dim in range(ndim))

    symbols = itertools.chain(prefix, args, suffix)
    symbols = map(str, symbols)

    return ",".join(symbols)
