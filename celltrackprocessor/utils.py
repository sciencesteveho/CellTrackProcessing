#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions for cell track processor"""

from typing import Any, Dict, List


def list_from_dictvals(input: Dict[Any, Any]) -> List[Any]:
    return [
        input[key]
        for key in input.keys()
    ]
