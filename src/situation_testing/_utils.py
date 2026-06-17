# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

from typing import Dict


def get_pro_value(sensitive_val: Dict):
    if "protected" in sensitive_val.keys():
        return sensitive_val["protected"]
    else:
        raise ValueError("missing 'protected' key in dict value for the sensitive attribute(s)")


def get_neg_value(target_val: Dict):
    if "negative" in target_val.keys():
        return target_val["negative"]
    else:
        raise ValueError("missing 'negative' key in dict value for the target attribute")


#
# EOF
#
