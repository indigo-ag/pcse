# -*- coding: utf-8 -*-
# Copyright (c) 2004-2016 Alterra, Wageningen-UR
# Steven Hoek (steven.hoek@wur.nl), April 2016

"""
Data providers for weather, soil, crop, timer and site data. Also
a class for testing STU suitability for a given crop.

Data providers for CGMS18 are mostly compatible with a CGMS 14 database schema
some differences are implemented here.
"""
from ..cgms14.data_providers import SoilDataIterator as SoilDataIterator14


class SoilDataIterator(SoilDataIterator14):
    """Soil data iterator for CGMS18."""

    tbl_link_sm_grid_cover = "link_cover_grid_smu"
