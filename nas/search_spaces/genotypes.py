from mltool.ModelArchi.ModelSearch.genotype import Genotype
ZERO = ["none"]
BASE = ["max_pool_3x3","avg_pool_3x3","skip_connect"]
BASES= ["[symmetry_keep]max_pool_3x3","[symmetry_keep]avg_pool_3x3","[symmetry_keep]skip_connect"]
PLAIN= [    "sep_conv_3x3",
            "sep_conv_5x5",
            "dil_conv_3x3",
            "dil_conv_5x5",
            "sep_conv_7x7",
            ]
COMPLEX=[    "[cplx]sep_conv_3x3",
             "[cplx]sep_conv_5x5",
             "[cplx]sep_conv_7x7",
             "[cplx]dil_conv_3x3",
             "[cplx]dil_conv_5x5",
            ]
SYMMETRY_P4=[   "[symP4]sep_conv_3x3",
                "[symP4]sep_conv_5x5",
                "[symP4]sep_conv_7x7",
                "[symP4]dil_conv_3x3",
                "[symP4]dil_conv_5x5",
            ]
SYMMETRY_Z2=[   "[symZ2]sep_conv_3x3",
                "[symZ2]sep_conv_5x5",
                "[symZ2]sep_conv_7x7",
                "[symZ2]dil_conv_3x3",
                "[symZ2]dil_conv_5x5",
            ]
SYMMETRY_P4Z2=[ "[symP4Z2]sep_conv_3x3",
                "[symP4Z2]sep_conv_5x5",
                "[symP4Z2]sep_conv_7x7",
                "[symP4Z2]dil_conv_3x3",
                "[symP4Z2]dil_conv_5x5",
            ]

SYMMETRY_auto_P4=[ "[auto][symP4]sep_conv_3x3",
                   "[auto][symP4]sep_conv_5x5",
                   "[auto][symP4]sep_conv_7x7",
                   "[auto][symP4]dil_conv_3x3",
                   "[auto][symP4]dil_conv_5x5",
            ]
SYMMETRY_auto_Z2=[   "[auto][symZ2]sep_conv_3x3",
                "[auto][symZ2]sep_conv_5x5",
                "[auto][symZ2]sep_conv_7x7",
                "[auto][symZ2]dil_conv_3x3",
                "[auto][symZ2]dil_conv_5x5",
            ]
SYMMETRY_auto_P4Z2=[ "[auto][symP4Z2]sep_conv_3x3",
                "[auto][symP4Z2]sep_conv_5x5",
                "[auto][symP4Z2]sep_conv_7x7",
                "[auto][symP4Z2]dil_conv_3x3",
                "[auto][symP4Z2]dil_conv_5x5",
            ]

PCDARTS=PCDARTS_NORMAL= ZERO + BASE + PLAIN
PCDARTS_NOZERO   = BASE + PLAIN
PCDARTS_COMPLEX  = ZERO + BASE + COMPLEX
PCDARTS_ADD_COMPLEX = PCDARTS+PCDARTS_COMPLEX

PCDARTS_ADD_Z2   =  PCDARTS + SYMMETRY_Z2
PCDARTS_ADD_P4Z2 =  PCDARTS + SYMMETRY_P4Z2

PCDARTS_SYMMETRY_auto_P4   = ZERO + BASES + SYMMETRY_auto_P4
PCDARTS_SYMMETRY_auto_P4Z2 = ZERO + BASES + SYMMETRY_auto_P4Z2
PCDARTS_SYMMETRY_auto_Z2   = ZERO + BASES + SYMMETRY_auto_Z2
