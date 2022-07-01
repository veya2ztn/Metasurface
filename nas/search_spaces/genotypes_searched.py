from mltool.ModelArchi.ModelSearch.genotype import Genotype

PC_DARTS_metas = Genotype(
    normal=[('sep_conv_3x3', 1),('max_pool_3x3', 0),
            ('sep_conv_3x3', 2),('max_pool_3x3', 0),
            ('skip_connect', 2),('avg_pool_3x3', 3),
            ('avg_pool_3x3', 2),('dil_conv_3x3', 3)],
            normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1),('dil_conv_5x5', 0),
            ('dil_conv_3x3', 2),('max_pool_3x3', 1),
            ('skip_connect', 3),('dil_conv_3x3', 2),
            ('skip_connect', 4),('max_pool_3x3', 0)],
            reduce_concat=range(2, 6)
    )

PC_DARTS_metas2 = Genotype(
    normal=[('sep_conv_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('max_pool_3x3', 0),
            ('sep_conv_3x3', 2), ('dil_conv_5x5', 0),
            ('sep_conv_5x5', 3), ('sep_conv_3x3', 2)],
            normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0),
            ('skip_connect', 2), ('max_pool_3x3', 1),
            ('dil_conv_3x3', 2), ('skip_connect', 3),
            ('skip_connect', 4), ('sep_conv_5x5', 3)],
            reduce_concat=range(2, 6))

PC_DARTS_metas_d =Genotype(
    normal=[('sep_conv_3x3', 1), ('deleted', 0),
            ('skip_connect', 2), ('deleted', 0),
            ('sep_conv_3x3', 2), ('dil_conv_5x5', 0),
            ('sep_conv_5x5', 3), ('sep_conv_3x3', 2)],
            normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1), ('deleted', 0),
            ('skip_connect', 2), ('max_pool_3x3', 1),
            ('dil_conv_3x3', 2), ('skip_connect', 3),
            ('skip_connect', 4), ('deleted', 3)],
            reduce_concat=range(2, 6),
    layer_num = 16
    )

PC_DARTS_complex_1=Genotype(
    normal=[('[cplx]sep_conv_3x3', 0), ('[cplx]sep_conv_3x3', 1),
            ('[cplx]sep_conv_5x5', 2), ('[cplx]sep_conv_3x3', 0),
            ('[cplx]sep_conv_3x3', 3), ('avg_pool_3x3', 0),
            ('[cplx]sep_conv_5x5', 4), ('[cplx]sep_conv_3x3', 0)],
            normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1),        ('[cplx]dil_conv_5x5', 0),
             ('[cplx]sep_conv_5x5', 1), ('[cplx]sep_conv_5x5', 2),
             ('skip_connect', 2),       ('[cplx]dil_conv_3x3', 3),
             ('avg_pool_3x3', 1),       ('[cplx]sep_conv_7x7', 3)],
    reduce_concat=range(2, 6),
    layer_num = 8
    )
