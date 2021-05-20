from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

star_2  = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('skip_connect', 3), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6))
star_3  = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
star_4  = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 3), ('skip_connect', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
star_5  = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 3), ('avg_pool_3x3', 3), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 3), ('skip_connect', 1), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))
star_6  = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('max_pool_3x3', 3), ('avg_pool_3x3', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0), ('skip_connect', 3), ('max_pool_3x3', 2), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))
star_7  = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('max_pool_3x3', 3), ('avg_pool_3x3', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0), ('skip_connect', 3), ('max_pool_3x3', 2), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))
star_8  = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('skip_connect', 2), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 2)], reduce_concat=range(2, 6))
star_9  = Genotype(normal=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 3), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 1), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))
star_10 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('skip_connect', 1), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('avg_pool_3x3', 3), ('skip_connect', 2), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
star_11 = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 3), ('dil_conv_5x5', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))
star_12 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 2), ('skip_connect', 3), ('sep_conv_5x5', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 2), ('avg_pool_3x3', 3), ('dil_conv_5x5', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))
star_13 = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3), ('dil_conv_3x3', 1), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 2), ('avg_pool_3x3', 3), ('skip_connect', 2), ('skip_connect', 4)], reduce_concat=range(2, 6))
star_14 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
star_15 = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1), ('skip_connect', 3), ('sep_conv_5x5', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 3), ('dil_conv_3x3', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
star_16 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 3), ('sep_conv_5x5', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 3), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))
star_17 = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('skip_connect', 2), ('avg_pool_3x3', 3), ('dil_conv_5x5', 2), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))
star_18 = Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('skip_connect', 1), ('dil_conv_3x3', 3), ('dil_conv_5x5', 0), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))
star_19 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 0), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))
star_20 = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 2), ('dil_conv_5x5', 3), ('avg_pool_3x3', 1), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
star_21 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 3), ('avg_pool_3x3', 1), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))
star_22 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
star_23 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
star_24 = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 3), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))
star_25 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 1), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 2), ('skip_connect', 3), ('max_pool_3x3', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
star_26 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('dil_conv_3x3', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('dil_conv_5x5', 3), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))
star_27 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 3), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
star_28 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))
star_29 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('skip_connect', 2), ('sep_conv_5x5', 3), ('skip_connect', 0), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))
star_30 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))
star_31 = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_3x3', 3), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
star_32 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('dil_conv_3x3', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 2), ('skip_connect', 3), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))
star_33 = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('avg_pool_3x3', 3), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 3), ('sep_conv_5x5', 3), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))





darts25 = Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 4), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
darts200 = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_5x5', 3), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('dil_conv_5x5', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
darts50 = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
darts100 = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_5x5', 3), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
darts50nowarmup = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('skip_connect', 2), ('dil_conv_5x5', 4), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('skip_connect', 3), ('skip_connect', 2), ('dil_conv_3x3', 3), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
darts25nowarmup = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('dil_conv_3x3', 2), ('dil_conv_5x5', 4), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
darts50nowarmup_unrolled = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3), ('skip_connect', 2), ('dil_conv_3x3', 4), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('skip_connect', 2), ('skip_connect', 4)], reduce_concat=range(2, 6))
darts25nowarmup_unrolled = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))


rtk_200 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 2), ('skip_connect', 3), ('skip_connect', 2),  ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0),('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 3), ('sep_conv_5x5', 0), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))
rtk_25 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 0),  ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0),('dil_conv_3x3', 2), ('skip_connect', 2), ('skip_connect', 3), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
rtk_50 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))
rtk_100 = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
rtk_25_unrolled =  Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 3), ('max_pool_3x3', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
nash50_unrolled = Genotype(normal=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('max_pool_3x3', 1), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
nash25_unrolled = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
nash50_yijie = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('skip_connect', 3), ('max_pool_3x3', 3), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
rtk_50_unrolled = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 2), ('avg_pool_3x3', 2), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('skip_connect', 3)], reduce_concat=range(2, 6)) 
rtk_25_yijie = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))
nash_25_yijie = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 1), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

star_41 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 3), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))
star_42 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))
star_43 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('skip_connect', 2), ('max_pool_3x3', 3), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('skip_connect', 3), ('dil_conv_5x5', 1), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))
star_44 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1), ('skip_connect', 3), ('max_pool_3x3', 1), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 3), ('max_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))


######## DARTS Space ########
##### darts
darts_pt_s5_0 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 2), ('skip_connect', 4)], reduce_concat=range(2, 6)) # 2.85
darts_pt_s5_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('dil_conv_3x3', 3), ('sep_conv_5x5', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6)) # 3.05
darts_pt_s5_2 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 3), ('max_pool_3x3', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('skip_connect', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6)) # 2.66
darts_pt_s5_3 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 1), ('max_pool_3x3', 3), ('sep_conv_5x5', 2), ('skip_connect', 3)], reduce_concat=range(2, 6)) # 3.33

#### blank
blank_pt_s5_0 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 0), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 6)) # 3.04
blank_pt_s5_1 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 2), ('sep_conv_3x3', 3), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 2), ('max_pool_3x3', 3), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6)) # 3.15
blank_pt_s5_2 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 3), ('max_pool_3x3', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6)) # 2.58
blank_pt_s5_3 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 3), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6)) # 3.40


# #'none',
#     'max_pool_3x3', # 0
#     'avg_pool_3x3', # 1
#     'skip_connect', # 2
#     'sep_conv_3x3', # 3
#     'sep_conv_5x5', # 4
#     'dil_conv_3x3', # 5
#     'dil_conv_5x5' # 6




# Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 2), ('skip_connect', 3), ('skip_connect', 2),  ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0),('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 3), ('sep_conv_5x5', 0), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))

star_45 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 1), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))
star_46 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))
star_47 = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('dil_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
star_48 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))
star_49 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1), ('skip_connect', 3), ('max_pool_3x3', 1), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 3), ('max_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))
star_50 = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 2), ('skip_connect', 3), ('dil_conv_5x5', 2), ('skip_connect', 4)], reduce_concat=range(2, 6))
star_51 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('skip_connect', 2), ('skip_connect', 3), ('dil_conv_3x3', 1), ('skip_connect', 4)], reduce_concat=range(2, 6))
star_52 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('skip_connect', 4)], reduce_concat=range(2, 6))