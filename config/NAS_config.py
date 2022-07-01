from .config import *
NAS_TEMPLATE = Config({
    "PROJECT_TYPE" :"NAS_SEARCH",
	"continue_mode":"new",
    "run": Config({ "seed": 'random',
                    "last_ckpt": 0,
                    "test_mode": False,
                    "gpu": 0}),
    "search": Config({  "single_level": False,
                        "exclude_zero": False,
                        "track_running_stats": True,
                        "method": "eedarts",#<----
                        "search_space": "pcdarts",#<----
                        "nodes": 4,
                        "unrolled":None,
                        "arch_grad_clip": None,
                        "train_portion": 0.5,
                        "arch_learning_rate": 1,#<----
                        "edge_learning_rate": 1,#<----
                        "discrete": False,
                        "adapt_lr": False,
                        "arch_weight_decay": 0.0,
                        "gd": False,
                        "learn_edges": True,
                        "trace_norm": 8,
                        "op_names": "PCDARTS_SYMMETRY_P4Z2"#<----
                        }),
    "train": Config({"batch_size": 1000,
                     "epochs": 300,
                     "drop_path_prob": 0
                    }),
    "model": Config({"arch":None,
                     "init_channels": 16,
                     "nodes": 4,
                     "layers": 8,
                     "auxiliary": False,
                     "auxiliary_weight": 0.4,
                    }),
    "data": Config({"criterion_type":"default","dataset": "msdataRCurve32"}),
    "scheduler":Config({"optimizer_type":'Adam',
						"scheduler": "cosine",
                        "scheduler_epochs": 50,
                        "lr_anneal_cycles": 1,
                        "learning_rate": 0.1,
                        "learning_rate_min": 0.0,
                        "momentum": 0.9,
                        "weight_decay": 0.0003,
                        "grad_clip": 5,
                    })
})
