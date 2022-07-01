from config import *
import time
def get_nasconfig_name(nc):
    name = f"search-{nc.data.dataset}-{nc.search.method}-{nc.search.search_space}-{nc.search.op_names}.json"
    return name
NAS_CONFIG = NAS_TEMPLATE.copy()
NAS_CONFIG.continue_mode='new'
NAS_CONFIG.run.last_ckpt=None
NAS_CONFIG.run.test_mode=True
NAS_CONFIG.search=NAS_CONFIG.search.copy({"arch_learning_rate": 1,
                                            "edge_learning_rate": 1,
                                            "op_names": "PCDARTS_SYMMETRY_auto_Z2"})
NAS_CONFIG.train=NAS_CONFIG.train.copy({"batch_size": 100,
                                         "epochs": 300,
                                         "drop_path_prob": 0,
                                        })
NAS_CONFIG.model=NAS_CONFIG.model.copy({"arch":None
                                        })

NAS_CONFIG.data=NAS_CONFIG.data.copy({"criterion_type":"default","dataset": "msdataT"})
NAS_CONFIG.scheduler=Config({"optimizer_type":'Adam',
						"scheduler": "cosine",
                        "scheduler_epochs": 50,
                        "lr_anneal_cycles": 1,
                        "learning_rate": 0.1,
                        "learning_rate_min": 0.0,
                        "momentum": 0.9,
                        "weight_decay": 0.0003,
                        "grad_clip": 5,
                    })
time_now = time.strftime("%Y%m%d%H%M%S")
NAS_CONFIG.save(f"projects/undo/{time_now}-{get_nasconfig_name(NAS_CONFIG)}")
