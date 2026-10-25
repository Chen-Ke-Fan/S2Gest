def build_model(configer, in_channels, num_classes, **kwargs):
    return S2Gest(configer, in_channels, num_classes, **kwargs)



def S2Gest(configer, in_channels, num_classes, **kwargs):
    from models.S2Gest import GestureClassification
    model_size = configer.get("network", "model_size")

    drop_path = kwargs.get("drop_path_rate", configer.get("network", "drop_path_rate"))
    cls_dropout = kwargs.get("classifier_dropout", configer.get("network", "classifier_dropout"))

    model_cfg = {
        "model_config": configer.get("network", "model_configs", model_size),
        "drop_path_rate": drop_path,
        "classifier_dropout": cls_dropout,
    }

    net = GestureClassification(
        in_channels=in_channels,
        num_classes=num_classes,
        **model_cfg
    )

    return net
