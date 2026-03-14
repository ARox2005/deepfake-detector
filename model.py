import timm

def create_model(pretrained=True):
    model = timm.create_model(
        'xception',
        pretrained=pretrained,
        num_classes=1
    )

    return model