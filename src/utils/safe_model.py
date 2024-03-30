import torch
import yaml
import os


def full_safe(path, model, config, epochs=None):
    
    if os.exist(os.path.join(path, config["model_name"])):
        pass
    else:
        os.mkdir(os.path.join(path, config["model_name"]))
        torch.save(model.state_dict(), os.path.join(path, config["model_name"], config["model_name"] + ".pth"))
        with open(os.path.join(path, config["model_name"], "config.yaml"), "w") as f:
            yaml.dump(config, f)

        

