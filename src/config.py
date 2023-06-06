import yaml
import pprint

class Config:
    def __init__(self, path_to_config):
        self.path_config = path_to_config

    def __str__(self):
        return pprint.pformat(self.config_dict, indent=4) 

    @property
    def config_dict(self):
        with open(self.path_config, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)

        return config

    @property
    def prompt_config(self):
        prompt_config = self.config_dict["prompts"]

        return prompt_config
    
    @property
    def train_config(self):
        train_config = self.config_dict["train"]

        return train_config

