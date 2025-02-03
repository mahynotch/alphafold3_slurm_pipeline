import yaml
import sys

def modify_config(config_path, input_param):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config.update(input_param)
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python modify_config.py '<key>: <value>'")
        sys.exit(1)

    config_path = './alphafold3_slurm/config.yaml'
    input_param = yaml.safe_load(sys.argv[1])
    modify_config(config_path, input_param)