

@hydra.main(config_path="../../config", config_name="main", version_base=None)
def generate_reports(config: DictConfig):
    """Function to process the data"""

if __name__ == "__main__":
    generate_reports()