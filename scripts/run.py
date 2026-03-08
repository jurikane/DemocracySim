"""
Script to run the DemocracySim model server.
Configure using a config file (YAML or TOML) inside the 'configs' folder.
Use --config to specify a config file (YAML or TOML).
Example:
python -m scripts.run -c default.yaml --no-browser
"""
import argparse
from mesa.visualization.ModularVisualization import ModularServer
from src.config.loader import load_config
from src.model_setup import make_server

def main():
    parser = argparse.ArgumentParser(description="Run DemocracySim")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to YAML/TOML config")
    parser.add_argument("--no-browser",
                        action="store_true",
                        help="Do not open browser on launch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    server: ModularServer = make_server(cfg)
    server.launch(open_browser=not args.no_browser)

if __name__ == "__main__":
    main()
