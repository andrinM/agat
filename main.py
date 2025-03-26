import json
import os
import subprocess
import sys

# load json configurations
with open('config.json', 'r') as f:
    config = json.load(f)

# set project root
project_root = config['settings']['working_directory']
os.chdir(project_root) 

# start script
def run_script(script_name):
    
    if script_name in config['settings']['scripts']:
        script_info = config['settings']['scripts'][script_name]
        command = f"{script_info['command']} {script_info['script']}"
        print(f"Running: {command}")
        subprocess.run(command, shell=True)
    else:
        print(f"Error: Script '{script_name}' not found in config.json")

# run main with ecaluation or home 
if __name__ == "__main__":
    import argparse
    
    
    parser = argparse.ArgumentParser(description="Run selected script.")
    parser.add_argument("script", choices=config['settings']['scripts'].keys(), help="Choose which script to run (evaluation/home)")
    
    args = parser.parse_args()
    run_script(args.script)