import re, sys, os, json
from pathlib import Path
import torch
repo_start = f'../../'
sys.path.append(repo_start)

from modules.binn_eql.model_wrapper_2d import model_wrapper
from modules.binn_eql.build_binn_eql_net import BINN

def extract_final_loss(filepath, loss_types=['pde', 'gls', 'reg']):
    """
    Reads the given file and extracts the validation loss (as a float)
    from the line containing the word 'Elapsed'.
    """
    # Regular expression to capture the validation loss
    # This regex looks for "Val loss = " followed by a floating point number (possibly in scientific notation)
    # val_loss_pattern = re.compile(r"Val loss = ([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
    pde_pattern = re.compile(r"Val PDE = ([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
    gls_pattern = re.compile(r"Val GLS = ([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
    reg_pattern = re.compile(r"Val Reg = ([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
        
    with open(filepath, 'r') as file:
        for line in file:
            # Check if the line contains "Elapsed"
            if "Elapsed" in line:
                # Search the line for the pattern
                pde = float(pde_pattern.search(line).group(1))
                gls = float(gls_pattern.search(line).group(1))
                reg = float(reg_pattern.search(line).group(1))
                
                loss = 0
                
                if 'pde' in loss_types: loss += pde
                if 'gls' in loss_types: loss += gls
                if 'reg' in loss_types: loss += reg
                
                return loss
                
    # If no matching line was found
    return None

def process_directory(parent_dir, loss_types):
    """
    For each subdirectory under parent_dir, finds slurm files with the pattern "slurm-####.out",
    selects the one with the lower number, extracts the validation loss, and prints the subdirectory path
    alongside the extracted loss.
    """
    results = {}
    # Use pathlib to iterate over subdirectories
    for subdir in Path(parent_dir).iterdir():
        if subdir.is_dir():
            # Find files matching pattern "slurm-*.out"
            slurm_files = list(subdir.glob("slurm-*.out"))
            if not slurm_files:
                # print(f"No slurm files found in {subdir}")
                results[str(subdir)] = None
                continue
            
            # Extract numeric part from filename (assumes naming: slurm-####.out)
            def extract_number(file_path):
                match = re.search(r"slurm-(\d+)\.out", file_path.name)
                return int(match.group(1)) if match else float('inf')
            
            # Sort files by the extracted number
            slurm_files.sort(key=extract_number)
            # Take the first file (lowest number)
            first_file = slurm_files[0]
            val_loss = extract_final_loss(first_file, loss_types)
            
            if val_loss is not None:
                results[str(subdir)] = val_loss
            else:
                # print(f"Could not extract validation loss from file {first_file} in {subdir}")
                results[str(subdir)] = None
    
    return results

if __name__ == '__main__':        
    
    # Get parent directory
    parent_directory = str(sys.argv[1])
    
    # Get training data
    subdirectory = os.listdir(parent_directory)[0]
    
    config_path = f'{parent_directory}/{subdirectory}/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    training_data_path = config['training_data_path']
    training_data = torch.load(training_data_path)['training_data']

    # Get losses
    loss_types = ['reg', 'pde']
    losses = process_directory(parent_directory, loss_types)

    # Sort the results by validation loss (lowest first)
    sorted_losses = sorted(losses.items(), key=lambda item: (item[1] is None, item[1]))

    # Print the sorted results
    for dir_name, val_loss in sorted_losses:
        if val_loss:
            print(f"\nDirectory: {dir_name.split('/')[-1]}", flush=True)
            print(f"Validation Loss: {val_loss}", flush=True)
            
            # Load model
            binn = BINN(
                dimensions=2,
                species=2, 
                train_data=training_data, 
                diff_coeffs=[0.01, 1],
                # uv_layers=[256, 256, 256, 256, 2],
                # diff_coeffs=(),
                duplicates=5)
                # duplicates=20)

            binn.to('cpu')

            parameters = binn.parameters()

            opt = torch.optim.Adam(parameters, lr=0.001)

            model = model_wrapper(
                model=binn,
                optimizer=opt,
                loss=binn.loss,
                dir_name=dir_name,
                save_name=f'{dir_name}/binn')
                    
            model.load(f"{dir_name}/binn_best_val_model", device='cpu')
            # model.model.prune(thresh=5)
            # Print equation
            fn = f'{dir_name}/equation.txt'
            for term in model.model.generate_equation():
                print(f'{term}', flush=True)
            
            model.model.fine_tune_eql(threshold=0.01, epsilon=0.1)
            for term in model.model.generate_equation():
                print(f'{term}', flush=True)
                
            if not model.model.diff_coeffs:          
                print(f'{[D.item() for D in model.model.diffusion_fitter()]}\n', flush=True)
