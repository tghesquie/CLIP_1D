""" Post Processing"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
import os

def list_npz_files(directory, prefix="results"):
    """List all .npz files in the specified directory and its subdirectories that start with the given prefix."""
    npz_files = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir.startswith(prefix):
                folder_path = os.path.join(root, dir)
                for file in os.listdir(folder_path):
                    if file.endswith('.npz'):
                        npz_files.append(os.path.join(folder_path, file))
    return npz_files

def load_and_process_files(npz_files):
    data_collection = []
    for file in npz_files:
        with np.load(file, allow_pickle=True) as data:
            # Extract data or simply store for later processing
            parameters = data['inputs'].item() 
            stress = data['stress']
            imposed_disp = data['imposed_disp']
            seperation = data['seperation']
            cohesive_damage =data['cohesive_damage']
            coh_disp_act = data['coh_disp_act']
            bulk_disp_act = data['bulk_disp_act']
            tot_disp_act = data['tot_disp_act']
            coh_disp_exp = data['coh_disp_exp']
            bulk_disp_exp = data['bulk_disp_exp']
            tot_disp_exp = data['tot_disp_exp']

            # Collect data in a list or process as needed
            data_collection.append({
                'filename': file,
                'parameters': parameters,
                'stress': stress,
                'imposed_disp': imposed_disp,
                'seperation':seperation,
                'cohesive_damage':cohesive_damage,
                'coh_disp_act':coh_disp_act,
                'bulk_disp_act': bulk_disp_act,
                'tot_disp_act':tot_disp_act,
                'coh_disp_exp':coh_disp_exp,
                'bulk_disp_exp':bulk_disp_exp,
                'tot_disp_exp':tot_disp_exp
            })

    return data_collection

def format_x_ticks(value, pos):
            return f"{value:.1e}"

def plot_all_stress_vs_displacement(processed_data):
    plt.figure(figsize=(10, 6)) 

    for entry in processed_data:
       
        Dm = entry['parameters'].get('Dm', 'unknown')
        alpha = entry['parameters'].get('alpha', 'unknown')
        sigc = entry['parameters'].get('sigc', 'unknown')
        Gc = entry['parameters'].get('Gc', 'unknown')
        functional_choice = entry['parameters'].get('functional_choice','unknown')
        
        if functional_choice == 'CZM':
            label = f"{functional_choice}"
        elif functional_choice == 'LIP':
            label = f"{functional_choice}"
        else :
            alpha_formatted = f"{alpha:.3f}"
            label = f"{functional_choice},$D_m$={Dm}, $\\alpha$={alpha_formatted}"
        plt.plot(entry['imposed_disp'], entry['stress'], label=label)

    plt.title("Stress [$\sigma$] vs Imposed Displacement [$u_t$]", fontsize = 'large')
    plt.xlabel("Imposed Displacement [m]", fontsize = 'large')
    plt.ylabel("Stress [Pa] ", fontsize = 'large')
    plt.axhline(y=sigc, color='red', linestyle='--')
    plt.text((Gc*2)/(sigc), sigc, 'Stress limit', color='red', fontsize=10, va='bottom', ha='right')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    plt.legend(fontsize = 'large') 
    plt.grid(True) 
    plt.savefig('stress_vs_imposed_displacement.png')
    plt.show()

def plot_all_dissipation(processed_data):
     
    fig, axs = plt.subplots(2, 3, figsize = (12, 8))

    for entry in processed_data :
        
        Dm = entry['parameters'].get('Dm', 'unknown')
        alpha = entry['parameters'].get('alpha', 'unknown')
        
        functional_choice = entry['parameters'].get('functional_choice','unknown')
        
        if functional_choice == 'CZM' or functional_choice == 'LIP':
            continue

        alpha_formatted = f"{alpha:.3f}"
        label = f"{functional_choice},$D_m$={Dm}, $\\alpha$={alpha_formatted}"

        axs[0,0].plot(entry['imposed_disp'], entry['coh_disp_act'] )
        axs[0,1].plot(entry['imposed_disp'], entry['bulk_disp_act'] )
        axs[0,2].plot(entry['imposed_disp'], entry['tot_disp_act'],label = label )

        axs[1,0].plot(entry['imposed_disp'], entry['coh_disp_exp'] )
        if functional_choice == 'CLIP_4terms':
        
            axs[1,1].plot(entry['imposed_disp'], entry['bulk_disp_exp'] )

        axs[1,2].plot(entry['imposed_disp'], entry['tot_disp_exp'],label = label )

    axs[0,0].set_title("Percentage of Cohesive Dissipation", fontweight= 'bold', fontsize = 'large')
    axs[0,1].set_title("Percentage of Bulk Dissipation", fontweight= 'bold', fontsize = 'large')
    axs[0,2].set_title("Total Dissipation Percentage", fontweight= 'bold', fontsize = 'large')

    axs[0,0].set_xlabel("Imposed displacement [m]", fontsize = 'large')
    axs[0,1].set_xlabel("Imposed displacement [m]", fontsize = 'large')
    axs[0,2].set_xlabel("Imposed displacement [m]", fontsize = 'large')
    axs[1,0].set_xlabel("Imposed displacement [m]", fontsize = 'large')
    axs[1,1].set_xlabel("Imposed displacement [m]", fontsize = 'large')
    axs[1,2].set_xlabel("Imposed displacement [m]", fontsize = 'large')

    axs[0,2].legend()
    axs[1,2].legend()

    for ax in axs.flat:
        
        ax.xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_all_coh_stress_vs_seperation(processed_data):
    plt.figure(figsize=(10, 6)) 
    
    for entry in processed_data :
        
        Gc = entry['parameters'].get('Gc', 'unknown')
        sigc = entry['parameters'].get('sigc', 'unknown')
        wc = (2*Gc)/(sigc)
        seperation = entry['seperation']
        N_elements = entry['parameters'].get('N_elements', 'unknown')
        functional_choice = entry['parameters'].get('functional_choice','unknown')

        if functional_choice == 'CZM' or functional_choice == 'LIP':
            continue
        
        jump_coh = [arr[int((N_elements - 1)/2)] for arr in seperation]
        
        d_area = entry['stress'].copy()
        sig = [sigc,0]
        w = [0,wc]
        area = np.trapz(d_area,jump_coh)
        label = fr'$D_m = $, $Area = ${area:.2f}'
        plt.plot(jump_coh, entry['stress'], marker = 'x',label= label)
    plt.plot(w,sig,color = 'black')
    plt.xlabel("Cohesvie zone opening [m]", fontsize = 'large')
    plt.ylabel("Stress [Pa] ", fontsize = 'large')
    plt.axhline(y=sigc, color='red', linestyle='--')
    plt.text(8e-5, 3e6, 'Stress limit', color='red', fontsize=10, va='bottom', ha='right')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    plt.title("Cohesive Stress [$\sigma$] vs Opening [$\omega$]", fontsize = 'large')
    plt.legend(fontsize = 'large')
    plt.grid(True)
    plt.show()

def plot_all_damge_vs_imposed_disp(processed_data):
    plt.figure(figsize=(10, 6)) 

    for entry in processed_data :
        max_d_str = []
        damage = entry['cohesive_damage']
        max_d_str.extend([np.max(arr) for arr in damage])

        Dm = entry['parameters'].get('Dm', 'unknown')
        alpha = entry['parameters'].get('alpha', 'unknown')
        
        functional_choice = entry['parameters'].get('functional_choice','unknown')
        
        if functional_choice == 'CZM' or functional_choice == 'LIP':
            continue
        alpha_formatted = f"{alpha:.3f}"
        label = f"{functional_choice},$D_m$={Dm}, $\\alpha$={alpha_formatted}"
        plt.plot(entry['imposed_disp'], max_d_str,label = label)

    plt.xlabel("Imposed Displacement [m]", fontsize = 'large')
    plt.ylabel("Cohesive Damage ", fontsize = 'large')
    plt.title('Damage [$d$] vs Imposed displacement [$u_t$]', fontsize = 'large')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    plt.grid(True)
    plt.legend(fontsize='large')
    plt.show()
    
def plot_all(processed_data):

    fig1, axs1 = plt.subplots(1, 1, figsize=(10, 6))
    fig2, axs2 = plt.subplots(2, 3, figsize=(12, 8))
    fig3, axs3 = plt.subplots(1, 1, figsize=(10, 6))
    fig4, axs4 = plt.subplots(1, 1, figsize=(10, 6))

    for entry in processed_data:
        Dm = entry['parameters'].get('Dm', 'unknown')
        alpha = entry['parameters'].get('alpha', 'unknown')
        sigc = entry['parameters'].get('sigc', 'unknown')
        Gc = entry['parameters'].get('Gc', 'unknown')
        wc = (2*Gc)/(sigc)
        seperation = entry['seperation']
        N_elements = entry['parameters'].get('N_elements', 'unknown')
        functional_choice = entry['parameters'].get('functional_choice','unknown')
        

        if functional_choice == 'CZM':
            label = f"{functional_choice}"
        elif functional_choice == 'LIP':
            label = f"{functional_choice}"
        else :
            alpha_formatted = f"{alpha:.3f}"
            label = f"{functional_choice},$D_m$={Dm}, $\\alpha$={alpha_formatted}"

        axs1.plot(entry['imposed_disp'], entry['stress'], label=label)

        if functional_choice == 'CZM' or functional_choice == 'LIP':
            continue

        axs2[0,0].plot(entry['imposed_disp'], entry['coh_disp_act'] )
        axs2[0,1].plot(entry['imposed_disp'], entry['bulk_disp_act'] )
        axs2[0,2].plot(entry['imposed_disp'], entry['tot_disp_act'],label = label )
        axs2[1,0].plot(entry['imposed_disp'], entry['coh_disp_exp'] )

        if functional_choice == 'CLIP_4terms':        
            axs2[1,1].plot(entry['imposed_disp'], entry['bulk_disp_exp'] )

        axs2[1,2].plot(entry['imposed_disp'], entry['tot_disp_exp'],label = label )

        max_d_str = []
        damage = entry['cohesive_damage']
        max_d_str.extend([np.max(arr) for arr in damage])

        axs3.plot(entry['imposed_disp'], max_d_str,label = label)

        jump_coh = [arr[int((N_elements - 1)/2)] for arr in seperation]
        d_area = entry['stress'].copy()
        sig = [sigc,0]
        w = [0,wc]
        area = np.trapz(d_area,jump_coh)
        label = fr'$D_m = $, $Area = ${area:.2f}'

        axs4.plot(jump_coh, entry['stress'], marker = 'x',label= label)

    axs1.set_title("Stress [$\sigma$] vs Imposed Displacement [$u_t$]", fontsize = 'large')
    axs1.set_xlabel("Imposed Displacement [m]", fontsize = 'large')
    axs1.set_ylabel("Stress [Pa] ", fontsize = 'large')
    axs1.axhline(y=sigc, color='red', linestyle='--')
    axs1.text((Gc*2)/(sigc), sigc, 'Stress limit', color='red', fontsize=10, va='bottom', ha='right')
    axs1.xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    axs1.legend(fontsize = 'large') 
    axs1.grid(True)

    axs2[0,0].set_title("Percentage of Cohesive Dissipation", fontweight= 'bold', fontsize = 'large')
    axs2[0,1].set_title("Percentage of Bulk Dissipation", fontweight= 'bold', fontsize = 'large')
    axs2[0,2].set_title("Total Dissipation Percentage", fontweight= 'bold', fontsize = 'large')
    axs2[0,0].set_xlabel("Imposed displacement [m]", fontsize = 'large')
    axs2[0,1].set_xlabel("Imposed displacement [m]", fontsize = 'large')
    axs2[0,2].set_xlabel("Imposed displacement [m]", fontsize = 'large')
    axs2[1,0].set_xlabel("Imposed displacement [m]", fontsize = 'large')
    axs2[1,1].set_xlabel("Imposed displacement [m]", fontsize = 'large')
    axs2[1,2].set_xlabel("Imposed displacement [m]", fontsize = 'large')
    axs2[0,2].legend()
    axs2[1,2].legend()
    for ax in axs2.flat:            
        ax.xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
        ax.grid(True)

    axs3.set_title('Damage [$d$] vs Imposed displacement [$u_t$]', fontsize = 'large')
    axs3.set_xlabel("Imposed Displacement [m]", fontsize = 'large')
    axs3.set_ylabel("Cohesive Damage ", fontsize = 'large')
    axs3.xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    axs3.legend(fontsize = 'large') 
    axs3.grid(True)

    axs4.plot(w,sig,color = 'black')
    axs4.set_xlabel("Cohesvie zone opening [m]", fontsize = 'large')
    axs4.set_ylabel("Stress [Pa] ", fontsize = 'large')
    axs4.axhline(y=sigc, color='red', linestyle='--')
    axs4.text(8e-5, 3e6, 'Stress limit', color='red', fontsize=10, va='bottom', ha='right')
    axs4.xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    axs4.set_title("Cohesive Stress [$\sigma$] vs Opening [$\omega$]", fontsize = 'large')
    axs4.legend(fontsize = 'large')
    axs4.grid(True)

    plt.tight_layout()
    plt.show()




script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
npz_files = list_npz_files(script_dir)
processed_data = load_and_process_files(npz_files)

plot_all_stress_vs_displacement(processed_data)
plot_all_dissipation(processed_data)
plot_all_coh_stress_vs_seperation(processed_data)
plot_all_damge_vs_imposed_disp(processed_data)
plot_all(processed_data)


