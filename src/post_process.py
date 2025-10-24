""" Post Processing"""
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter, ScalarFormatter, MaxNLocator, FormatStrFormatter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from functions import gd_cohesive_std
import matplotlib.gridspec as gridspec
import pandas as pd

plt.rcParams.update({
    'axes.labelsize': 20,      # X and Y axis labels
    'xtick.labelsize': 20,     # X axis tick labels  
    'ytick.labelsize': 20,     # Y axis tick labels
    'legend.fontsize': 17,      # Legend text
    'axes.titlesize': 20,      # Plot title
    'font.family': 'sans-serif'
})


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
    """
    Load data from .npz files and process it based on the 'functional_choice' parameter.
    """
    data_collection = []
    for file in npz_files:
        with np.load(file, allow_pickle=True) as data:
            # Extract data
            parameters = data['inputs'].item()       
            functional_choice = parameters.get('functional_choice')      
            stress = data['stress']
            imposed_disp = data['imposed_disp']
            

            if functional_choice in ['CLIP-3terms', 'CLIP-4terms']:
                seperation = data['seperation']
                cohesive_damage =data['cohesive_damage']
                bulk_damage = data['bulk_damage']
                bulk_damage_overall = data['bulk_damage_overall']
                coh_disp_act = data['coh_disp_act']
                bulk_disp_act = data['bulk_disp_act']
                tot_disp_act = data['tot_disp_act']
                coh_disp_exp = data['coh_disp_exp']
                bulk_disp_exp = data['bulk_disp_exp']
                tot_disp_exp = data['tot_disp_exp']      
                lmb = data['lmb']
                pot_energy = data['potential_energy']
                coh_energy = data['cohesive_energy']
                total_energy = data['total_energy']
                displacement = data['displacement']

            elif functional_choice in ['CZM']:
                seperation = data['seperation']
                cohesive_damage =data['cohesive_damage']
                displacement = data['displacement']

            elif functional_choice in ['LIP']:
                coh_disp_act = data['coh_disp_act']
                bulk_disp_act = data['bulk_disp_act']
                tot_disp_act = data['tot_disp_act']
                coh_disp_exp = data['coh_disp_exp']
                bulk_disp_exp = data['bulk_disp_exp']
                tot_disp_exp = data['tot_disp_exp']     
         
            if parameters.get('functional_choice') in ['CLIP-3terms', 'CLIP-4terms']:
                # Collect data in a list
                data_collection.append({
                    'filename': file,
                    'parameters': parameters,
                    'stress': stress,
                    'imposed_disp': imposed_disp,
                    'seperation':seperation,
                    'cohesive_damage':cohesive_damage,
                    'bulk_damage' : bulk_damage,
                    'bulk_damage_overall' :bulk_damage_overall,
                    'coh_disp_act':coh_disp_act,
                    'bulk_disp_act': bulk_disp_act,
                    'tot_disp_act':tot_disp_act,
                    'coh_disp_exp':coh_disp_exp,
                    'bulk_disp_exp':bulk_disp_exp,
                    'tot_disp_exp':tot_disp_exp,
                    'lmb':lmb,
                    'potential_energy' : pot_energy,
                    'cohesive_energy' : coh_energy,
                    'total_energy' : total_energy,
                    'displacement':displacement,
                })

            elif  parameters.get('functional_choice') in ['CZM']:
                data_collection.append({
                    'filename': file,
                    'parameters': parameters,
                    'stress': stress,
                    'imposed_disp': imposed_disp,
                    'seperation':seperation,
                    'cohesive_damage':cohesive_damage,
                    'displacement':displacement,              
                })
            elif  parameters.get('functional_choice') in ['LIP']:
                data_collection.append({
                    'filename': file,
                    'parameters': parameters,
                    'stress': stress,
                    'imposed_disp': imposed_disp,
                    'coh_disp_act':coh_disp_act,
                    'bulk_disp_act': bulk_disp_act,
                    'tot_disp_act':tot_disp_act,
                    'coh_disp_exp':coh_disp_exp,
                    'bulk_disp_exp':bulk_disp_exp,
                    'tot_disp_exp':tot_disp_exp,                              
                })
            elif parameters.get('functional_choice') in ['Exact']:
                data_collection.append({
                    'filename': file,
                    'parameters': parameters,
                    'stress': stress,
                    'imposed_disp': imposed_disp,                                   
                })

    return data_collection

def format_x_ticks(value, pos):
            return f"{value:.1e}"

def plot_all_stress_vs_displacement(processed_data):
    plt.figure(figsize=(10, 6))

    # Storage for handles
    handles = {'Exact': None, 'CLIP': [], 'CZM': None, 'LIP': None}

    for entry in processed_data:
        params = entry['parameters']
        Dm = params.get('Dm', 'unknown')
        Gc = params.get('Gc', 'unknown')
        sigc = params.get('sigc', 'unknown')
        alpha = params.get('alpha', 'unknown')
        functional_choice = params.get('functional_choice', 'unknown')

        # Default style
        color = "C0"
        # color = "#6699FF"
        linestyle = '--'
        linewidth = 1.5
        zorder = 4
        marker = 's'
        label = f"CLIP, $D_m$={Dm}"

        if functional_choice == 'CZM':
            # color = "#CC9900"
            color = "C1"
            linestyle = '--'
            zorder = 3
            marker = 'd'
            label = "CZM"
            linewidth = 1.5
        elif functional_choice == 'LIP':
            color = "C2"
            linestyle = '--'
            zorder = 2
            marker = 'o'
            label = "LIP-field"
            linewidth = 1.5
        elif functional_choice == 'Exact':
            color = "black"
            linestyle = '-'
            zorder = 1
            marker = None
            label = "Exact"
            linewidth = 3

        ln, = plt.plot(
            entry['imposed_disp'], entry['stress'],
            label=label, color=color,
            linestyle=linestyle, marker=marker,
            linewidth=linewidth, zorder=zorder
        )

        # Store handle
        if functional_choice == 'Exact':
            handles['Exact'] = ln
        elif functional_choice == 'CZM':
            handles['CZM'] = ln
        elif functional_choice == 'LIP':
            handles['LIP'] = ln
        else:
            handles['CLIP'].append(ln)

    # Draw stress limit line once (use last sigc, Gc from loop)
    if 'sigc' in params and 'Gc' in params:
        sigc = params['sigc']
        Gc = params['Gc']
        wc = (2*Gc) / sigc
        plt.axhline(y=sigc, color='red', linestyle='--')
        plt.text((Gc * 2) / sigc, sigc, r'Stress limit $(\sigma_c)$',
                 color='red', fontsize=12, va='bottom', ha='right')
        # plt.axvline(x=wc, color='red', linestyle='--')
        # plt.text(wc, sigc/2, r'Critical opening $(\omega_c)$',
        # color='red', fontsize=10, va='bottom', ha='left', rotation=90)

    # Build ordered legend: Exact, CLIP entries, CZM, LIP
    ordered_handles = []
    ordered_labels = []

    if handles['Exact'] is not None:
        ordered_handles.append(handles['Exact'])
        ordered_labels.append('Exact')
    for ln in handles['CLIP']:
        ordered_handles.append(ln)
        ordered_labels.append(ln.get_label())
    if handles['CZM'] is not None:
        ordered_handles.append(handles['CZM'])
        ordered_labels.append('CZM')
    if handles['LIP'] is not None:
        ordered_handles.append(handles['LIP'])
        ordered_labels.append('LIP-field')

    # plt.title("Stress [$\\sigma$] vs Imposed displacement [$u_t$]", fontsize='large')
    plt.xlabel("Imposed displacement [m]")
    plt.ylabel("Stress [Pa]")
    # plt.xlabel(r"$\dfrac{\sigma}{\sigma_c}$", fontsize='large')
    # plt.ylabel("Stress [Pa]", fontsize='large')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    plt.grid(True)
    plt.legend(ordered_handles, ordered_labels, shadow=True)
    plt.tight_layout()
    plt.savefig('stress_vs_imposed_displacement.png', dpi=300)
    plt.show()

def plot_mesh_conv(processed_data):
    """
    Plot stress vs imposed displacement for multiple mesh sizes,
    with legend entries: Exact (thicker), then he=lc/5, lc/10, lc/20, lc/40.
    """
    mesh_order = ['5', '10', '20', '40']
    mesh_styles = {
        '5':  {'color': 'C0', 'linestyle': '--', 'marker': 's', 'zorder': 2},
        '10': {'color': 'C1', 'linestyle': '--', 'marker': 'd', 'zorder': 3},
        '20': {'color': 'C2', 'linestyle': '--', 'marker': 'o', 'zorder': 4},
        '40': {'color': 'C3', 'linestyle': '--', 'marker': 'x', 'zorder': 5},
    }

    mesh_lines = {}
    exact_line = None

    plt.figure(figsize=(10, 6))
    for entry in processed_data:
        params = entry['parameters']
        he = params.get('he')
        functional_choice = params.get('functional_choice', '')
        sigc = params.get('sigc')
        Gc  = params.get('Gc')

        x = entry['imposed_disp']
        y = entry['stress']

        # Skip CZM and LIP entirely
        if functional_choice in ['CZM', 'LIP']:
            continue

        # Exact case: plot first, thicker
        if functional_choice == 'Exact':
            ln, = plt.plot(
                x, y,
                color='black',
                linestyle='-',
                linewidth=3,   # thicker line
                zorder=1,
                label='Exact'
            )
            exact_line = ln
            continue

        # CLIP cases
        denom = f"{int(he):.0f}" if isinstance(he, (int, float)) else str(he)
        style = mesh_styles.get(denom, {'color':'k','linestyle':'-', 'marker':'o'})
        label = rf"$h_e = \frac{{\ell_c}}{{{denom}}}$"

        ln, = plt.plot(
            x, y,
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=1.5,
            zorder=style['zorder'],
            label=label,
            marker=style['marker'],
        )
        mesh_lines[denom] = ln

    # Stress limit line
    if sigc is not None and Gc is not None:
        plt.axhline(y=sigc, color='red', linestyle='--')
        x_text = (2 * Gc) / sigc
        plt.text(
            x_text, sigc,
            r'Stress limit $(\sigma_c)$',
            color='red',
            fontsize=12,
            va='bottom',
            ha='right'
        )

    # Build legend: Exact first, then mesh_order
    ordered_handles = []
    ordered_labels = []
    if exact_line is not None:
        ordered_handles.append(exact_line)
        ordered_labels.append('Exact')
    for denom in mesh_order:
        if denom in mesh_lines:
            ordered_handles.append(mesh_lines[denom])
            ordered_labels.append(rf"$h_e = \dfrac{{\ell_c}}{{{denom}}}$")

    # Final formatting
    # plt.title("Stress [$\\sigma$] vs Imposed displacement [$u_t$]")
    plt.xlabel("Imposed displacement [m]")
    plt.ylabel("Stress [Pa]")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    plt.grid(True)
    plt.legend(ordered_handles, ordered_labels, loc='best', shadow=True)
    plt.tight_layout()
    plt.savefig('Mesh_convergence.png', dpi=600)
    plt.show()

def plot_lda_opening(processed_data):
    for entry in processed_data :         
        functional_choice = entry['parameters'].get('functional_choice','unknown')
        if functional_choice in ['CZM', 'LIP', 'Exact']:
          continue
        Dm = entry['parameters'].get('Dm', 'unknown')
        Gc = entry['parameters'].get('Gc', 'unknown')
        sigc = entry['parameters'].get('sigc', 'unknown')
        seperation = entry['seperation']
        N_elements = entry['parameters'].get('N_elements', 'unknown') 
        print("Number of elements", N_elements)             
        jump_coh = [arr[int((N_elements - 1)/2)] for arr in seperation]        

       
        label = fr'$D_m = ${Dm}'
        plt.plot(jump_coh, entry['lmb'], marker = 'x',label= label)

    
    plt.xlabel("Cohesvie zone opening [m]", fontsize = 'large')
    plt.ylabel("Stress [Pa] ", fontsize = 'large')
    plt.title("Cohesive Stress [$\sigma$] vs Opening [$\omega$]", fontsize = 'large')
    #plt.legend(fontsize = 'large')
    plt.grid(True)
    plt.show()

def plot_all_dissipation(processed_data):     
    fig, axs = plt.subplots(2, 3, figsize = (12, 8))
    for entry in processed_data :        
        Dm = entry['parameters'].get('Dm', 'unknown')
        alpha = entry['parameters'].get('alpha', 'unknown')        
        functional_choice = entry['parameters'].get('functional_choice','unknown')        
        if functional_choice in ['CZM',  'Exact']:
          continue

        label = f"{functional_choice}, $D_m$ = {Dm}"
        axs[0,0].plot(entry['imposed_disp'], entry['coh_disp_act'] )
        axs[0,1].plot(entry['imposed_disp'], entry['bulk_disp_act'] )
        axs[0,2].plot(entry['imposed_disp'], entry['tot_disp_act'],label = label )
        axs[1,0].plot(entry['imposed_disp'], entry['coh_disp_exp'] )
        if functional_choice in[ 'CLIP-4terms', 'LIP']:        
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

def plot_sep_dissipation(processed_data):
    # Create separate figures for each plot
    fig_coh, ax_coh = plt.subplots(figsize=(6, 5))
    fig_bulk, ax_bulk = plt.subplots(figsize=(6, 5))
    fig_total, ax_total = plt.subplots(figsize=(6, 5))

    # Your existing filtering and data processing code...
    filtered_entries = [
        entry for entry in processed_data 
        if entry['parameters'].get('functional_choice') not in ['CZM', 'Exact','LIP']
    ]
    
    # Plot on each figure separately
    for entry in filtered_entries:
        Gc = entry['parameters'].get('Gc', 'unknown')
        Dm = entry['parameters'].get('Dm', 0.0)
        label = f"$D_m$ = {Dm:.1f}"
        
        # Your color logic...
        if Dm == 0.0:
            color = '#AA0000'
        elif Dm == 0.3:
            color = '#DD2222'
        elif Dm == 0.6:
            color = '#FF5555'
        else:
            color = '#FFAAAA'

        ax_coh.plot(entry['imposed_disp'], entry['coh_disp_act']/Gc*100, color=color, linewidth=2.5)
        ax_bulk.plot(entry['imposed_disp'], entry['bulk_disp_act']/Gc*100, color=color, linewidth=2.5)
        ax_total.plot(entry['imposed_disp'], entry['tot_disp_act']/Gc*100, label=label, color=color, linewidth=2.5)

    # Configure each plot
    # ax_coh.set_title("Cohesive dissipation", fontsize='large')
    ax_coh.set_ylabel("Cohesive dissipation [%]", fontsize = 16)
    ax_coh.set_xlabel("Imposed displacement [m]", fontsize = 16)
    ax_coh.set_xlim(0, 1.25e-4)
    ax_coh.xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    ax_coh.grid(True)

    # ax_bulk.set_title("Bulk dissipation", fontsize='large')
    ax_bulk.set_ylabel("Bulk dissipation [%]", fontsize = 16)
    ax_bulk.set_xlabel("Imposed displacement [m]", fontsize = 16)
    ax_bulk.set_ylim(-5, 105)
    ax_bulk.set_xlim(0, 1.25e-4)
    ax_bulk.xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    ax_bulk.grid(True)

    # ax_total.set_title("Total dissipation", fontsize='large')
    ax_total.set_ylabel("Total dissipation [%]", fontsize = 16)
    ax_total.set_xlabel("Imposed displacement [m]", fontsize = 16)
    ax_total.set_xlim(0, 1.25e-4)
    ax_total.xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    ax_total.grid(True)
    
    # Add legend to total plot
    handles, labels = ax_total.get_legend_handles_labels()
    import re
    def extract_dm(label):
        match = re.search(r'D_m\$ = ([\d\.]+)', label)
        return float(match.group(1)) if match else float('inf')
    
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda hl: extract_dm(hl[1]))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    ax_coh.legend(sorted_handles, sorted_labels, fontsize='large',  loc='best', shadow=True)
    ax_bulk.legend(sorted_handles, sorted_labels, fontsize='large',  loc='best', shadow=True)
    ax_total.legend(sorted_handles, sorted_labels, fontsize='large',  loc='best', shadow=True)

    for ax in (ax_coh, ax_bulk, ax_total):
    #     ax.tick_params(axis='x', labelsize=11, pad=4)
    #     ax.margins(x=0.02)
            sf_x = ScalarFormatter(useMathText=True); sf_x.set_scientific(True); sf_x.set_powerlimits((0,0))
            ax.xaxis.set_major_formatter(sf_x)   # both axes in sci notation (×10^n offset)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))   # fewer x ticks
            ax.margins(x=0.02)
    
    # Save each figure separately
    fig_coh.tight_layout()
    fig_coh.savefig('Dissip_cohesive.png', dpi=400, bbox_inches='tight')
    
    fig_bulk.tight_layout()
    fig_bulk.savefig('Dissip_bulk.png', dpi=400, bbox_inches='tight')
    
    fig_total.tight_layout()
    fig_total.savefig('Dissip_total.png', dpi=400, bbox_inches='tight')
    
    plt.show()

def plot_potential_energy(processed_data):
    for entry in processed_data :         
        functional_choice = entry['parameters'].get('functional_choice','unknown')
        if functional_choice in ['CZM', 'LIP', 'Exact']:
          continue
        Dm = entry['parameters'].get('Dm', 'unknown')
                
        label = f"{functional_choice}, $D_m$ = {Dm}"
        plt.plot(entry['imposed_disp'], entry['potential_energy'], label = label)

    plt.title("Potential Energy [J]", fontsize = 'large', fontweight = 'bold')
    plt.xlabel("Imposed displacement [m]", fontsize = 'large')
    plt.ylabel("Potential Energy [J] ", fontsize = 'large')
    plt.legend(fontsize = 'large')
    plt.grid(True)
    plt.show()

def plot_cohesive_energy(processed_data):
    for entry in processed_data :         
        functional_choice = entry['parameters'].get('functional_choice','unknown')
        if functional_choice in ['CZM', 'LIP', 'Exact']:
          continue
        Dm = entry['parameters'].get('Dm', 'unknown')
                
        label = f"{functional_choice}, $D_m$ = {Dm}"
        plt.plot(entry['imposed_disp'], entry['cohesive_energy'], label = label)

    plt.title("cohesive_Energy [J]", fontsize = 'large', fontweight = 'bold')
    plt.xlabel("Imposed displacement [m]", fontsize = 'large')
    plt.ylabel("Cohesive_Energy [J] ", fontsize = 'large')
    plt.legend(fontsize = 'large')
    plt.grid(True)
    plt.show()

def plot_total_energy(processed_data):
    for entry in processed_data :         
        functional_choice = entry['parameters'].get('functional_choice','unknown')
        if functional_choice in ['CZM', 'LIP', 'Exact']:
          continue
        Dm = entry['parameters'].get('Dm', 'unknown')
                
        label = f"{functional_choice}, $D_m$ = {Dm}"
        plt.plot(entry['imposed_disp'], entry['total_energy'], label = label)

    plt.title("Total Energy [J]", fontsize = 'large', fontweight = 'bold')
    plt.xlabel("Imposed displacement [m]", fontsize = 'large')
    plt.ylabel("Total Energy [J] ", fontsize = 'large')
    plt.legend(fontsize = 'large')
    plt.grid(True)
    plt.show()

def plot_energy_comb(processed_data):
    for entry in processed_data :         
        functional_choice = entry['parameters'].get('functional_choice','unknown')
        if functional_choice in ['CZM', 'LIP', 'Exact']:
          continue
        Dm = entry['parameters'].get('Dm', 'unknown')
                
        label = f"{functional_choice}, $D_m$ = {Dm}"
        plt.plot(entry['imposed_disp'], entry['potential_energy'], label = f'Potential Energy - {label}')
        plt.plot(entry['imposed_disp'], entry['cohesive_energy'], label = f'Cohesive Energy - {label}')
        plt.plot(entry['imposed_disp'], entry['tot_disp_act'], label = f'Dissipated Energy - {label}')
        plt.plot(entry['imposed_disp'], entry['total_energy'], label = f'Total Energy - {label}', color = 'black')

    plt.title("Energy [J]", fontsize = 'large', fontweight = 'bold')
    plt.xlabel("Imposed displacement [m]", fontsize = 'large')
    plt.ylabel("Energy [J] ", fontsize = 'large')
    plt.legend(fontsize = 'large')
    plt.grid(True)
    plt.show()

def plot_all_coh_stress_vs_seperation(processed_data):
    plt.figure(figsize = (10, 6)) 
    
    for entry in processed_data :         
        functional_choice = entry['parameters'].get('functional_choice','unknown')
        if functional_choice in [ 'LIP', 'Exact']:
          continue
        Dm = entry['parameters'].get('Dm', 'unknown')
        Gc = entry['parameters'].get('Gc', 'unknown')
        sigc = entry['parameters'].get('sigc', 'unknown')
        wc = (2*Gc)/(sigc)
        seperation = entry['seperation']
        print(seperation)
        N_elements = entry['parameters'].get('N_elements', 'unknown')
        if functional_choice in ['CZM']:
            jump_coh = [arr[0] for arr in seperation]        
        else:
            jump_coh = [arr[int((N_elements - 1)/2)] for arr in seperation]
        print(jump_coh)
        d_area = entry['stress'].copy()
        sig = [sigc,0]
        w = [0,wc]
        area = np.trapz(d_area,jump_coh)
        if functional_choice in ['CZM']:
            label = f"{functional_choice}, $Area = ${area:.2f}"
        else:
            label = fr'$D_m = ${Dm}, $Area = ${area:.2f}'
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
    plt.tight_layout()
    plt.savefig('Cohesive_stress_vs_opening.png', dpi=300)
    plt.show()

def plot_damage_combine(processed_data):
    # compute max length once
    max_L = max(entry['parameters'].get('L', 0) for entry in processed_data)
    
    for entry in processed_data:
        Dm = entry['parameters'].get('Dm', 'unknown')
        alpha = entry['parameters'].get('alpha', 'unknown')
        functional_choice = entry['parameters'].get('functional_choice', 'unknown')
        if functional_choice in ['CZM', 'LIP', 'Exact']:
            continue

        plt.figure(figsize=(10, 7.5))
        ax = plt.gca()

        ij = [int(len(entry['imposed_disp'])/7.5), int(len(entry['imposed_disp'])/5),
              int(len(entry['imposed_disp'])/2.5), -1]
        bulk = ['0.25', '0.5', '0.75', '']
        
        for i, b in zip(ij, bulk):
            if i < len(entry['cohesive_damage']):
                d_data = entry['cohesive_damage'][i]
                bulk_data = entry['bulk_damage_overall'][i]
                L = entry['parameters'].get('L', 'unknown')

                plt.scatter(L/2, max(d_data), marker='o', s=50,
                            label=f'Cohesive, ($u_t = {b}\\,\\omega_c$)')
                plt.plot(np.linspace(0, L, len(bulk_data)), bulk_data,
                         label=f'Bulk, ($u_t = {b}\\,\\omega_c$)', linewidth=2.5)
                
        plt.axhline(y=Dm, color='red', linestyle='--')
        plt.text(0.05 * max_L, Dm, '$D_m$', color='red', fontsize=12, va='bottom', ha='right')
        plt.xlabel('x [m]', )
        plt.ylabel('Damage [-]',)
        plt.xlim(0, max_L)  # ensure same x range across all plots
        plt.ylim(-0.01, 1.05)      # damage range typically 0–1, fixes y-scale too

        # consistent number formatting
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        plt.legend(shadow=True)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Damage_along_bar_{Dm}.png', dpi=600)
        plt.close()

def plot_all_damge_vs_imposed_disp(processed_data):
    plt.figure(figsize=(10, 6)) 

    for entry in processed_data :

        functional_choice = entry['parameters'].get('functional_choice','unknown')        
        if functional_choice in [ 'LIP', 'Exact']:
          continue

        max_d_str = []
        damage = entry['cohesive_damage']
        max_d_str.extend([np.max(arr) for arr in damage])
        Dm = entry['parameters'].get('Dm', 'unknown')
        alpha = entry['parameters'].get('alpha', 'unknown')      
        if functional_choice in ['CZM']:
            label = f"{functional_choice}"
        else:
            label = f"{functional_choice}, $D_m$={Dm}"

        # plt.plot(entry['imposed_disp'], max_d_str,label = label, marker = 'o')
        plt.plot( max_d_str,entry['stress'],label = label, marker = 'o')

    plt.xlabel("Cohesive Damage [-]", fontsize = 'large')
    plt.ylabel("Stress [Pa]", fontsize = 'large')
    # plt.title('Stress [$d$] vs Cohesive Damage [$d$]', fontsize = 'large')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    plt.grid(True)
    plt.legend(fontsize='large')
    plt.tight_layout()
    plt.savefig('Stress vs cohesive damage.png', dpi=300)
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
        N_elements = entry['parameters'].get('N_elements', 'unknown')
        functional_choice = entry['parameters'].get('functional_choice','unknown')  
        label = functional_choice if functional_choice in ['CZM', 'LIP', 'Exact'] else f"{functional_choice}, $D_m$={Dm}, $\\alpha$={alpha:.3f}"

        axs1.plot(entry['imposed_disp'], entry['stress'], label=label)
        if functional_choice in ['CZM', 'LIP', 'Exact']:
          continue
        seperation = entry['seperation']

        axs2[0,0].plot(entry['imposed_disp'], entry['coh_disp_act'] )
        axs2[0,1].plot(entry['imposed_disp'], entry['bulk_disp_act'] )
        axs2[0,2].plot(entry['imposed_disp'], entry['tot_disp_act'],label = label )
        axs2[1,0].plot(entry['imposed_disp'], entry['coh_disp_exp'] )
        if functional_choice == 'CLIP-4terms':        
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

def plot_functions_vs_damage(d_values):
    functions_gd_std = gd_cohesive_std()
    functions_gd_std_values = functions_gd_std.get_value(d_values)
   
    plt.figure(figsize=(10, 6))
    plt.plot(d_values, functions_gd_std_values, label='gd_cohesive_std')
    
    plt.xlabel('d')
    plt.ylabel('g(d)')
    plt.title('Plot of $g(d)$ against $d$ for gd_cohesive_std')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_func_values_vs_damage(class_names, d_values, additional_params, module_name='functions'):
    plt.figure(figsize=(10, 6))
    
    for class_name in class_names:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance = cls()
        g_values = instance.get_value(d_values)
        # work in progress to handle the extra parameters in functions 
        # if class_name in additional_params:
        #     parameters = additional_params[class_name]['parameters']
        #     instance = cls(parameters)
        #     g_values = instance.get_value(d_values)
        # else :
        #     instance = cls()
        #     g_values = instance.get_value(d_values)
        plt.plot(d_values, g_values, label=f'{class_name}')
        plt.xlabel('d')
        plt.ylabel(f'{class_name}')
        plt.title(f'Plot of {class_name} vs damage')
        plt.legend()
        plt.grid(True)
        plt.show()

def damage_comp_plot(dt, max_steps, d_aka, d_test):

    time = [i * dt for i in range(max_steps)]

    plt.plot(time, d_aka, label='Akantu')
    plt.plot(time, d_test, label='Test')

    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Damage', fontsize = 'large')
    plt.title('Damage comparison over time', fontweight = 'bold', fontsize = 'large')
    plt.legend()
    plt.grid(True)

    formatter = ticker.FormatStrFormatter('%.1e')  
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/damage_comp.png', dpi=300)
    plt.show()

def energies_comparison(results_aka, *results):
    dt = results_aka['time_step']
    max_steps = results_aka['inputs'].get('max_steps')
    time = [i * dt for i in range(max_steps)]

    E_pot_aka = results_aka['potential_energy']
    E_kin_aka = results_aka['kinetic_energy']
    E_dis_aka = results_aka['dissipated_energy']
    E_rev_aka = results_aka['reversible_energy']
    ext_aka = results_aka['external_work']
    tot_aka = results_aka['total_energy']

    plt.figure()
    plt.plot(time, E_pot_aka, label = 'Epot_Akantu',linestyle = '-.',  color = 'C0')
    for result in results :
        E_pot = result['potential_energy']
        plt.plot(time, E_pot, label = fr'Dm = {result["inputs"].get("Dm")}, $\ell_c = \frac{{L}}{{{result["inputs"].get("nlc")}}}$')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Potential Energy (J)', fontsize = 'large')
    plt.title('Potential energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/pot.png', dpi=300)
    #plt.show()

    plt.figure()
    plt.plot(time, E_kin_aka, label = 'Ekin_Akantu',linestyle = '-.', color = 'C2')
    for result in results :
        E_kin = result['kinetic_energy']
        plt.plot(time, E_kin, label = fr'Dm = {result["inputs"].get("Dm")}, $\ell_c = \frac{{L}}{{{result["inputs"].get("nlc")}}}$')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Kinetic Energy (J)', fontsize = 'large')
    plt.title('Kinetic energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/kin.png', dpi=300)
    #plt.show()

    plt.figure()
    linestyles = ['-.', ':', '--']
    plt.plot(time, E_dis_aka, label = 'Dissipation_Akantu', color = 'black')
    for i,result in enumerate(results) :
        E_dis = result['dissipated_energy']
        Edis_bulk_str = result['dissipated_bulk_energy']
        Edis_coh_str = result['dissipated_coh_energy']
        plt.plot(time, E_dis,label = fr'Dm = {result["inputs"].get("Dm")}',color = 'red')
        plt.plot(time, Edis_bulk_str,label = fr'Bulk, Dm = {result["inputs"].get("Dm")}', linestyle=linestyles[i % len(linestyles)]  )
        plt.plot(time, Edis_coh_str,label = fr'Cohesive, Dm = {result["inputs"].get("Dm")}',linestyle=linestyles[i % len(linestyles)])
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Dissipated Energy (J)', fontsize = 'large')
    plt.title('Dissipated energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/dissip.png', dpi=300)
    #plt.show()

    plt.figure()
    linestyles = ['-.', '--',':']
    #colors = ['mediumslateblue', 'tomato', 'seagreen', 'darkorange', 'deepskyblue', 'orchid', 'darkkhaki', 'royalblue', 'coral']

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    plt.plot(time, E_dis_aka, label = 'Edis_Akantu', linestyle = '-.', color = 'black')
    for i, result in enumerate(results) :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_dis = result['dissipated_energy']
        Edis_bulk_str = result['dissipated_bulk_energy']
        Edis_coh_str = result['dissipated_coh_energy']
        E_pot = result['potential_energy']
        E_dis_act = result['total_dissipation_actual']
        Edis_bulk_act_str = result['bulk_dissipation_actual']
        Edis_coh_act_str = result['coh_dissipation_actual']
        Edis_total_act_str = result['total_dissipation_actual']
        #plt.plot(time, E_pot, label = fr'Dm = {result["inputs"].get("Dm")}')
        #plt.plot(time, E_dis_act,label = fr'Dm = {result["inputs"].get("Dm")}')
        #plt.plot(time, E_pot, label = fr'Dm = {result["inputs"].get("Dm")}')
        plt.plot(time, Edis_bulk_act_str,label = fr'Bulk, Dm = {result["inputs"].get("Dm")}', linestyle=linestyles[0], color = colors[i % len(colors)])
        plt.plot(time, Edis_coh_act_str,label = fr'Cohesive, Dm = {result["inputs"].get("Dm")}',linestyle=linestyles[1],color = colors[i % len(colors)] )
        plt.plot(time, Edis_total_act_str,label = fr'Total, Dm = {result["inputs"].get("Dm")}',color = colors[i % len(colors)])

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Dissipated Energy (J)', fontsize = 'large')
    plt.title('Dissipated energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/dissip_act.png', dpi=300)
    #plt.show()

    plt.figure()
    plt.plot(time, E_rev_aka, label = "Ecoh_Akantu", linestyle = '-.', color = 'C4')
    for result in results :
        E_rev = result['cohesive_energy']
        plt.plot(time, E_rev, label = fr'Dm = {result["inputs"].get("Dm")}, $\ell_c = \frac{{L}}{{{result["inputs"].get("nlc")}}}$')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Cohesive Energy (J)', fontsize = 'large')
    plt.title('Cohesive energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/coh.png', dpi=300)
    #plt.show()

    # plt.figure()
    # plt.plot(time, ext_aka, label = "Ext_Akantu", linestyle = '-.', color = 'C1')
    # for result in results :
    #     ext = result['external_work']
    #     plt.plot(time, ext, label = fr'Dm = {result["inputs"].get("Dm")}, $\ell_c = \frac{{L}}{{{result["inputs"].get("nlc")}}}$')
    # plt.legend()
    # plt.grid(True)
    # plt.xlabel('Time (s)', fontsize = 'large')
    # plt.ylabel('External Work (J)', fontsize = 'large')
    # plt.title('External work comparison over time', fontweight = 'bold', fontsize = 'large')
    # formatter = ticker.FormatStrFormatter('%.1e')
    # plt.gca().xaxis.set_major_formatter(formatter)
    # plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/energy_ext.png', dpi=300)
    #plt.show()

    plt.figure()
    plt.plot(time, tot_aka, label = "Tot_akantu", linestyle = '-.', color = 'black')
    for result in results :
        tot = result['total_energy']
        tot_verlet = result['total_energy_verlet']
        #plt.plot(time, tot, label = fr'Dm = {result["inputs"].get("Dm")}, $\ell_c = \frac{{L}}{{{result["inputs"].get("nlc")}}}$')
        plt.plot(time, tot_verlet, label = fr'Verv, Dm = {result["inputs"].get("Dm")}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Total Energy (J)', fontsize = 'large')
    plt.title('Total Energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/tot.png', dpi=300)
    #plt.show()

    plt.figure()
    linestyles = ['-.', '--',':']
    plt.plot(time, E_pot_aka, label = 'Potential, Akantu',  color = 'C1')
    plt.plot(time, E_kin_aka, label = 'Kinetic, Akantu', color = 'C0')
    plt.plot(time, E_dis_aka, label = 'Dissipated, Akantu',  color = 'red')
    plt.plot(time, E_rev_aka, label = "Cohesive, Akantu",  color = 'C2')
    plt.plot(time, tot_aka, label = "Total, ", color = 'black')
    for i, result in enumerate(results) :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_pot = result['potential_energy']
        E_kin = result['kinetic_energy']
        E_dis = result['total_dissipation_actual']
        E_rev = result['cohesive_energy']
        tot = result['total_energy']
        tot_verlet = result['total_energy_verlet']
        plt.plot(time, E_pot, label = fr'Potential ,Dm = {result["inputs"].get("Dm")} ', linestyle=linestyles[i % len(linestyles)], color = 'C1')
        plt.plot(time, E_kin, label = fr'Kinetic, Dm = {result["inputs"].get("Dm")} ',linestyle=linestyles[i % len(linestyles)], color = 'C0')
        plt.plot(time, E_dis, label = fr'Dissipated, Dm = {result["inputs"].get("Dm")} ',linestyle=linestyles[i % len(linestyles)], color = 'red')
        plt.plot(time, E_rev, label = fr'Cohesive, Dm = {result["inputs"].get("Dm")} ', linestyle=linestyles[i % len(linestyles)], color = 'C2')
        plt.plot(time, tot, label = fr'Total, ,Dm = {result["inputs"].get("Dm")} ', linestyle=linestyles[i % len(linestyles)], color = 'black')
        plt.plot(time, tot_verlet, label = fr'total_energy_verlet, ,Dm = {result["inputs"].get("Dm")} ', linestyle=linestyles[i % len(linestyles)])
    
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Energy Comparison(J)', fontsize = 'large')
    plt.title('Energies comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/energy_comp.png', dpi=300)
    plt.show()

def energies_comparison_string(dt,results_lip,*results):
    
    t_vect = results_lip['t_vect']
    elastic_energy_vect =results_lip['elastic_energy_vect']
    kinetic_energy_vect =results_lip['kinetic_energy_vect']
    dissipated_energy_vect =results_lip['dissipated_energy_vect']
    bulk_dissipation_actual =results_lip['bulk_dissipation_actual']
    Total_energy = results_lip['Total_energy']
    Total_energy_actual =results_lip['Total_energy_actual']

    plt.figure()
    plt.plot(t_vect, elastic_energy_vect, label = 'LIP', color = 'black')
    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_pot = result['potential_energy']
        plt.plot(time, E_pot, label = fr'{result["method"]}, Dm = {result["inputs"].get("Dm")}, $\gamma = {result["inputs"].get("gamma")}$')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Potential Energy (J)', fontsize = 'large')
    plt.title('Potential energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/pot.png', dpi=300)
    #plt.show()

    plt.figure()
    plt.plot(t_vect, kinetic_energy_vect, label = 'LIP',color = 'black')
    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_kin = result['kinetic_energy']
        plt.plot(time, E_kin, label = fr'{result["method"]},Dm = {result["inputs"].get("Dm")}, $\gamma = {result["inputs"].get("gamma")}$')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Kinetic Energy (J)', fontsize = 'large')
    plt.title('Kinetic energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/kin.png', dpi=300)
    #plt.show()

    plt.figure()
    plt.plot(t_vect, dissipated_energy_vect, label = 'LIP', color = 'black')
    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_dis = result['dissipated_energy']
        Edis_bulk_str = result['dissipated_bulk_energy']
        Edis_coh_str = result['dissipated_coh_energy']
        plt.plot(time, E_dis,label = fr'{result["method"]},,Dm = {result["inputs"].get("Dm")}, $\gamma = {result["inputs"].get("gamma")}$')
        plt.plot(time, Edis_bulk_str, label = fr'{result["method"]}, Bulk, ,Dm = {result["inputs"].get("Dm")}, $\gamma = {result["inputs"].get("gamma")}$', linestyle = '--')
        plt.plot(time, Edis_coh_str,label = fr'{result["method"]}, Cohesive, ,Dm = {result["inputs"].get("Dm")}, $\gamma = {result["inputs"].get("gamma")}$',linestyle = '--')
        
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Dissipated Energy (J)', fontsize = 'large')
    plt.title('Dissipated energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/dissip.png', dpi=300)
    #plt.show()

    plt.figure()
    linestyles = ['--',':']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    plt.plot(t_vect, bulk_dissipation_actual, label = 'LIP', color = 'black')
    for i, result in enumerate(results) :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_dis_act = result['total_dissipation_actual']
        Edis_bulk_act_str = result['bulk_dissipation_actual']
        Edis_coh_act_str = result['coh_dissipation_actual']

        plt.plot(time, E_dis_act,label = fr'{result["method"]}, Dm = {result["inputs"].get("Dm")}', color = colors[i % len(colors)])
        plt.plot(time, Edis_bulk_act_str, label = fr'{result["method"]}, Bulk, ,Dm = {result["inputs"].get("Dm")}, $\gamma = {result["inputs"].get("gamma")}$', linestyle = linestyles[0] ,color = colors[i % len(colors)])
        plt.plot(time, Edis_coh_act_str,label = fr'{result["method"]}, Cohesive, ,Dm = {result["inputs"].get("Dm")}, $\gamma = {result["inputs"].get("gamma")}$',linestyle = linestyles[1], color = colors[i % len(colors)])
        
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Dissipated Energy (J)', fontsize = 'large')
    plt.title('Dissipated energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/dissip.png', dpi=300)
    #plt.show()

    plt.figure()
    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_rev = result['cohesive_energy']
        plt.plot(time, E_rev, label = fr'{result["method"]},Dm = {result["inputs"].get("Dm")}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Cohesive Energy (J)', fontsize = 'large')
    plt.title('Cohesive energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/coh.png', dpi=300)
    #plt.show()

    plt.figure()
    plt.plot(t_vect, Total_energy, label = 'LIP')
    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        tot = result['total_energy']
        tot_verlet = result['total_energy_verlet']
        plt.plot(time, tot, label = fr'{result["method"]},Dm = {result["inputs"].get("Dm")}')
        plt.plot(time, tot_verlet, label = fr'verv, {result["method"]},Dm = {result["inputs"].get("Dm")}')
        
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Total Energy (J)', fontsize = 'large')
    plt.title('Total Energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/tot.png', dpi=300)
    #plt.show()

    plt.figure()
    linestyles = ['-', '--',':'] 
    for i, result in enumerate(results) :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_pot = result['potential_energy']
        E_kin = result['kinetic_energy']
        E_dis = result['dissipated_energy']
        E_rev = result['cohesive_energy']
        tot = result['total_energy']
        plt.plot(time, E_pot, label = fr'Potential ,Dm = {result["inputs"].get("Dm")} ', linestyle=linestyles[i % len(linestyles)], color = 'C1')
        plt.plot(time, E_kin, label = fr'Kinetic, Dm = {result["inputs"].get("Dm")} ',linestyle=linestyles[i % len(linestyles)], color = 'C0')
        plt.plot(time, E_dis, label = fr'Dissipated, Dm = {result["inputs"].get("Dm")} ',linestyle=linestyles[i % len(linestyles)], color = 'red')
        plt.plot(time, E_rev, label = fr'Cohesive, Dm = {result["inputs"].get("Dm")} ', linestyle=linestyles[i % len(linestyles)], color = 'C2')
        plt.plot(time, tot, label = fr'Total, ,Dm = {result["inputs"].get("Dm")} ', linestyle=linestyles[i % len(linestyles)], color = 'black')
    
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Energy Comparison(J)', fontsize = 'large')
    plt.title('Energies comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/energy_comp.png', dpi=300)
    plt.show()

def energies_comparison_free(dt,*results):

    plt.figure()
    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_pot = result['potential_energy']
        plt.plot(time, E_pot, label = fr'{result["method"]}, Dm = {result["inputs"].get("Dm")}, $\gamma = {result["inputs"].get("gamma")}$')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Potential Energy (J)', fontsize = 'large')
    plt.title('Potential energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/pot.png', dpi=300)
    #plt.show()

    plt.figure() 
    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_kin = result['kinetic_energy']
        plt.plot(time, E_kin, label = fr'{result["method"]},Dm = {result["inputs"].get("Dm")}, $\gamma = {result["inputs"].get("gamma")}$')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Kinetic Energy (J)', fontsize = 'large')
    plt.title('Kinetic energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/kin.png', dpi=300)
    #plt.show()

    plt.figure()
    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_dis = result['dissipated_energy']
        Edis_bulk_str = result['dissipated_bulk_energy']
        Edis_coh_str = result['dissipated_coh_energy']
        plt.plot(time, E_dis,label = fr'{result["method"]},,Dm = {result["inputs"].get("Dm")}, $\gamma = {result["inputs"].get("gamma")}$')
        plt.plot(time, Edis_bulk_str, label = fr'{result["method"]}, Bulk, ,Dm = {result["inputs"].get("Dm")}, $\gamma = {result["inputs"].get("gamma")}$', linestyle = '--')
        plt.plot(time, Edis_coh_str,label = fr'{result["method"]}, Cohesive, ,Dm = {result["inputs"].get("Dm")}, $\gamma = {result["inputs"].get("gamma")}$',linestyle = '--')
        
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Dissipated Energy (J)', fontsize = 'large')
    plt.title('Dissipated energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/dissip.png', dpi=300)
    #plt.show()

    plt.figure()
    linestyles = ['--',':']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']    
    for i, result in enumerate(results) :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_dis_act = result['total_dissipation_actual']
        Edis_bulk_act_str = result['bulk_dissipation_actual']
        Edis_coh_act_str = result['coh_dissipation_actual']

        plt.plot(time, E_dis_act,label = fr'{result["method"]}, Dm = {result["inputs"].get("Dm")}', color = colors[i % len(colors)])
        plt.plot(time, Edis_bulk_act_str, label = fr'{result["method"]}, Bulk, ,Dm = {result["inputs"].get("Dm")}, $\gamma = {result["inputs"].get("gamma")}$', linestyle = linestyles[0] ,color = colors[i % len(colors)])
        plt.plot(time, Edis_coh_act_str,label = fr'{result["method"]}, Cohesive, ,Dm = {result["inputs"].get("Dm")}, $\gamma = {result["inputs"].get("gamma")}$',linestyle = linestyles[1], color = colors[i % len(colors)])
        
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Dissipated Energy (J)', fontsize = 'large')
    plt.title('Dissipated energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/dissip.png', dpi=300)
    #plt.show()

    plt.figure()
    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_rev = result['cohesive_energy']
        plt.plot(time, E_rev, label = fr'{result["method"]},Dm = {result["inputs"].get("Dm")}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Cohesive Energy (J)', fontsize = 'large')
    plt.title('Cohesive energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/coh.png', dpi=300)
    #plt.show()

    plt.figure()
    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        tot = result['total_energy']
        tot_verlet = result['total_energy_verlet']
        plt.plot(time, tot, label = fr'{result["method"]},Dm = {result["inputs"].get("Dm")}')
        plt.plot(time, tot_verlet, label = fr'verv, {result["method"]},Dm = {result["inputs"].get("Dm")}')
        
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Total Energy (J)', fontsize = 'large')
    plt.title('Total Energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/tot.png', dpi=300)
    #plt.show()

    plt.figure()
    linestyles = ['-', '--',':'] 
    for i, result in enumerate(results) :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_pot = result['potential_energy']
        E_kin = result['kinetic_energy']
        E_dis = result['dissipated_energy']
        E_rev = result['cohesive_energy']
        tot = result['total_energy']
        plt.plot(time, E_pot, label = fr'Potential ,Dm = {result["inputs"].get("Dm")} ', linestyle=linestyles[i % len(linestyles)], color = 'C1')
        plt.plot(time, E_kin, label = fr'Kinetic, Dm = {result["inputs"].get("Dm")} ',linestyle=linestyles[i % len(linestyles)], color = 'C0')
        plt.plot(time, E_dis, label = fr'Dissipated, Dm = {result["inputs"].get("Dm")} ',linestyle=linestyles[i % len(linestyles)], color = 'red')
        plt.plot(time, E_rev, label = fr'Cohesive, Dm = {result["inputs"].get("Dm")} ', linestyle=linestyles[i % len(linestyles)], color = 'C2')
        plt.plot(time, tot, label = fr'Total, ,Dm = {result["inputs"].get("Dm")} ', linestyle=linestyles[i % len(linestyles)], color = 'black')
    
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Energy Comparison(J)', fontsize = 'large')
    plt.title('Energies comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/energy_comp.png', dpi=300)
    plt.show()

def energy_comp_plot(max_steps, results_aka, results_opti_czm = None):

    dt = results_aka['time_step']
    time = [i * dt for i in range(max_steps)]

    E_pot = results_aka['potential_energy']
    E_kin = results_aka['kinetic_energy']
    E_dis = results_aka['dissipated_energy']
    E_rev = results_aka['reversible_energy']
    ext = results_aka['external_work']
    tot = results_aka['total_energy']

    Ep_str = results_opti_czm['potential_energy']
    Ekin_str = results_opti_czm['kinetic_energy']
    Edis_str = results_opti_czm['dissipated_energy']
    Edis_bulk_str = results_opti_czm['dissipated_bulk_energy']
    Edis_coh_str = results_opti_czm['dissipated_coh_energy']
    Ecoh_str = results_opti_czm['cohesive_energy']
    ext_str = results_opti_czm['external_work']
    tot_str = results_opti_czm['total_energy']

    plt.plot(time, E_pot, label = 'Epot_Akantu',linestyle = '-.',  color = 'C0')
    plt.plot(time, Ep_str, label='Potential', color = 'C0')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Energy (J)', fontsize = 'large')
    plt.title('Energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/pot.png', dpi=300)
    plt.show()

    plt.plot(time, E_kin, label = 'Ekin_Akantu',linestyle = '-.', color = 'C2')
    plt.plot(time, Ekin_str, label = 'Kinetic',color = 'C2')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Energy (J)', fontsize = 'large')
    plt.title('Energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/kin.png', dpi=300)
    plt.show()

    plt.plot(time, E_dis, label = 'Edis_Akantu', linestyle = '-.', color = 'C3')
    plt.plot(time, Edis_str, label = 'Dissipated', color = 'C3')
    plt.plot(time, Edis_bulk_str, label = 'Dissipated bulk', linestyle = '--')
    plt.plot(time, Edis_coh_str, label = 'Dissipated cohesive',linestyle = '--')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Energy (J)', fontsize = 'large')
    plt.title('Energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/dissip.png', dpi=300)
    plt.show()

    plt.plot(time, E_rev, label = "Ecoh_Akantu", linestyle = '-.', color = 'C4')
    plt.plot(time, Ecoh_str, label = "Cohesive", color = 'C4')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Energy (J)', fontsize = 'large')
    plt.title('Energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/coh.png', dpi=300)
    plt.show()

    plt.plot(time, ext, label = "Ext_Akantu", linestyle = '-.', color = 'C1')
    plt.plot(time, ext_str, label = "External Work", color = 'C1')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Energy (J)', fontsize = 'large')
    plt.title('Energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/energy_cextomp.png', dpi=300)
    plt.show()

    plt.plot(time, tot, label = "Tot_akantu", linestyle = '-.', color = 'black')
    plt.plot(time, tot_str, label = "Total",marker='o', color = 'black')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Energy (J)', fontsize = 'large')
    plt.title('Energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/tot.png', dpi=300)
    plt.show()

    # plt.plot(time, E_pot, label='Potential', color = 'C0')
    # plt.plot(time, E_kin, label = 'Kinetic', color = 'C2')
    # plt.plot(time, E_dis, label = 'Dissipated', color = 'C3')
    # plt.plot(time, E_rev, label = "Cohesive", color = 'C4')
    # plt.plot(time, tot, label = "Total", color = 'black')
    # plt.legend()
    # plt.grid(True)
    # plt.xlabel('Time (s)', fontsize = 'large')
    # plt.ylabel('Energy (J)', fontsize = 'large')
    # plt.title('Energy comparison over time', fontweight = 'bold', fontsize = 'large')
    # formatter = ticker.FormatStrFormatter('%.1e')
    # plt.gca().xaxis.set_major_formatter(formatter)
    # plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/energy_comp.png', dpi=300)
    # plt.show()
    #linestyle='-', marker='o'
    plt.plot(time, Ep_str, label='Potential', color = 'C0')
    plt.plot(time, Ekin_str, label = 'Kinetic', color = 'C2')
    plt.plot(time, Edis_str, label = 'Dissipated', color = 'C3')
    plt.plot(time, Ecoh_str, label = "Cohesive", color = 'C4')
    #plt.plot(time, ext_str, label = "External Work", color = 'C1')
    plt.plot(time, tot_str, label = "Total", color = 'black')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Energy (J)', fontsize = 'large')
    plt.title('Energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/energy_comp.png', dpi=300)
    plt.show()

def tract_sep_law(results_no_opti_clip):
    opening = results_no_opti_clip['opening']
    traction = results_no_opti_clip['traction']
    plt.plot(opening, traction, marker = 'o')
    plt.plot([0 ,results_no_opti_clip['inputs'].get('wc') ], [results_no_opti_clip['inputs'].get('sigc'), 0 ], color = 'black')
    plt.xlabel('Opening (m)', fontsize = 'large')
    plt.ylabel('Traction (Pa)', fontsize = 'large')
    plt.title('Traction - Seperation Law', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.grid(True)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/tsl.png', dpi=300)
    
    plt.show()

def bulk_damage_at_nodes(results_no_opti_clip) :
    bulk_damage_nodes_str = results_no_opti_clip['bulk_damage_nodes']
    x_test = np.linspace(results_no_opti_clip['inputs'].get('dx'), results_no_opti_clip['inputs'].get('L')-results_no_opti_clip['inputs'].get('dx'), (results_no_opti_clip['inputs'].get('N_elements')-1))
    x_test = np.linspace(0., results_no_opti_clip['inputs'].get('L'), (results_no_opti_clip['inputs'].get('N_elements')+1))
    print(x_test, )
    for i in (bulk_damage_nodes_str):
        plt.plot(x_test, i)
    
    plt.xlabel('Along the bar (m)', fontsize = 'large')
    plt.ylabel('Bulk Damage', fontsize = 'large')
    plt.title('Bulk Damage at the nodes', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    #plt.gca().xaxis.set_major_formatter(formatter)
    plt.grid(True)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/bulk_damage.png', dpi=300)
    
    plt.show()
    
def strain_at_timestep(dt, time_index, *results):
    plt.figure()
    
    for result in results:
        st = np.array(result['strain_str']).T
        col_data = [strain[time_index] for strain in st]
        
        L = result['inputs'].get('L')
        N_elements = result['inputs'].get('N_elements')

        plt.plot(np.linspace(0, L, N_elements), col_data, marker='o', label=f'Dm = {result["inputs"].get("Dm")}',)
    
    plt.xlabel('Along the bar (m)', fontsize='large')
    plt.ylabel('Strain', fontsize='large')
    plt.title(f'Strain along the bar at timestep {dt*time_index: .3e}', fontweight='bold', fontsize='large')
    
    formatter = ticker.FormatStrFormatter('%.2e')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.grid(True)
    plt.legend()
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/strain.png', dpi=300)
    
    plt.show()

def opening_along_time(dt, *results):
    plt.figure()

    for result in results:
        Time = [i * dt for i in range(0, result['inputs'].get('max_steps'))]
        #plt.plot(Time, result['opening'], marker = 'o', label=f'Dm = {result["inputs"].get("Dm")}')
        plt.plot(Time, result['strain_change_str'], marker = 'o', label=f'Dm = {result["inputs"].get("Dm")}')

    plt.xlabel('Time (s)', fontsize='large')
    plt.ylabel('Opening (m)', fontsize='large')
    plt.title(f'Opening ($\omega$) vs Time (t)', fontweight='bold', fontsize='large')
    formatter = ticker.FormatStrFormatter('%.2e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.grid(True)
    plt.legend()
    plt.show()

def GD_function_value_along_bar(dt, time_index, *results):
    plt.figure()

    for result in results:
        L = result['inputs'].get('L')
        N_elements = result['inputs'].get('N_elements')

        plt.plot(np.linspace(0, L, N_elements-1), result['GD_function'][time_index], label=f'Dm = {result["inputs"].get("Dm")}', marker = 'o')

    plt.xlabel('Along the bar (m)', fontsize='large')
    plt.ylabel('G(D) value', fontsize='large')
    plt.title(f'G(D) value along the bar at time step {dt*time_index: .3e}', fontweight='bold', fontsize='large')
    
    plt.grid(True)
    plt.legend()
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/GD_value.png', dpi=300)
    
    plt.show()

def Stress_along_bar_at_time_step(dt, time_index,*results ):
    plt.figure()

    for result in results:
        L = result['inputs'].get('L')
        N_elements = result['inputs'].get('N_elements')
        col_index = len(result['velocity'])

        #plt.plot(result['nodes_str'], result['stress_str'][time_index], label=f'Dm = {result["inputs"].get("Dm")}')
        plt.plot(np.linspace(0, L, N_elements +1 ), result['stress_crack_str'][time_index], label=f'Dm = {result["inputs"].get("Dm")}')

    plt.xlabel('Along the bar (m)', fontsize='large')
    plt.ylabel('Stress (N/m^2)', fontsize='large')
    plt.title(f'Stress along the bar at time step {dt*( time_index): .3e}', fontweight='bold', fontsize='large')
    plt.grid(True)
    plt.legend()
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/stress.png', dpi=300)
    plt.show()

def velocity_along_bar_at_timestep(dt, time_index, *results):
    plt.figure()

    for result in results:
        col_index = len(result['velocity'])

        plt.plot(result['nodes_str'], result['velocity'][time_index], label=f'Dm = {result["inputs"].get("Dm")}')
        #plt.plot(result['nodes_str_crack'], result['velocity_crack'][time_index], label=f'Dm = {result["inputs"].get("Dm")}')

    plt.xlabel('Along the bar (m)', fontsize='large')
    plt.ylabel('Velocity (m/s)', fontsize='large')
    plt.title(f'Velocity along the bar at time step {dt*( time_index): .3e}', fontweight='bold', fontsize='large')
    plt.grid(True)
    plt.legend()
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/vel.png', dpi=300)
    plt.show()

def displacement_along_bar_at_timestep(dt, time_index, *results):
    plt.figure()

    for result in results:
        col_index = len(result['displacement'])

        #plt.plot(result['nodes_str_crack'], result['displacement_crack'][time_index], label=f'Dm = {result["inputs"].get("Dm")}')
        plt.plot(result['nodes_str'], result['displacement'][time_index], label=f'Dm = {result["inputs"].get("Dm")}')

    plt.xlabel('Along the bar (m)', fontsize='large')
    plt.ylabel('Displacement (m)', fontsize='large')
    plt.title(f'Displacement along the bar at time step {dt*( time_index): .3e}', fontweight='bold', fontsize='large')
    plt.grid(True)
    plt.legend()
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/disp.png', dpi=300)
    plt.show()

def acceleration_along_bar_at_timestep(dt, time_index, *results):
    plt.figure()

    for result in results:
        L = result['inputs'].get('L')
        N_elements = result['inputs'].get('N_elements')
        col_index = len(result['acceleration'])

        #plt.plot(result['nodes_str_crack'], result['acceleration_crack'][time_index], label=f'Dm = {result["inputs"].get("Dm")}')
        plt.plot(result['nodes_str'], result['acceleration'][time_index], label=f'Dm = {result["inputs"].get("Dm")}')

    plt.xlabel('Along the bar (m)', fontsize='large')
    plt.ylabel(f'Acceleration $(m/s^2)$', fontsize='large')
    plt.title(f'Acceleration along the bar at time step {dt*(time_index): .3e}', fontweight='bold', fontsize='large')
    plt.grid(True)
    plt.legend()
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/acc.png', dpi=300)
    plt.show()

def velocity_along_the_bar(*results):
    plt.figure()

    for i, result in enumerate(results) :
        if i%20 == 0 :
            # plt.plot(result['nodes_str'], np.array(result['velocity']).T)
            # plt.plot(result['nodes_str_crack'], np.array(result['velocity_crack']).T)
            
            plt.plot(result['nodes_str'][1:],np.array(result['strain_change_str']).T)
            plt.xlabel('Along the bar (m)', fontsize='large')
            plt.ylabel('Velocity (m/s)', fontsize='large')
            plt.title('Velocity (v) along the bar', fontweight='bold', fontsize='large')
            formatter = ticker.FormatStrFormatter('%.2e')
            plt.gca().yaxis.set_major_formatter(formatter)
            plt.grid(True)
            plt.legend()
            plt.show()

def velocity_along_the_bar_3D(dt, *results):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for result in results:
        total_time = (result['inputs'].get('max_steps') * dt )
        len_nodes_str = len(result['velocity'])  # Length of nodes_str
        len_nodes_crack = len(result['velocity_crack'])  # Length of nodes_str_crack
        total_length = len_nodes_str + len_nodes_crack  # Total length of both node sets

        # Time split based on the length of nodes
        time_split_1 = total_time * (len_nodes_str / total_length)
        time_split_2 = total_time * (len_nodes_crack / total_length)

        time_steps_1 = np.linspace(0, time_split_1, len_nodes_str)  # Time steps for nodes_str
        time_steps_2 = np.linspace(time_split_1, total_time, len_nodes_crack)  # Time steps for nodes_str_crack
        
        # Plot for nodes_str
        if 'nodes_str' in result and 'velocity' in result:
            
            nodes = result['nodes_str']
            velocity = (result['velocity'])
            T, X = np.meshgrid(time_steps_1, np.arange(len(nodes)))
            ax.plot_surface(X, T, np.vstack(velocity).T, cmap='viridis', alpha=0.7)

        # Plot for nodes_str_crack
        if 'nodes_str_crack' in result and 'velocity_crack' in result:
            nodes_crack = result['nodes_str_crack']
            velocity_crack = np.array(result['velocity_crack'], dtype=object)
            T_crack, X_crack = np.meshgrid(time_steps_2, np.arange(len(nodes_crack)))
            ax.plot_surface(X_crack, T_crack, np.vstack(velocity_crack).T, cmap='viridis', alpha=0.7)

    ax.set_xlabel('Along the bar (m)')
    ax.set_ylabel('Time (s)')
    ax.set_zlabel('Velocity (m/s)')
    ax.set_title('3D Velocity Along the Bar Over Time')
    plt.show()

def stress_strain_plot(*results):

    plt.figure()
    element_index = [39] 
    for result in results:
        for element in element_index:
            stress_list = np.array(result['stress_str'])
            strain_list = np.array(result['strain_str'])

            stress_for_element = [stress[element] for stress in stress_list]
            strain_for_element = [strain[element] for strain in strain_list]

            # print("Stress for element= ", stress_for_element)
            # print("Strain for element= ", strain_for_element)

            plt.plot(strain_for_element, stress_for_element,  label=f'Dm = {result["inputs"].get("Dm")}, el = {element}')

    plt.xlabel('strain', fontsize='large')
    plt.ylabel('Stress', fontsize='large')
    plt.title('Stress-Strain Relationship', fontweight='bold', fontsize='large')
    plt.grid(True)
    plt.legend()
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/stress_strain.png', dpi=300)
    plt.show()

def plot_l2_norm(processed_data):
    mesh_styles = {
        '5':  {'color': 'C0', 'linestyle': '--', 'marker': 's', 'zorder': 2},
        '10': {'color': 'C1', 'linestyle': '--', 'marker': 'd', 'zorder': 3},
        '20': {'color': 'C2', 'linestyle': '--', 'marker': 'o', 'zorder': 4},
        '40': {'color': 'C3', 'linestyle': '--', 'marker': 'x', 'zorder': 5},
    }

    dm_groups = {}

    for entry in processed_data:
        params = entry['parameters']
        he = params.get('he')
        L = params.get('L')
        lc = L / 4
        E = params.get('E')
        Dm = params.get('Dm')

        functional_choice = params.get('functional_choice', '')
        if functional_choice == 'Exact':
            continue

        sigc = params.get('sigc')
        Gc = params.get('Gc')
        wc = (2 * Gc) / sigc

        x = entry['imposed_disp']
        y = entry['stress']

        opening = [(xi - (sigc * L) / E) / (1 - (sigc * L) / (E * wc)) for xi in x]
        opening = [val if val >= 0 else 0 for val in opening]

        exact_stress = [sigc * (1 - (xi / wc)) for xi in opening]
        exact_stress = [val if val >= 0 else 0 for val in exact_stress]

        if len(exact_stress) > 0:
            exact_stress[0] = 0

        exact_stress = np.array(exact_stress)
        y = np.array(y)

        if len(exact_stress) != len(y):
            print("Warning: length mismatch between exact stress and numerical stress")
            continue

        diff = exact_stress - y
        l2 = np.linalg.norm(diff) / np.linalg.norm(exact_stress)

        x_val = lc / he

        if Dm not in dm_groups:
            dm_groups[Dm] = {'x': [], 'l2': [], 'he': []}
        dm_groups[Dm]['x'].append(x_val)
        dm_groups[Dm]['l2'].append(l2)
        dm_groups[Dm]['he'].append(str(int(he)) if he is not None else '')

    plt.figure()

    default_colors = [f'C{i}' for i in range(len(dm_groups))]

    for idx, (Dm_val, data) in enumerate(sorted(dm_groups.items())):
        sorted_indices = sorted(range(len(data['he'])), key=lambda i: float(data['he'][i]))
        x_sorted = np.array([data['x'][i] for i in sorted_indices])
        l2_sorted = np.array([data['l2'][i] for i in sorted_indices])
        he_sorted = [data['he'][i] for i in sorted_indices]

        color = default_colors[idx]

        # Scatter points
        for i, he_val in enumerate(he_sorted):
            style = mesh_styles.get(he_val, {'marker':'o'})
            plt.scatter(x_sorted[i], l2_sorted[i], color=color, marker=style['marker'], zorder=style['zorder'])

        # Line
        plt.plot(x_sorted, l2_sorted, label=f"$D_m = {Dm_val}$", color=color)

        # Compute convergence rate (slope in log-log space)
        # log_x = np.log(x_sorted)
        # log_y = np.log(l2_sorted)
        # slope, _ = np.polyfit(log_x, log_y, 1)
        # convergence_rate = slope  # negative because x = lc/he
        # print(f"Convergence rate for Dm={Dm_val}: {convergence_rate:.3f}")

        # # Display on plot
        # plt.text((x_sorted[1]+ x_sorted[0])/2, (l2_sorted[0] +l2_sorted[1])/2, f"p≈{convergence_rate:.2f}", color=color,
        #          fontsize=14, verticalalignment='top')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.xaxis.get_major_formatter().set_scientific(True)
    ax.xaxis.get_major_formatter().set_powerlimits((-1,1))

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((-1,1))

    plt.xlabel(r'$h_e \,[m]$')
    plt.ylabel(r'$\eta_{err}$ [-]')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig('L2_norm.png', dpi=600)
    plt.show()

def plot_displacement(processed_data):
    plt.figure()
    
    for entry in processed_data:
        L = entry['parameters'].get('L')
        sigc = entry['parameters'].get('sigc')
        Gc = entry['parameters'].get('Gc')
        wc = (2 * Gc) / sigc
        E = entry['parameters'].get('E')
        N_elements = entry['parameters'].get('N_elements')

        ut = entry['imposed_disp']
        stress = entry['stress']
        w = (ut - (stress*L)/E)
        disp_right = ut
        disp_left = np.zeros_like(disp_right)
        disp_exact_str = []
        for i in range(len(ut)):
            if i % 1 == 0:
                disp = [disp_left[i], ((disp_right[i] - w[i])/2) ,(disp_right[i] -  (disp_right[i] - w[i])/2) ,disp_right[i]]
                # disp_left = np.linspace(0, disp[1], N_elements//2 +1)
                # disp_right = np.linspace(disp[2], ut[i], N_elements//2 +1)
                # disp_exact = np.concatenate((disp_left, disp_right[1:]))
                # print(len(disp_exact), N_elements)
                disp_exact_str.append(np.array(disp))

        print("disp_exact= ", disp_exact_str)
        print("disp_czm = ", entry['displacement'])

    for entry in processed_data:
        params = entry['parameters']
        he = params.get('he')
        L = params.get('L')
        Dm = params.get('Dm')
        for i in range(len(entry['displacement'])):
            # print(len(entry['displacement'][i]))
            if i % 1 == 0:
                plt.plot(np.linspace(0, L, len(entry['displacement'][0])), entry['displacement'][i],marker = 'o', label=f'Dm = {Dm}, he = {he}, step = {i}')

    plt.xlabel('Imposed Displacement (m)', fontsize='large')
    plt.ylabel('Stress (Pa)', fontsize='large')
    plt.title('Stress vs Imposed Displacement', fontweight='bold', fontsize='large')
    plt.grid(True)
    # plt.legend()
    plt.savefig('stress_vs_displacement.png', dpi=600)
    plt.show()

def collapse_split_nodes(u_split, jump_idx=None):
    """
    Collapse duplicated internal nodes (split nodes) by taking one value,
    but keep the first and last nodes unchanged.
    The node at jump_idx (cohesive interface) is not collapsed.
    """
    n_split = len(u_split)
    if n_split < 3:
        return np.array(u_split)  # nothing to collapse

    u_collapsed = [u_split[0]]  # keep first node

    i = 1
    while i < n_split - 1:
        # Skip averaging at jump index
        if jump_idx is not None and (i == jump_idx or i+1 == jump_idx):
            u_collapsed.append(u_split[i])
            i += 1
        elif i+1 < n_split - 1:
            # average duplicated node
            u_collapsed.append((u_split[i] + u_split[i+1])/2)
            i += 2
        else:
            u_collapsed.append(u_split[i])
            i += 1

    u_collapsed.append(u_split[-1])  # keep last node
    return np.array(u_collapsed)

def compute_global_relative_L2(entry):
    disp_czm = np.asarray(entry['displacement'])   # shape: (n_steps, n_nodes_split)
    ut_array = np.asarray(entry['imposed_disp'])
    stress = np.asarray(entry['stress'])
    E = entry['parameters'].get('E', 1.0)
    L = entry['parameters'].get('L', 1.0)
    N_elements = entry['parameters'].get('N_elements')
    n_steps, n_nodes_split = disp_czm.shape
    n_nodes_true = N_elements+1  # true mesh nodes

    h_e = L / N_elements  # element length

    # --- Function to compute element-wise L2 for numerator ---
    def elementwise_l2(u_split, u_exact):

        l2 = 0.0
        N_elements_local =int( len(u_exact))-1
        for e in range(N_elements_local):        
            u_left = u_split[e]
            u_right = u_split[e+1]

            u_ex_left = u_exact[e]
            u_ex_right = u_exact[e+1]

            # Trapezoidal integration over element
            l2 += (h_e / 2) * ((u_left - u_ex_left)**2 + (u_right - u_ex_right)**2)

        return l2

    num_acc = 0.0
    den_acc = 0.0

    for i in range(n_steps):
        ut = float(ut_array[i])

        # --- Construct exact displacement (piecewise linear)
        x_nodes = np.linspace(0, L, n_nodes_true)
        u_exact = np.where(x_nodes <= L/2,
                           (stress[i]*x_nodes)/E,
                           ((stress[i]*L)/(2*E))+ (ut -(stress[i]*L)/(E) ) + ((stress[i]*(x_nodes - L/2))/E))

        u_num = disp_czm[i, :]  # CZM split nodes
        u_num_collapsed = collapse_split_nodes(u_num, jump_idx=len(u_num)//2)

        # --- accumulate numerator using split nodes
        num_acc += elementwise_l2(u_num_collapsed, u_exact)

        # --- accumulate denominator using exact solution directly
        den_acc += np.trapz(u_exact**2, x=x_nodes)

    # --- Relative L2 norm ---
    global_rel_L2 = np.sqrt(num_acc) / np.sqrt(den_acc) if den_acc > 0 else np.nan
    return global_rel_L2

def plot_disp_L2(processed_data):
    """
    Plot relative L2 norm of displacement field versus mesh refinement for different Dm.
    """
    results = []

    for entry in processed_data:
        params = entry['parameters']
        he = params.get('he', 1.0)  # element size (or characteristic size)
        Dm = params.get('Dm', 0.0)
        L = params.get('L', 1.0)
        lc = L/4
        rel_L2 = compute_global_relative_L2(entry)
        print(f"he={lc/he:.5g}, Dm={Dm}, rel_L2={rel_L2:.5g}")

        results.append({"he": lc/he, "Dm": Dm, "rel_L2": rel_L2})

    # --- Convert to DataFrame and sort ---
    df = pd.DataFrame(results).sort_values(by=["Dm", "he"])

    # --- Plot ---
    plt.figure(figsize=(7, 5))
    for Dm, group in df.groupby("Dm"):
        plt.plot(group["he"], group["rel_L2"], marker="o", label=f"$D_m = ${Dm}")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.xaxis.get_major_formatter().set_scientific(True)
    ax.xaxis.get_major_formatter().set_powerlimits((-1,1))

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((-1,1))
    plt.xlabel(r'$h_e \,[m]$')
    plt.ylabel(r'$\eta_{err}$ [-]')
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig('disp_L2_norm.png', dpi=600)
    plt.show()


if __name__ == "__main__":
    ################################################################
    
    #Extract the files from the folder with prefix = 'results'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    npz_files = list_npz_files(script_dir)
    processed_data = load_and_process_files(npz_files)

    # ##############################################################
    # plot_cohesive_energy(processed_data)
    # plot_potential_energy(processed_data)
    # plot_total_energy(processed_data)
    # plot_energy_comb(processed_data)

    # plot_lda_opening(processed_data)
    # plot_l2_norm(processed_data)
    # plot_mesh_conv(processed_data)
    # plot_all_stress_vs_displacement(processed_data) 
    # plot_disp_L2(processed_data)
# 
    # plot_displacement(processed_data)
    # plot_sep_dissipation(processed_data)
    plot_damage_combine(processed_data)
    # plot_all_coh_stress_vs_seperation(processed_data)
    # plot_all_damge_vs_imposed_disp(processed_data)
    # plot_all(processed_data)
    # Dark2_r, Accent,Blues,Paired,Purples,Spectral
    # copper,cubehelix,tab20

    ##############################################################
    # # part to execute functions vs plot for different fucntions
    # d_values = np.linspace(0.0, 1.0, 100)
    # module_name='functions'
    # module = importlib.import_module(module_name)
    # cls = getattr(module, 'hd_cohesive_quad_4_terms')
    # parameters = {'beta' : 0.5}
    # instance = cls(parameters)
    # # class_names = ['gd_cohesive_std']
    # # plot_func_values_vs_damage(class_names, d_values, additional_params = None)

################################################################