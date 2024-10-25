""" Post Processing"""
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from functions import gd_cohesive_std

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

            elif functional_choice in ['CZM']:
                seperation = data['seperation']
                cohesive_damage =data['cohesive_damage']

            #elif functional_choice in ['LIP']:
         
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
                })

            elif  parameters.get('functional_choice') in ['CZM']:
                data_collection.append({
                    'filename': file,
                    'parameters': parameters,
                    'stress': stress,
                    'imposed_disp': imposed_disp,
                    'seperation':seperation,
                    'cohesive_damage':cohesive_damage,                
                })
            elif  parameters.get('functional_choice') in ['LIP']:
                data_collection.append({
                    'filename': file,
                    'parameters': parameters,
                    'stress': stress,
                    'imposed_disp': imposed_disp,                                   
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

    for entry in processed_data:       
        Dm = entry['parameters'].get('Dm', 'unknown')
        alpha = entry['parameters'].get('alpha', 'unknown')
        sigc = entry['parameters'].get('sigc', 'unknown')
        Gc = entry['parameters'].get('Gc', 'unknown')
        functional_choice = entry['parameters'].get('functional_choice','unknown')        
        #label = functional_choice if functional_choice in ['CZM', 'LIP', 'Exact'] else f"{functional_choice}, $D_m$={Dm}, $\\alpha$={alpha:.3f}"
        if functional_choice == 'CZM':
            label = f"{functional_choice} "
            color = "C1"
        
        elif functional_choice == 'LIP':
            label = f"{functional_choice}"
            color = "C2"
        
        elif functional_choice == 'Exact':
            label = f"{functional_choice}"
            color = "black"
        
        else :
            label = f"CLIP , $D_m$ = {Dm}"
            color = "C0"

        plt.plot(entry['imposed_disp'], entry['stress'], label=label, color = color)

    plt.title("Stress [$\sigma$] vs Imposed Displacement [$u_t$]", fontsize = 'large', fontweight = 'bold')
    plt.xlabel("Imposed Displacement [m]", fontsize = 'large')
    plt.ylabel("Stress [Pa] ", fontsize = 'large')
    plt.axhline(y=sigc, color='red', linestyle='--')
    plt.text((Gc*2)/(sigc), sigc, 'Stress limit', color='red', fontsize=10, va='bottom', ha='right')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_ticks))
    plt.legend(fontsize = 'large') 
    plt.grid(True) 
    plt.savefig('/home/ssshetty/Home/Main/Presentation/Graphs/hybrid_model/stress_vs_imposed_displacement_report.png',dpi = 300)
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
        if functional_choice in ['CZM', 'LIP', 'Exact']:
          continue

        label = f"{functional_choice}, $D_m$ = {Dm}, $\\alpha$ = {alpha:.3f}"
        axs[0,0].plot(entry['imposed_disp'], entry['coh_disp_act'] )
        axs[0,1].plot(entry['imposed_disp'], entry['bulk_disp_act'] )
        axs[0,2].plot(entry['imposed_disp'], entry['tot_disp_act'],label = label )
        axs[1,0].plot(entry['imposed_disp'], entry['coh_disp_exp'] )
        if functional_choice == 'CLIP-4terms':        
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
        if functional_choice in ['CZM', 'LIP', 'Exact']:
          continue
        Dm = entry['parameters'].get('Dm', 'unknown')
        Gc = entry['parameters'].get('Gc', 'unknown')
        sigc = entry['parameters'].get('sigc', 'unknown')
        wc = (2*Gc)/(sigc)
        seperation = entry['seperation']
        N_elements = entry['parameters'].get('N_elements', 'unknown')              
        jump_coh = [arr[int((N_elements - 1)/2)] for arr in seperation]        
        d_area = entry['stress'].copy()
        sig = [sigc,0]
        w = [0,wc]
        area = np.trapz(d_area,jump_coh)
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
    plt.show()

def plot_damage_combine(processed_data):
    plt.figure(figsize=(10, 6))
    for entry in processed_data :        
        Dm = entry['parameters'].get('Dm', 'unknown')
        alpha = entry['parameters'].get('alpha', 'unknown')        
        functional_choice = entry['parameters'].get('functional_choice','unknown')  
        if functional_choice in ['CZM', 'LIP', 'Exact']:
            continue
        ij = [int(len(entry['imposed_disp'])/7.5) ,int(len(entry['imposed_disp'])/5) ,int(len(entry['imposed_disp'])/2.5) , -1]    
        bulk = [ '0.25', '0.5','0.75', '' ]
        for i,b in zip(ij, bulk):
            if i < len(entry['cohesive_damage']) :
                d_data = entry['cohesive_damage'][i]
                bulk_data = entry['bulk_damage_overall'][i]
                plt.scatter( entry['parameters'].get('L', 'unknown')/2, max(d_data) )
                plt.plot(np.linspace(0,  entry['parameters'].get('L', 'unknown'), len(bulk_data)), bulk_data, label = f'$u_t = {b} \omega_c$')
                
        plt.axhline(y=Dm, color='red', linestyle='--')
        plt.text( entry['parameters'].get('L', 'unknown')*0.05, Dm, '$D_m$', color='red', fontsize=10, va='bottom', ha='right')
        plt.xlabel('Position along the bar [m]',fontsize = 'large')
        plt.ylabel('Cohesive and Bulk Damage', fontsize = 'large')
        plt.legend(fontsize = 'large')
        plt.title('Damage along the bar',fontsize = 'large')
        plt.grid(True)
        plt.show()

def plot_all_damge_vs_imposed_disp(processed_data):
    plt.figure(figsize=(10, 6)) 

    for entry in processed_data :

        functional_choice = entry['parameters'].get('functional_choice','unknown')        
        if functional_choice in ['CZM', 'LIP', 'Exact']:
          continue

        max_d_str = []
        damage = entry['cohesive_damage']
        max_d_str.extend([np.max(arr) for arr in damage])
        Dm = entry['parameters'].get('Dm', 'unknown')
        alpha = entry['parameters'].get('alpha', 'unknown')      
        label = f"{functional_choice}, $D_m$={Dm}, $\\alpha$={alpha:.3f}"

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
    plt.plot(time, E_dis_aka, label = 'Edis_Akantu', linestyle = '-.', color = 'black')
    for result in results :
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
        plt.plot(time, Edis_bulk_act_str,label = fr'Bulk, Dm = {result["inputs"].get("Dm")}', linestyle = '-.')
        plt.plot(time, Edis_coh_act_str,label = fr'Cohesive, Dm = {result["inputs"].get("Dm")}',linestyle = '--')
        plt.plot(time, Edis_total_act_str,label = fr'Total, Dm = {result["inputs"].get("Dm")}')

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
        plt.plot(time, tot, label = fr'Dm = {result["inputs"].get("Dm")}, $\ell_c = \frac{{L}}{{{result["inputs"].get("nlc")}}}$')
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

def energies_comparison_string(dt, *results):
    
    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_pot = result['potential_energy']
        plt.plot(time, E_pot, label = fr'Dm = {result["inputs"].get("Dm")}, $\ell_c = \frac{{L}}{{{result["inputs"].get("nlc")}}}$')
        # plt.plot(time, E_pot, label = fr'Dm = {result["inputs"].get("Dm")}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Potential Energy (J)', fontsize = 'large')
    plt.title('Potential energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/pot.png', dpi=300)
    plt.show()

    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_kin = result['kinetic_energy']
        plt.plot(time, E_kin, label = fr'Dm = {result["inputs"].get("Dm")}, $\ell_c = \frac{{L}}{{{result["inputs"].get("nlc")}}}$')
        # plt.plot(time, E_kin, label = fr'Dm = {result["inputs"].get("Dm")}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Kinetic Energy (J)', fontsize = 'large')
    plt.title('Kinetic energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/kin.png', dpi=300)
    plt.show()

    
    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_dis = result['dissipated_energy']
        Edis_bulk_str = result['dissipated_bulk_energy']
        Edis_coh_str = result['dissipated_coh_energy']
        plt.plot(time, E_dis,label = fr'Dm = {result["inputs"].get("Dm")}, $\ell_c = \frac{{L}}{{{result["inputs"].get("nlc")}}}$')
        plt.plot(time, Edis_bulk_str, label = fr'Bulk, Dm = {result["inputs"].get("Dm")}, $\ell_c = \frac{{L}}{{{result["inputs"].get("nlc")}}}$', linestyle = '--')
        plt.plot(time, Edis_coh_str,label = fr'Cohesive, Dm = {result["inputs"].get("Dm")}, $\ell_c = \frac{{L}}{{{result["inputs"].get("nlc")}}}$',linestyle = '--')

        # plt.plot(time, E_dis,label = fr'Dm = {result["inputs"].get("Dm")}')
        # plt.plot(time, Edis_bulk_str, label = fr'Bulk, Dm = {result["inputs"].get("Dm")}', linestyle = '--')
        # plt.plot(time, Edis_coh_str,label = fr'Cohesive, Dm = {result["inputs"].get("Dm")}',linestyle = '--')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Dissipated Energy (J)', fontsize = 'large')
    plt.title('Dissipated energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/dissip.png', dpi=300)
    plt.show()

    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_dis = result['dissipated_energy']
        Edis_bulk_str = result['dissipated_bulk_energy']
        Edis_coh_str = result['dissipated_coh_energy']

        E_dis_act = result['total_dissipation_actual']
        Edis_bulk_act_str = result['bulk_dissipation_actual']
        Edis_coh_act_str = result['coh_dissipation_actual']
        
        # plt.plot(time, E_dis,label = fr'Dm = {result["inputs"].get("Dm")}')
        # plt.plot(time, Edis_bulk_str, label = fr'Bulk, Dm = {result["inputs"].get("Dm")}')
        # plt.plot(time, Edis_coh_str,label = fr'Cohesive, Dm = {result["inputs"].get("Dm")}')

        #plt.plot(time, E_dis_act,label = fr'Dm = {result["inputs"].get("Dm")}', linestyle = '--')
        plt.plot(time, Edis_bulk_act_str, label = fr'Bulk, Dm = {result["inputs"].get("Dm")}', linestyle = '--')
        plt.plot(time, Edis_coh_act_str,label = fr'Cohesive, Dm = {result["inputs"].get("Dm")}',linestyle = '--')

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Dissipated Energy (J)', fontsize = 'large')
    plt.title('Dissipated energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/dissip.png', dpi=300)
    plt.show()

    
    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        E_rev = result['cohesive_energy']
        plt.plot(time, E_rev, label = fr'Dm = {result["inputs"].get("Dm")}, $\ell_c = \frac{{L}}{{{result["inputs"].get("nlc")}}}$')
        #plt.plot(time, E_rev, label = fr'Dm = {result["inputs"].get("Dm")}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Cohesive Energy (J)', fontsize = 'large')
    plt.title('Cohesive energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/coh.png', dpi=300)
    plt.show()

    for result in results :
        max_steps = result['inputs'].get('max_steps')
        time = [i * dt for i in range(max_steps)]
        tot = result['total_energy']
        plt.plot(time, tot, label = fr'Dm = {result["inputs"].get("Dm")}, $\ell_c = \frac{{L}}{{{result["inputs"].get("nlc")}}}$')
        #plt.plot(time, tot, label = fr'Dm = {result["inputs"].get("Dm")}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Total Energy (J)', fontsize = 'large')
    plt.title('Total Energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/tot.png', dpi=300)
    plt.show()

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
        plt.plot(Time, result['opening'], marker = 'o', label=f'Dm = {result["inputs"].get("Dm")}')

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

    for result in results :
        plt.plot(result['nodes_str'], np.array(result['velocity']).T)
        plt.plot(result['nodes_str_crack'], np.array(result['velocity_crack']).T)

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

if __name__ == "__main__":
    ################################################################
    #Extract the files from the folder with prefix = 'results'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    npz_files = list_npz_files(script_dir)
    processed_data = load_and_process_files(npz_files)

    ##############################################################
    plot_cohesive_energy(processed_data)
    plot_potential_energy(processed_data)
    plot_total_energy(processed_data)
    plot_energy_comb(processed_data)

    plot_lda_opening(processed_data)
    plot_all_stress_vs_displacement(processed_data)
    plot_all_dissipation(processed_data)
    plot_damage_combine(processed_data)
    plot_all_coh_stress_vs_seperation(processed_data)
    plot_all_damge_vs_imposed_disp(processed_data)
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
