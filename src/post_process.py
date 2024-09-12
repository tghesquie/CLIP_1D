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
                    'tot_disp_exp':tot_disp_exp
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

def energy_comp_plot(dt, max_steps, E_pot, Ep_str, E_kin, Ekin_str, E_dis, Edis_str):

    time = [i * dt for i in range(max_steps)]

    plt.plot(time, E_pot, label = 'Epot_Aka')
    plt.plot(time, Ep_str, label='Epot_test')
    plt.plot(time, E_kin, label = 'Ekin_Aka')
    plt.plot(time, Ekin_str, label = 'Ekin_test')
    plt.plot(time, E_dis, label = 'Edis_Aka')
    plt.plot(time, Edis_str, label = 'Edis_test')
    #linestyle='-', marker='o'
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time (s)', fontsize = 'large')
    plt.ylabel('Energy (Nm)', fontsize = 'large')
    plt.title('Energy comparison over time', fontweight = 'bold', fontsize = 'large')
    formatter = ticker.FormatStrFormatter('%.1e')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig('/home/ssshetty/Home/Main/Akantu/akantu/examples/c++/solid_mechanics_cohesive_model/test/len_bar/Algorithm_test/energy_comp.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    ################################################################
    #Extract the files from the folder with prefix = 'results'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    npz_files = list_npz_files(script_dir)
    processed_data = load_and_process_files(npz_files)

    ##############################################################

    plot_all_stress_vs_displacement(processed_data)
    # plot_all_dissipation(processed_data)
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
