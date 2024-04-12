import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

class PostProcess :

    def __init__(self,parameters,dumpers,exact_inc,exact_f):
        self.parameters = parameters
        self.dumpers = dumpers
        self.exact_inc = exact_inc
        self.exact_f = exact_f
        # self.comp_coh_disp = comp_coh_disp
        # self.comp_bulk_disp = comp_bulk_disp
        # self.alpha_plot = alpha_plot
        # self.Dm_plot = Dm_plot
        # self.alpha_values = alpha_values
        # self.Dm_values = Dm_values
        # self.choice_values = choice_values

    def format_x_ticks(self,value,_):
        return f"{value:.1e}"
    
    def plot_stress_ut(self):
        plt.figure()

        if len(self.dumpers) == 1:
            # Single simulation case
            dumper = self.dumpers[0]
            plt.plot(dumper.data["imposed_displacement"], dumper.data["force"], label='CLIP')
        else:
            # Multiple simulations case
            for i, dumper in enumerate(self.dumpers):
                
                alpha_index = i // len(self.parameters.Dm_values)
                Dm_index = i % len(self.parameters.Dm_values)

                alpha = self.parameters.alpha_values[alpha_index]
                Dm = self.parameters.Dm_values[Dm_index]

                label = fr'$\alpha=${alpha:.2f}, $D_m=${Dm:.2f}'
                
                plt.plot(dumper.data["imposed_displacement"], dumper.data["force"], label= label)

        plt.plot(self.exact_inc, self.exact_f, label='Exact Solution', color='black')
        plt.xlabel("Imposed Displacement [m]", fontsize='large')
        plt.ylabel("Stress [Pa] ", fontsize='large')
        plt.axhline(y=self.parameters.sigc, color='red', linestyle='--')
        plt.text(self.parameters.wc, self.parameters.sigc, 'Stress limit', color='red', fontsize=10, va='bottom', ha='right')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(self.format_x_ticks))
        plt.title("Stress [$\sigma$] vs Imposed Displacement [$u_t$]", fontsize='large')
        plt.legend(fontsize='large')
        plt.grid(True)
        plt.show()

    def plot_dissip_all(self):

        fig, axs = plt.subplots(2, 3, figsize=(12, 8))

        for i,dumper in enumerate(self.dumpers):

                alpha_index = i // len(self.parameters.Dm_values)
                Dm_index = i % len(self.parameters.Dm_values)

                alpha = self.parameters.alpha_values[alpha_index]
                Dm = self.parameters.Dm_values[Dm_index]
                label = fr'$\alpha=${alpha:.2f}, $D_m=${Dm:.2f}'

                axs[0,0].plot(dumper.data["imposed_displacement"], dumper.data["cohesive_dissip_expected"])
                axs[0,1].plot(dumper.data["imposed_displacement"], dumper.data["bulk_dissip_expected"])
                axs[0,2].plot(dumper.data["imposed_displacement"],dumper.data["total_dissip_expected"], label = label)

                axs[1,0].plot(dumper.data["imposed_displacement"],dumper.data["cohesive_dissip_actual"])
                axs[1,1].plot(dumper.data["imposed_displacement"], dumper.data["bulk_dissip_actual"])
                axs[1,2].plot(dumper.data["imposed_displacement"],dumper.data["total_dissip_actual"], label = label)

        for ax in axs.flat:
        
            ax.xaxis.set_major_formatter(FuncFormatter(self.format_x_ticks))
            ax.grid(True)

        for ax, col in zip(axs[0], ['Cohesive Dissipation', 'Bulk Dissipation', 'Total Dissipation']):
            ax.set_title(col, fontweight= 'bold', fontsize = 'large')
    

        axs[0,0].set_xlabel("Imposed displacement [m]")
        axs[0,1].set_xlabel("Imposed displacement [m]")
        axs[0,2].set_xlabel("Imposed displacement [m]")
        axs[1,0].set_xlabel("Imposed displacement [m]")
        axs[1,1].set_xlabel("Imposed displacement [m]")
        axs[1,2].set_xlabel("Imposed displacement [m]")

        axs[1,2].legend()
        axs[0,2].legend()

        for ax, row in zip(axs[:,0], ['Dissipation %']*2):
            ax.set_ylabel(row, fontweight= 'bold', fontsize = 'large')

        fig.text(0.01, 0.75, 'Expected', va='center', rotation='vertical', fontsize = 'large', fontweight = 'bold')
        fig.text(0.01, 0.26, 'Actual', va='center', rotation='vertical', fontsize = 'large', fontweight = 'bold')

        plt.tight_layout()
        plt.show()

    def plot_dissip_comp(self):
        for i in range(len(self.Dm_values)):
            for j, choice in enumerate(self.choice_values):
                comp_coh_disp_choice = self.comp_coh_disp[j][i]
                comp_bulk_disp_choice = self.comp_bulk_disp[j][i]
                Dm_value = self.Dm_values[i]
                
                # Plot comp_coh_disp vs Dm
                plt.figure(figsize=(8, 6))
                plt.plot(Dm_value, comp_coh_disp_choice, marker='o', label=f'Choice {choice}')
                plt.xlabel('Dm')
                plt.ylabel('Cohesive Dissipation')
                plt.title(f'Cohesive Dissipation vs Dm for Choice {choice}')
                plt.legend()
                plt.grid(True)
                plt.show()
                
                # Plot comp_bulk_disp vs Dm
                plt.figure(figsize=(8, 6))
                plt.plot(Dm_value, comp_bulk_disp_choice, marker='o', label=f'Choice {choice}')
                plt.xlabel('Dm')
                plt.ylabel('Bulk Dissipation')
                plt.title(f'Bulk Dissipation vs Dm for Choice {choice}')
                plt.legend()
                plt.grid(True)
                plt.show()




                                
if __name__ == '__main__':

    post_processes = []  
    #dumper_instance = main_run
    post_process = PostProcess(parameters, post_processes,exact_inc,exact_f,comp_coh_disp,comp_bulk_disp,alpha_plot,Dm_plot)
    post_process.plot_stress_ut()
    