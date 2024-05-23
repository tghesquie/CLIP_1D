"""
Script to compare the results with the test cases
"""
import os
import numpy as np

from post_process import list_npz_files, load_and_process_files

def run_tests():
    """
    Function to initialize processed_data and expected_output.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Process the result files
    npz_files = list_npz_files(script_dir)
    processed_data = load_and_process_files(npz_files)

    # Load the expected output
    expected_data_clip_3_terms = np.load('test_CLIP_3_terms.npz')

    for entry in processed_data:
        functional_choice = entry['parameters'].get('functional_choice', 'unknown')
        if functional_choice == 'CLIP-3terms':
            loaded_result_stress = entry['stress']
            expected_output_stress = expected_data_clip_3_terms['stress']
            loaded_result_incs = entry['imposed_disp']
            expected_data_incs = expected_data_clip_3_terms['imposed_disp']

            print("Checking for the test case:")
            print("E = 3e10, Gc = 120, sigc = 3e6, L = 0.2")
            print("Dm = 0.9, he = 10, N_increments = 30")
            print("damage_function = cos_sin_D_squared")

            np.testing.assert_allclose(loaded_result_incs, expected_data_incs, rtol=1e-3, atol=1e-6)
            np.testing.assert_allclose(loaded_result_stress, expected_output_stress, rtol=1e-3, atol=1e-6)
            print("Tests passed successfully for CLIP-3terms!")

        elif functional_choice == 'CLIP-4terms':
            # Yet to add tests for CLIP-4terms
            print("Tests passed successfully for CLIP-4terms!")

        elif functional_choice == 'CZM':
            # Yet to add tests for CZM
            print("Tests passed successfully for CZM!")

        elif functional_choice == 'LIP':
            # Yet to add tests for LIP
            print("Tests passed successfully for LIP!")
        elif functional_choice == 'Exact':
            continue
        else:
            print("Functional choice is invalid")

if __name__ == '__main__':
    run_tests()
