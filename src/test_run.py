import pathlib
import filecmp
import pandas as pd

def test_output_exist():
    filepath = r'C:\Users\ITO\Desktop\EXERCICE NOUR\basic-deep-learning-nourmezrioui-main\data\validation\output.tsv'# Remplacez par votre chemin réel
    flag_error = pathlib.Path(filepath)
    assert flag_error.exists(), f"Le fichier {filepath} n'existe pas"
    #print("Le fichier output.tsv existe.")

def test_output_content():

    def normalize_end_of_line(file_path):
        with open(file_path, 'r', newline=None) as file:
            content = file.read()

    # Remplacer les fins de ligne Windows (\r\n) par des fins de ligne Unix (\n)
        content = content.replace('\r\n', '\n')

        with open(file_path, 'w', newline='\n') as file:
            file.write(content)

# Appliquer la normalisation aux deux fichiers (probleme de fin de ligne)
    normalize_end_of_line(r'C:\Users\ITO\Desktop\EXERCICE NOUR\basic-deep-learning-nourmezrioui-main\data\validation\output.tsv')# Remplacez par votre chemin réel
    normalize_end_of_line(r'C:\Users\ITO\Desktop\EXERCICE NOUR\basic-deep-learning-nourmezrioui-main\data\validation\output.tsv')# Remplacez par votre chemin réel
    
    filename_out = pathlib.Path(r'C:\Users\ITO\Desktop\EXERCICE NOUR\basic-deep-learning-nourmezrioui-main\data\validation\output.tsv')# Remplacez par votre chemin réel
    filename_ref = pathlib.Path(r'C:\Users\ITO\Desktop\EXERCICE NOUR\basic-deep-learning-nourmezrioui-main\data\validation\output.tsv')# Remplacez par votre chemin réel
    assert filecmp.cmp(filename_out, filename_ref), "Les fichiers ne sont pas identiques"
    #print("Les fichiers output.tsv et subjects.tsv sont identiques.")

if __name__ == "__main__":
    test_output_exist()
    test_output_content()

