import os
import re

# Folder u kojem se nalaze fajlovi sa rezultatima
results_dir = "results"
output_file = "summary.csv"

# Header CSV fajla
with open(output_file, "w") as out:
    out.write("Instance,k,Population,Fitness,Time,Initialization\n")

    # Prolazak kroz sve fajlove u folderu
    for filename in os.listdir(results_dir):
        filepath = os.path.join(results_dir, filename)

        # Preskoci sve što nije običan fajl
        if not os.path.isfile(filepath):
            continue

        # Učitaj sadržaj fajla
        with open(filepath, "r") as f:
            content = f.read()

        # Regularni izrazi za pronalazak vrijednosti
        instance = re.search(r"Instance:\s*(.+)", content)
        k = re.search(r"k:\s*(\d+)", content)
        population = re.search(r"Population:\s*(\d+)", content)
        fitness = re.search(r"Fitness:\s*([\d.]+)", content)
        time_val = re.search(r"Time:\s*([\d.]+)", content)
        initialization = re.search(r"Initialization:\s*([\d.]+)", content)

        # Ekstrakcija vrijednosti (ako postoji)
        instance_val = instance.group(1).strip() if instance else ""
        k_val = k.group(1).strip() if k else ""
        pop_val = population.group(1).strip() if population else ""
        fit_val = fitness.group(1).strip() if fitness else ""
        time_val = time_val.group(1).strip() if time_val else ""
        init_val = initialization.group(1).strip() if initialization else ""

        # Zapiši u CSV fajl
        out.write(f"{instance_val},{k_val},{pop_val},{fit_val},{time_val},{init_val}\n")

print(f"✅ Rezultati su sakupljeni u '{output_file}'")

