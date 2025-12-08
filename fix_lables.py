import os

def fix_labels(directory):
    # Percorre todos os arquivos da pasta
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            changed = False
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(parts[0])
                    
                    # SE A CLASSE FOR 1, MUDA PARA 0
                    if class_id == 0:
                        parts[0] = "1" # Altera para 0
                        new_line = " ".join(parts) + "\n"
                        new_lines.append(new_line)
                        changed = True
                    else:
                        # Se já for 0 ou outra coisa, mantém igual
                        new_lines.append(line)
            
            # Só reescreve o arquivo se houve mudança
            if changed:
                with open(filepath, 'w') as f:
                    f.writelines(new_lines)
                print(f"Corrigido: {filename}")

# --- CONFIGURAÇÃO ---
# Coloque o caminho da sua pasta de labels aqui
# Exemplo: "meu_dataset/labels"
PATH_TO_LABELS = "/home/luiz-fonseca/.cache/kagglehub/datasets/wednesday233/corrosion-detect-dataset/versions/1/corrosion detect/labels" 

if os.path.exists(PATH_TO_LABELS):
    fix_labels(PATH_TO_LABELS)
    print("Processo finalizado.")
else:
    print("Pasta não encontrada.")