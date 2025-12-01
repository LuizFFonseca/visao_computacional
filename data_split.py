import os
import shutil
import random

def split_dataset(img_source_dir, txt_source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Separa o dataset em train, val e test.
    
    Args:
        img_source_dir: Pasta onde estão as imagens originais.
        txt_source_dir: Pasta onde estão os txts originais.
        output_dir: Pasta onde o dataset organizado será salvo.
        train_ratio, val_ratio, test_ratio: Proporções (devem somar 1.0).
    """
    
    # 1. Verificações iniciais
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        print("Erro: As proporções devem somar 1.0 (100%)")
        return

    # Lista todas as imagens (filtra por extensões comuns)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in os.listdir(img_source_dir) if os.path.splitext(f)[1].lower() in extensions]
    
    # Embaralha para garantir aleatoriedade
    random.seed(42) # Seed fixa para reproduzir o mesmo split se rodar de novo
    random.shuffle(images)
    
    total_images = len(images)
    print(f"Total de imagens encontradas: {total_images}")

    # Calcula os índices de corte
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    # Cria a estrutura de pastas do YOLO
    subsets = ['train', 'val', 'test']
    folders = ['images', 'labels']
    
    for subset in subsets:
        for folder in folders:
            os.makedirs(os.path.join(output_dir, folder, subset), exist_ok=True)

    # Função auxiliar para copiar arquivos
    def copy_files(file_list, subset_name):
        print(f"Copiando {len(file_list)} arquivos para {subset_name}...")
        for img_file in file_list:
            # Caminhos de origem
            src_img = os.path.join(img_source_dir, img_file)
            
            # Define o nome do arquivo de label correspondente
            file_name_no_ext = os.path.splitext(img_file)[0]
            src_txt = os.path.join(txt_source_dir, file_name_no_ext + ".txt")

            # Verifica se o label existe antes de copiar
            if not os.path.exists(src_txt):
                print(f"Aviso: Label não encontrado para {img_file}. Pulando.")
                continue

            # Caminhos de destino
            dst_img = os.path.join(output_dir, 'images', subset_name, img_file)
            dst_txt = os.path.join(output_dir, 'labels', subset_name, file_name_no_ext + ".txt")

            # Copia os arquivos
            shutil.copy(src_img, dst_img)
            shutil.copy(src_txt, dst_txt)

    # 3. Executa a cópia baseada nos índices
    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print(f"\nProcesso concluído! Dataset salvo em: {output_dir}")
    print(f"Treino: {len(train_files)} | Validação: {len(val_files)} | Teste: {len(test_files)}")

# --- CONFIGURAÇÃO ---
# Altere os caminhos abaixo para as suas pastas
INPUT_IMAGES = "/home/luiz-fonseca/.cache/kagglehub/datasets/wednesday233/corrosion-detect-dataset/versions/1/corrosion detect/images"  # Onde estão suas fotos hoje
INPUT_LABELS = "/home/luiz-fonseca/.cache/kagglehub/datasets/wednesday233/corrosion-detect-dataset/versions/1/corrosion detect/labels"   # Onde estão seus txts hoje
OUTPUT_DATASET = "dataset_yolo_final"       # Onde será criado o novo dataset

# Execute a função
# Exemplo: 70% Treino, 20% Validação, 10% Teste
if __name__ == "__main__":
    # Cria pastas de exemplo se não existirem (apenas para evitar erro se você rodar direto)
    if os.path.exists(INPUT_IMAGES) and os.path.exists(INPUT_LABELS):
        split_dataset(INPUT_IMAGES, INPUT_LABELS, OUTPUT_DATASET, 0.7, 0.2, 0.1)
    else:
        print("Por favor, configure as variáveis INPUT_IMAGES e INPUT_LABELS no código.")