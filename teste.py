import cv2
import numpy as np

def calcular_estatisticas_modelo(img_roi):
    """
    Calcula média e covariância inversa da região de interesse (amostra).
    """
    # Achatar a ROI para uma lista de pixels
    pixels = img_roi.reshape(-1, 3).astype(np.float32)
    
    # 1. Calcular Média
    mean = np.mean(pixels, axis=0)
    
    # 2. Calcular Covariância
    cov = np.cov(pixels, rowvar=False)
    
    # Adicionar ruído minúsculo à diagonal para evitar erro de matriz singular
    # caso a amostra seja muito uniforme (cor sólida perfeita)
    cov += np.eye(cov.shape[0]) * 1e-6
    
    # 3. Inversa da Covariância
    inv_cov = np.linalg.inv(cov)
    
    return mean, inv_cov

def calcular_mapa_distancia(img_completa, mean, inv_cov):
    """
    Calcula a distância de Mahalanobis para toda a imagem.
    Otimizado com operações vetoriais do Numpy.
    """
    h, w, c = img_completa.shape
    img_flat = img_completa.reshape(-1, 3).astype(np.float32)
    
    # Subtrair média
    diff = img_flat - mean
    
    # Calcular Mahalanobis: sqrt( diff * inv_cov * diff.T )
    # Passo 1: Multiplicação matricial (diff x inv_cov)
    temp = np.dot(diff, inv_cov)
    
    # Passo 2: Multiplicação elemento a elemento e soma
    dist_sq = np.sum(temp * diff, axis=1)
    
    # Passo 3: Raiz quadrada
    dist = np.sqrt(dist_sq)
    
    return dist.reshape(h, w)

def pipeline_deteccao(caminho_imagem):
    # 1. Carregar Imagem
    img_bgr = cv2.imread(caminho_imagem)
    if img_bgr is None:
        print(f"Erro: Não foi possível abrir a imagem {caminho_imagem}")
        return

    # Redimensionar se for muito grande (opcional, para caber na tela)
    if img_bgr.shape[1] > 1200:
        fator = 1200 / img_bgr.shape[1]
        img_bgr = cv2.resize(img_bgr, None, fx=fator, fy=fator)

    print("\n--- INSTRUÇÕES ---")
    print("1. Uma janela abrirá. Use o mouse para desenhar um retângulo sobre uma ÁREA DE CORROSÃO.")
    print("2. Pressione ESPAÇO ou ENTER para confirmar a seleção.")
    print("3. Pressione 'c' para cancelar a seleção e tentar de novo.")
    print("------------------\n")

    # 2. Seleção Manual da ROI (Exemplo de Corrosão)
    r = cv2.selectROI("Selecione Exemplo de Corrosao", img_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Selecione Exemplo de Corrosao")
    
    # r retorna (x, y, w, h). Se w ou h forem 0, usuário cancelou.
    if r[2] == 0 or r[3] == 0:
        print("Seleção cancelada.")
        return

    # Recortar a região de treino
    x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
    roi_bgr = img_bgr[y:y+h, x:x+w]

    # 3. Conversão de Espaço de Cor (LAB é melhor para percepção de cor)
    # L = Luminosidade, A/B = Canais de cor. Corrosão geralmente tem padrão forte em A/B.
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2Lab)

    # 4. Treinamento
    print("Processando estatísticas da amostra...")
    mean, inv_cov = calcular_estatisticas_modelo(roi_lab)

    # 5. Cálculo do Mapa de Distância (Parte pesada, fazemos apenas uma vez)
    print("Calculando distâncias em toda a imagem...")
    dist_map = calcular_mapa_distancia(img_lab, mean, inv_cov)

    # 6. Interface Interativa para ajuste do Threshold
    janela_nome = "Resultado (Aperte 'q' para sair)"
    cv2.namedWindow(janela_nome)
    
    # Criar Trackbar (Slider)
    # Valor inicial 30 (que representa 3.0), máximo 200 (20.0)
    cv2.createTrackbar("Sensibilidade", janela_nome, 30, 150, lambda x: None)

    while True:
        # Ler valor do slider
        val = cv2.getTrackbarPos("Sensibilidade", janela_nome)
        threshold = val / 10.0 # Converter int para float (ex: 30 -> 3.0)
        
        if threshold < 0.1: threshold = 0.1 # Evitar zero

        # Criar máscara binária baseada no threshold
        mask = (dist_map <= threshold).astype(np.uint8) * 255
        
        # Limpeza morfológica (remover ruídos pequenos)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask_limpa = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Criar visualização: Sobrepor vermelho onde foi detectado
        # Criar imagem vermelha
        vermelho = np.zeros_like(img_bgr)
        vermelho[:] = [0, 0, 255] # BGR
        
        # Misturar imagem original com a vermelha usando a máscara
        res_visual = img_bgr.copy()
        
        # Onde a mascara é branca, misturamos a cor vermelha
        locais_corrosao = mask_limpa > 0
        res_visual[locais_corrosao] = cv2.addWeighted(img_bgr[locais_corrosao], 0.7, vermelho[locais_corrosao], 0.3, 0)
        
        # Desenhar retângulo da amostra original para referência (em verde)
        cv2.rectangle(res_visual, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow(janela_nome, res_visual)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Salvar resultado
            cv2.imwrite("resultado_corrosao.png", res_visual)
            cv2.imwrite("mascara_corrosao.png", mask_limpa)
            print("Imagens salvas como 'resultado_corrosao.png' e 'mascara_corrosao.png'")

    cv2.destroyAllWindows()

# ======================================================
# CONFIGURAÇÃO
# ======================================================

# Substitua pelo caminho da sua imagem
CAMINHO_IMAGEM = "corrosion detect\images\image10.jpeg" 

# Tentar rodar (crie um arquivo dummy se não tiver imagem para testar o fluxo de erro)
if __name__ == "__main__":
    # Dica: Se quiser testar sem imagem, comente a linha abaixo e use o gerador sintético anterior,
    # mas o ideal é colocar o nome do seu arquivo jpg/png aqui.
    try:
        pipeline_deteccao(CAMINHO_IMAGEM)
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        print("Verifique se o caminho da imagem está correto.")