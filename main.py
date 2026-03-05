# ko-embed/main.py
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(f"Project: {cfg.project.name}")
    
    # 1. 모델 인스턴스화 (설정 기반으로 자동 생성)
    # cfg.model에 정의된 _target_ 클래스를 찾아 파라미터를 주입하며 생성합니다.
    encoder = instantiate(cfg.model)
    
    # 2. 테스트 텍스트
    texts = ["어떻게 하면 좋을지 모르겠어", "도움이 필요해"]
    
    # 3. 임베딩 실행
    embeddings = encoder.encode(texts)
    print(f"Embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    main()