from src.fraud_detection_autoencoder import ExperimentConfig, run_experiment


def main() -> None:
    config = ExperimentConfig()
    result = run_experiment(config)
    print(result["summary_text"])


if __name__ == "__main__":
    main()

