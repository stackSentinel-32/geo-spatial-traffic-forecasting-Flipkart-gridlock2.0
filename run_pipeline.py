import config
from pipeline import run_pipeline


def main():
    run_pipeline(
        train_path=config.TRAIN_PATH,
        test_path=config.TEST_PATH,
        submit_dir=config.SUBMIT_DIR,
        v2_submission_path=config.V2_SUBMISSION_PATH,
    )


if __name__ == '__main__':
    main()
