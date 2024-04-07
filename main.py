import logging

from ods_mlops.log_set import init_logging


def main():
    logging.info("Run app")


if __name__ == "__main__":
    init_logging()
    main()
