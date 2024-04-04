import logging

import pytest
from ods_mlops.log_set import init_logging


@pytest.fixture(scope="session", autouse=True)
def init_logging_in_session():
    init_logging("./log_settings.yaml")


def test_logging(caplog):
    root_logger = logging.getLogger()
    text = "Test"
    root_logger.info(text)
    assert text in caplog.records[-1].message
    assert caplog.records[-1].levelno == logging.INFO
