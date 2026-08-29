"""Run the promoted RTC-V3 versus FedAvg comparison matrix."""

import logging

from experiments.rtc_fedavg_comparison import *  # noqa: F401,F403
from experiments.rtc_fedavg_comparison import parse_args, run_comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_comparison(parse_args())
