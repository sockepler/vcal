"""Keep EDA library paths in simulator children, separate from Python/Qt."""
import os


def simulation_environment():
    environment = os.environ.copy()
    if "VCAL_SIM_LD_LIBRARY_PATH" in environment:
        environment["LD_LIBRARY_PATH"] = environment["VCAL_SIM_LD_LIBRARY_PATH"]
    return environment
